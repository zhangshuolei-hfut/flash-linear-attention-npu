#!/usr/bin/env python3
"""Compare fused prepare_wy_repr_bwd with the old da + full path."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch_npu  # noqa: F401
import fla_npu  # noqa: F401


@dataclass(frozen=True)
class CompareResult:
    name: str
    passed: bool
    max_abs: float
    mean_abs: float
    rmse: float
    argmax: tuple[int, ...]


def _load_ct():
    """Load ct from the active Python environment."""

    try:
        import ct  # type: ignore

        if hasattr(ct, "isclose"):
            return ct
    except Exception as exc:
        print(f"[WARN] import ct failed: {exc!r}")
        return None
    return None


def prepare_cu_seqlens(T: int, L: int = 33) -> list[int]:
    """Build balanced varlen cu_seqlens ending at T."""

    if L < 2:
        raise ValueError("L must be at least 2.")
    cu_seqlens = [0]
    num_seq = L - 1
    for idx in range(1, num_seq + 1):
        cu_seqlens.append((T * idx) // num_seq)
    return cu_seqlens


def prepare_chunk_indices(cu_seqlens: Sequence[int], chunk_size: int) -> list[int]:
    """Build flat [seq_idx, chunk_idx, ...] chunk indices."""

    chunk_indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = int(cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx])
        for chunk_idx in range((seq_len + chunk_size - 1) // chunk_size):
            chunk_indices.extend([seq_idx, chunk_idx])
    return chunk_indices


def _rand(shape: Sequence[int], dtype: torch.dtype, scale: float = 0.08) -> torch.Tensor:
    return (torch.rand(tuple(shape), dtype=torch.float32) * scale).to(dtype)


def _sync() -> None:
    torch.npu.synchronize()


def _to_npu(inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(t.npu() for t in inputs)


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float, tuple[int, ...]]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0
    flat_idx = int(diff.argmax().item()) if diff.numel() else 0
    argmax = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_idx), diff.shape))
    return max_abs, mean_abs, rmse, argmax


def _compare_one(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    diff_thd: float,
    pct_thd: float,
    ct_module,
) -> CompareResult:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    max_abs, mean_abs, rmse, argmax = _error_stats(actual_cpu, expected_cpu)

    passed = max_abs <= diff_thd
    if ct_module is not None:
        try:
            ct_ret = ct_module.isclose(actual_cpu, expected_cpu, diff_thd=diff_thd, pct_thd=pct_thd)
            if isinstance(ct_ret, bool):
                passed = ct_ret
        except Exception as exc:
            print(f"[WARN] ct.isclose failed for {name}: {exc!r}")

    print(
        f"[{name}] max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, "
        f"rmse={rmse:.6e}, argmax={argmax}, passed={passed}"
    )
    if actual_cpu.numel():
        print(
            f"    fused={actual_cpu[argmax].item():.8e}, "
            f"split={expected_cpu[argmax].item():.8e}"
        )
    return CompareResult(name, passed, max_abs, mean_abs, rmse, argmax)


def test_prepare_wy_repr_bwd(
    B: int,
    HK: int,
    HV: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    ktype,
    btype,
    cu_seqlens=None,
    chunk_indices=None,
    seed: int = 0,
    diff_thd: float = 1e-2,
    pct_thd: float = 1e-2,
) -> dict[str, CompareResult]:
    """Run fused op and old da + full path, then compare four public outputs."""

    if HV % HK != 0:
        raise ValueError(f"HV ({HV}) must be a positive multiple of HK ({HK}).")

    torch.manual_seed(seed)
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    k = _rand((B, HK, T, K), ktype)
    v = _rand((B, HV, T, V), ktype)
    beta = _rand((B, HV, T), btype, scale=0.05)
    A = _rand((B, HV, T, chunk_size), ktype, scale=0.04)
    dw = _rand((B, HV, T, K), ktype)
    du = _rand((B, HV, T, V), ktype)
    g = _rand((B, HV, T), btype, scale=0.03)

    k_n, v_n, beta_n, A_n, dw_n, du_n, g_n = _to_npu((k, v, beta, A, dw, du, g))

    print(
        "case:",
        f"B={B}, HK={HK}, HV={HV}, T={T}, K={K}, V={V}, chunk_size={chunk_size},",
        f"ktype={ktype}, btype={btype}, varlen={cu_seqlens is not None}",
    )
    if cu_seqlens is not None:
        print(f"varlen: seq_num={len(cu_seqlens) - 1}, chunk_num={len(chunk_indices) // 2}")

    fused = torch.ops.npu.npu_prepare_wy_repr_bwd(
        k_n,
        v_n,
        beta_n,
        A_n,
        dw_n,
        du_n,
        g_n,
        chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    _sync()

    dA_split = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k_n,
        v_n,
        beta_n,
        A_n,
        dw_n,
        du_n,
        g_n,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    split = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
        k_n,
        v_n,
        beta_n,
        A_n,
        dA_split,
        dw_n,
        du_n,
        g_n,
        chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    _sync()

    ct_module = _load_ct()
    if ct_module is None:
        print("[WARN] ct.isclose is unavailable; using built-in error statistics only.")
    else:
        print(f"ct loaded from: {getattr(ct_module, '__file__', '<unknown>')}")

    results = {}
    for name, actual, expected in zip(("dk", "dv", "dbeta", "dg"), fused, split):
        results[name] = _compare_one(
            name,
            actual,
            expected,
            diff_thd=diff_thd,
            pct_thd=pct_thd,
            ct_module=ct_module,
        )

    failed = [name for name, result in results.items() if not result.passed]
    if failed:
        raise AssertionError(f"fused vs split mismatch: {failed}")
    print("[PASS] fused outputs match da + full outputs.")
    return results


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fused prepare_wy_repr_bwd with da + full.")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--HK", type=int, default=32)
    parser.add_argument("--HV", type=int, default=32)
    parser.add_argument("--T", type=int, default=65536)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--ktype", default="fp16")
    parser.add_argument("--btype", default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--seq-num", type=int, default=32)
    parser.add_argument("--diff-thd", type=float, default=1e-2)
    parser.add_argument("--pct-thd", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cu_seqlens = None
    chunk_indices = None
    if args.varlen:
        cu_seqlens = prepare_cu_seqlens(args.T, args.seq_num + 1)
        chunk_indices = prepare_chunk_indices(cu_seqlens, args.chunk_size)

    test_prepare_wy_repr_bwd(
        B=args.B,
        HK=args.HK,
        HV=args.HV,
        T=args.T,
        K=args.K,
        V=args.V,
        chunk_size=args.chunk_size,
        ktype=_dtype(args.ktype),
        btype=_dtype(args.btype),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        seed=args.seed,
        diff_thd=args.diff_thd,
        pct_thd=args.pct_thd,
    )


if __name__ == "__main__":
    main()

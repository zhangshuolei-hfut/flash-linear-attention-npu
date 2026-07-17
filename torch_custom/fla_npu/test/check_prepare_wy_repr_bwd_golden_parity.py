#!/usr/bin/env python3
"""Check fused prepare_wy_repr_bwd CPU golden against old da + full goldens.

This is a small-shape formula parity check. It imports the existing test
modules, builds deterministic CPU inputs, computes dA with the old da golden,
feeds that dA into the old full golden, and compares the stitched result with
the fused public golden.
"""

from __future__ import annotations

import argparse
import sys
import types
from dataclasses import dataclass
from typing import Optional

import torch

if "torch_npu" not in sys.modules:
    sys.modules["torch_npu"] = types.ModuleType("torch_npu")
if "fla_npu" not in sys.modules:
    sys.modules["fla_npu"] = types.ModuleType("fla_npu")
if not hasattr(torch, "npu"):
    torch.npu = types.SimpleNamespace(  # type: ignore[attr-defined]
        set_device=lambda *_args, **_kwargs: None,
        set_compile_mode=lambda *_args, **_kwargs: None,
        config=types.SimpleNamespace(allow_internal_format=False),
    )

import test_npu_prepare_wy_repr_bwd as fused
import test_npu_prepare_wy_repr_bwd_da as da
import test_npu_prepare_wy_repr_bwd_full as full


@dataclass(frozen=True)
class ParityCase:
    name: str
    B: int
    H: int
    T: int
    K: int
    V: int
    chunk_size: int
    seed: int
    cu_seqlens: Optional[tuple[int, ...]] = None


CASES = [
    ParityCase(
        name="fixed_one_chunk",
        B=1,
        H=1,
        T=5,
        K=4,
        V=6,
        chunk_size=8,
        seed=1,
    ),
    ParityCase(
        name="fixed_two_chunks_tail",
        B=2,
        H=2,
        T=9,
        K=3,
        V=5,
        chunk_size=4,
        seed=2,
    ),
    ParityCase(
        name="varlen_multi_seq_tail",
        B=1,
        H=2,
        T=10,
        K=3,
        V=4,
        chunk_size=4,
        seed=3,
        cu_seqlens=(0, 3, 10),
    ),
]


def make_inputs(case: ParityCase) -> tuple[tuple[torch.Tensor, ...], Optional[list[int]], Optional[list[int]], int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    def randn(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float64) * scale

    k = randn((case.B, case.H, case.T, case.K), 0.17)
    v = randn((case.B, case.H, case.T, case.V), 0.13)
    beta = randn((case.B, case.H, case.T), 0.11)
    A = randn((case.B, case.H, case.T, case.chunk_size), 0.07)
    dw = randn((case.B, case.H, case.T, case.K), 0.19)
    du = randn((case.B, case.H, case.T, case.V), 0.23)
    g = randn((case.B, case.H, case.T), 0.05)

    if case.cu_seqlens is None:
        return (k, v, beta, A, dw, du, g), None, None, (case.T + case.chunk_size - 1) // case.chunk_size

    cu_seqlens = list(case.cu_seqlens)
    chunk_indices = fused.build_chunk_indices(cu_seqlens, case.chunk_size)
    return (k, v, beta, A, dw, du, g), cu_seqlens, chunk_indices, len(chunk_indices) // 2


def old_split_golden(
    inputs: tuple[torch.Tensor, ...],
    chunk_size: int,
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
    nt: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k, v, beta, A, dw, du, g = inputs
    B, H, T, K = k.shape
    V = v.shape[-1]

    dA = da.compute_dA_cpu(
        A,
        dw,
        g,
        beta,
        k,
        v,
        du,
        chunk_indices,
        cu_seqlens,
        B,
        H,
        T,
        K,
        chunk_size,
        nt,
    )
    dk = full.compute_dk_golden_high_precision(A, dw, g, beta, dA, k, cu_seqlens, chunk_indices, B, H, H, T, K, chunk_size, nt)
    dv = full.compute_dv_golden_high_precision(A, du, beta, cu_seqlens, chunk_indices, B, H, T, V, chunk_size, nt)
    dbeta = full.compute_dbeta_golden_high_precision(
        A,
        dw,
        g,
        beta,
        dA,
        k,
        v,
        du,
        cu_seqlens,
        chunk_indices,
        B,
        H,
        H,
        T,
        K,
        chunk_size,
        nt,
    )
    dg = full.compute_dg_golden_high_precision(A, dw, g, beta, dA, k, cu_seqlens, chunk_indices, B, H, H, T, K, chunk_size, nt)
    return dk, dv, dbeta, dg


def compare_case(case: ParityCase, *, rtol: float, atol: float) -> bool:
    inputs, cu_seqlens, chunk_indices, nt = make_inputs(case)
    expected = old_split_golden(inputs, case.chunk_size, cu_seqlens, chunk_indices, nt)
    actual = fused.prepare_wy_repr_bwd_golden(
        *inputs,
        case.chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    print(
        f"CASE {case.name}: B={case.B} H={case.H} T={case.T} K={case.K} "
        f"V={case.V} chunk={case.chunk_size} cu_seqlens={cu_seqlens}",
        flush=True,
    )
    passed = True
    for name, got, ref in zip(("dk", "dv", "dbeta", "dg"), actual, expected):
        got64 = got.to(torch.float64)
        ref64 = ref.to(torch.float64)
        diff = (got64 - ref64).abs()
        flat_idx = int(diff.argmax().item()) if diff.numel() else 0
        idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx), diff.shape)) if diff.numel() else ()
        close = torch.allclose(got64, ref64, rtol=rtol, atol=atol)
        max_abs = float(diff[idx].item()) if diff.numel() else 0.0
        print(f"  {name}: close={close} max_abs={max_abs:.6e} idx={idx}", flush=True)
        if not close:
            print(f"    fused={float(got64[idx]):.12e} split={float(ref64[idx]):.12e}", flush=True)
            passed = False
    print(f"  RESULT {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="prepare_wy_repr_bwd CPU golden parity check")
    parser.add_argument("--case", action="append", dest="cases", help="case name to run; repeatable")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = CASES
    if args.cases:
        names = set(args.cases)
        selected = [case for case in CASES if case.name in names]
        missing = names - {case.name for case in selected}
        if missing:
            raise ValueError(f"unknown case(s): {sorted(missing)}")

    ok = True
    for case in selected:
        ok = compare_case(case, rtol=args.rtol, atol=args.atol) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""prepare_wy_repr_bwd 新算子测试入口。

本脚本依据 `docs/prepare_wy_repr_bwd_design.md` 和
`prepare_wy_repr_bwd_flow_npu_affinity.drawio` 中的总 DAG 公式生成 CPU
golden，并通过 `torch.ops.npu.npu_prepare_wy_repr_bwd` 调用新算子做对比。

说明：
1. 当前阶段只要求生成测试脚本，不要求算子已经跑通；因此提供
   `--golden-only` 模式，便于先检查参考公式和用例构造。
2. 这里不再沿用旧 `da/full` 拆分脚本的命名与中间接口，而是直接
   面向新算子的四个公开输出：dk、dv、dbeta、dg。
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch


try:
    import torch_npu  # noqa: F401
    import fla_npu  # noqa: F401

    HAS_NPU_RUNTIME = hasattr(torch, "npu")
except Exception:
    HAS_NPU_RUNTIME = False


@dataclass(frozen=True)
class PrepareWyReprBwdCase:
    """单条测试用例配置。

    设计文档当前限定 A2/A3，V 只覆盖 128/256，chunk_size 只覆盖 64/128。
    `seq_lens=None` 表示固定长度场景；否则表示 varlen 场景，此时 B 必须为 1。
    """

    name: str
    B: int
    HK: int
    HV: int
    T: int
    K: int
    V: int
    chunk_size: int
    data_dtype: torch.dtype
    gate_dtype: torch.dtype
    seq_lens: Optional[tuple[int, ...]] = None
    seed: int = 20260711


def _configure_npu() -> None:
    """配置 NPU 测试环境。

    旧测试脚本统一使用 TEST_DEVICE_ID 环境变量，这里保持一致。
    若本地没有 torch_npu/fla_npu，仅允许 `--golden-only` 路径继续运行。
    """

    if not HAS_NPU_RUNTIME:
        return
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def build_cu_seqlens(seq_lens: Sequence[int]) -> list[int]:
    """根据每段真实长度生成 Python list 形式的 cu_seqlens。

    新算子的 Python 绑定使用 `int[]?`，因此这里直接返回 list[int]，
    避免 Tensor 与 OptionalIntArrayRef 之间的类型歧义。
    """

    cu_seqlens = [0]
    for seq_len in seq_lens:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        cu_seqlens.append(cu_seqlens[-1] + int(seq_len))
    return cu_seqlens


def build_chunk_indices(cu_seqlens: Sequence[int], chunk_size: int) -> list[int]:
    """生成扁平化 chunk_indices。

    varlen 场景中每个 chunk 用 `(seq_idx, chunk_idx_in_seq)` 定位。
    算子接口期望扁平 list，所以返回格式为：
    `[seq0, chunk0, seq0, chunk1, seq1, chunk0, ...]`。
    """

    chunk_indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = int(cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx])
        for chunk_idx in range(_ceil_div(seq_len, chunk_size)):
            chunk_indices.extend([seq_idx, chunk_idx])
    return chunk_indices


def iter_chunk_ranges(
    B: int,
    T: int,
    chunk_size: int,
    cu_seqlens: Optional[Sequence[int]],
    chunk_indices: Optional[Sequence[int]],
) -> Iterable[tuple[int, int, int]]:
    """按设计文档中的 chunk 规则枚举 `(batch, bos, eos)`。

    固定长度：每个 batch 直接按 `T/chunk_size` 切分。
    变长：当前设计要求 B=1，通过 `cu_seqlens + chunk_indices` 找到
    全局 token 范围。
    """

    if cu_seqlens is None:
        for batch in range(B):
            for chunk_idx in range(_ceil_div(T, chunk_size)):
                bos = chunk_idx * chunk_size
                eos = min(bos + chunk_size, T)
                yield batch, bos, eos
        return

    if B != 1:
        raise ValueError("varlen prepare_wy_repr_bwd requires B == 1")
    if chunk_indices is None or len(chunk_indices) % 2 != 0:
        raise ValueError("varlen prepare_wy_repr_bwd requires flat chunk_indices")

    for pair_idx in range(len(chunk_indices) // 2):
        seq_idx = int(chunk_indices[pair_idx * 2])
        local_chunk_idx = int(chunk_indices[pair_idx * 2 + 1])
        seq_bos = int(cu_seqlens[seq_idx])
        seq_eos = int(cu_seqlens[seq_idx + 1])
        bos = seq_bos + local_chunk_idx * chunk_size
        eos = min(bos + chunk_size, seq_eos)
        yield 0, bos, eos


def validate_case(case: PrepareWyReprBwdCase) -> None:
    """检查测试用例是否落在当前算子设计约束内。"""

    if case.HV % case.HK != 0:
        raise ValueError(f"{case.name}: HV must be divisible by HK")
    if case.V not in (128, 256):
        raise ValueError(f"{case.name}: V must be 128 or 256")
    if case.chunk_size not in (64, 128):
        raise ValueError(f"{case.name}: chunk_size must be 64 or 128")
    if case.seq_lens is not None:
        if case.B != 1:
            raise ValueError(f"{case.name}: varlen case requires B == 1")
        if sum(case.seq_lens) != case.T:
            raise ValueError(f"{case.name}: sum(seq_lens) must equal T")


def make_inputs(
    case: PrepareWyReprBwdCase,
) -> tuple[tuple[torch.Tensor, ...], Optional[list[int]], Optional[list[int]]]:
    """构造一组稳定、数值范围较小的输入。

    g 会参与 exp 计算，随机范围刻意压小，避免 golden 中指数放大后
    淹没真实公式错误。A/dw/du/k/v/beta 也使用小尺度随机数，便于后续
    精度定位时观察输出差异。
    """

    validate_case(case)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    def randn(shape: Sequence[int], dtype: torch.dtype, scale: float) -> torch.Tensor:
        data = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
        return data.to(dtype)

    k = randn((case.B, case.HK, case.T, case.K), case.data_dtype, 0.08)
    v = randn((case.B, case.HV, case.T, case.V), case.data_dtype, 0.08)
    beta = randn((case.B, case.HV, case.T), case.gate_dtype, 0.05)
    A = randn((case.B, case.HV, case.T, case.chunk_size), case.data_dtype, 0.04)
    dw = randn((case.B, case.HV, case.T, case.K), case.data_dtype, 0.08)
    du = randn((case.B, case.HV, case.T, case.V), case.data_dtype, 0.08)
    g = randn((case.B, case.HV, case.T), case.gate_dtype, 0.03)

    if case.seq_lens is None:
        return (k, v, beta, A, dw, du, g), None, None

    cu_seqlens = build_cu_seqlens(case.seq_lens)
    chunk_indices = build_chunk_indices(cu_seqlens, case.chunk_size)
    return (k, v, beta, A, dw, du, g), cu_seqlens, chunk_indices


def prepare_wy_repr_bwd_golden(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    *,
    cu_seqlens: Optional[Sequence[int]] = None,
    chunk_indices: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按设计文档公式计算 `dk/dv/dbeta/dg`。

    关键约定：
    - 每个 value head `hv` 读取 `hk = hv // (HV // HK)` 对应的 k head。
    - `dk` 对同一个 hk 的多个 value-head 贡献累加。
    - `dv/dbeta/dg` 按 value head 独立写回。
    - golden 全部转为 float64 计算，最后再 cast 到各输出约定 dtype。
    """

    B, HK, T, K = k.shape
    _, HV, _, V = v.shape
    heads_per_k = HV // HK

    k64 = k.cpu().to(torch.float64)
    v64 = v.cpu().to(torch.float64)
    beta64 = beta.cpu().to(torch.float64)
    A64 = A.cpu().to(torch.float64)
    dw64 = dw.cpu().to(torch.float64)
    du64 = du.cpu().to(torch.float64)
    g64 = g.cpu().to(torch.float64)

    dk_acc = torch.zeros((B, HK, T, K), dtype=torch.float64)
    dv_acc = torch.zeros((B, HV, T, V), dtype=torch.float64)
    dbeta_acc = torch.zeros((B, HV, T), dtype=torch.float64)
    dg_acc = torch.zeros((B, HV, T), dtype=torch.float64)

    for batch, bos, eos in iter_chunk_ranges(B, T, chunk_size, cu_seqlens, chunk_indices):
        C = eos - bos
        if C <= 0:
            continue

        # 严格下三角 causal mask，对应流程图中的 lower-triangular where。
        causal_mask = torch.tril(torch.ones((C, C), dtype=torch.bool), diagonal=-1)

        for hv in range(HV):
            hk = hv // heads_per_k

            b_k = k64[batch, hk, bos:eos, :]
            b_v = v64[batch, hv, bos:eos, :]
            b_b = beta64[batch, hv, bos:eos]
            b_A = A64[batch, hv, bos:eos, :C]
            b_dw = dw64[batch, hv, bos:eos, :]
            b_du = du64[batch, hv, bos:eos, :]
            b_g = g64[batch, hv, bos:eos]

            b_g_exp = torch.exp(b_g)
            b_bg = b_b * b_g_exp
            b_kbg = b_k * b_bg[:, None]
            b_vb = b_v * b_b[:, None]

            b_dA0 = b_dw @ b_kbg.T
            b_dA1 = b_du @ b_vb.T
            b_dA2 = b_dA0 + b_dA1
            b_dA3 = torch.where(causal_mask, b_dA2, torch.zeros_like(b_dA2))
            b_dA4 = b_dA3 @ b_A.T
            b_dA5 = b_A.T @ b_dA4
            b_g_sub_exp = torch.exp(b_g[:, None] - b_g[None, :])
            b_dA6 = -b_dA5 * b_g_sub_exp
            b_dA = torch.where(causal_mask, b_dA6, torch.zeros_like(b_dA6))

            b_dvb = b_A.T @ b_du
            b_dv = b_dvb * b_b[:, None]

            b_dkb = b_dA @ b_k
            b_kb = b_k * b_b[:, None]
            b_dkbg = b_A.T @ b_dw
            b_dkbgk = b_dkbg * b_k

            b_dk = b_dA.T @ b_kb
            b_dk = b_dk + b_dkb * b_b[:, None]
            b_dk = b_dk + b_dkbg * b_bg[:, None]

            b_db = (b_dkb * b_k).sum(dim=1)
            b_db = b_db + (b_dkbgk * b_g_exp[:, None]).sum(dim=1)
            b_db = b_db + (b_dvb * b_v).sum(dim=1)

            b_kk = b_k @ b_k.T
            b_A_beta = b_kk * b_b[:, None]
            b_AdA = b_dA * b_A_beta
            b_dg = (b_dkbgk * b_bg[:, None]).sum(dim=1)
            b_dg = b_dg + b_AdA.sum(dim=1) - b_AdA.sum(dim=0)

            dk_acc[batch, hk, bos:eos, :] += b_dk
            dv_acc[batch, hv, bos:eos, :] = b_dv
            dbeta_acc[batch, hv, bos:eos] = b_db
            dg_acc[batch, hv, bos:eos] = b_dg

    return (
        dk_acc.to(dtype=k.dtype),
        dv_acc.to(dtype=v.dtype),
        dbeta_acc.to(dtype=beta.dtype),
        dg_acc.to(dtype=g.dtype),
    )


def run_npu_operator(
    inputs: tuple[torch.Tensor, ...],
    chunk_size: int,
    *,
    cu_seqlens: Optional[Sequence[int]] = None,
    chunk_indices: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """调用新公开接口，并把结果搬回 CPU 便于和 golden 对比。"""

    if not HAS_NPU_RUNTIME:
        raise RuntimeError("torch_npu/fla_npu is not available; use --golden-only")

    k, v, beta, A, dw, du, g = (tensor.npu() for tensor in inputs)
    dk, dv, dbeta, dg = torch.ops.npu.npu_prepare_wy_repr_bwd(
        k,
        v,
        beta,
        A,
        dw,
        du,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    torch.npu.synchronize()
    return dk.cpu(), dv.cpu(), dbeta.cpu(), dg.cpu()


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """给不同输出 dtype 选择较宽松的初始阈值。

    当前算子还处于生成阶段，阈值只作为脚本默认值；后续跑通后可结合
    ATK 双标杆统计结果再收紧。
    """

    if dtype == torch.float32:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 8e-2, 8e-2
    return 5e-2, 5e-2


def assert_close_outputs(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """逐输出比较，并在失败时打印最大误差位置。"""

    for name, got, ref in zip(("dk", "dv", "dbeta", "dg"), actual, expected):
        rtol, atol = _tolerance(ref.dtype)
        try:
            torch.testing.assert_close(
                got.to(torch.float32),
                ref.to(torch.float32),
                rtol=rtol,
                atol=atol,
                check_dtype=False,
            )
        except AssertionError:
            diff = (got.to(torch.float32) - ref.to(torch.float32)).abs()
            flat_idx = int(diff.argmax().item())
            idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx), diff.shape))
            print(f"[FAIL] {name}: max_abs_diff={diff[idx].item():.6e}, idx={idx}")
            print(f"       actual={got[idx].item():.6e}, expected={ref[idx].item():.6e}")
            raise


def default_cases() -> list[PrepareWyReprBwdCase]:
    """覆盖当前设计文档中的主要路径。

    - fixed_fp16_gqa_v128: 固定长度 + GQA 分组 + V=128。
    - fixed_bf16_gate_fp32_v256: bf16 数据 + fp32 beta/g + V=256。
    - fixed_fp16_chunk128: chunk_size=128 尾块路径。
    - varlen_fp16_multi_seq: B=1 变长，多段序列、多 chunk。
    """

    return [
        PrepareWyReprBwdCase(
            name="fixed_fp16_gqa_v128",
            B=1,
            HK=2,
            HV=4,
            T=96,
            K=128,
            V=128,
            chunk_size=64,
            data_dtype=torch.float16,
            gate_dtype=torch.float16,
            seed=11,
        ),
        PrepareWyReprBwdCase(
            name="fixed_bf16_gate_fp32_v256",
            B=1,
            HK=1,
            HV=2,
            T=80,
            K=128,
            V=256,
            chunk_size=64,
            data_dtype=torch.bfloat16,
            gate_dtype=torch.float32,
            seed=22,
        ),
        PrepareWyReprBwdCase(
            name="fixed_fp16_chunk128",
            B=1,
            HK=2,
            HV=2,
            T=129,
            K=128,
            V=256,
            chunk_size=128,
            data_dtype=torch.float16,
            gate_dtype=torch.float16,
            seed=33,
        ),
        PrepareWyReprBwdCase(
            name="varlen_fp16_multi_seq",
            B=1,
            HK=2,
            HV=4,
            T=185,
            K=128,
            V=128,
            chunk_size=64,
            data_dtype=torch.float16,
            gate_dtype=torch.float16,
            seq_lens=(31, 65, 89),
            seed=44,
        ),
    ]


def select_cases(names: Optional[Sequence[str]]) -> list[PrepareWyReprBwdCase]:
    cases = default_cases()
    if not names:
        return cases
    selected = []
    name_set = set(names)
    for case in cases:
        if case.name in name_set:
            selected.append(case)
    missing = name_set - {case.name for case in selected}
    if missing:
        raise ValueError(f"unknown case(s): {sorted(missing)}")
    return selected


def run_case(case: PrepareWyReprBwdCase, golden_only: bool) -> None:
    inputs, cu_seqlens, chunk_indices = make_inputs(case)
    expected = prepare_wy_repr_bwd_golden(
        *inputs,
        case.chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    print(
        f"[GOLDEN] {case.name}: "
        f"B={case.B}, HK={case.HK}, HV={case.HV}, T={case.T}, "
        f"K={case.K}, V={case.V}, chunk_size={case.chunk_size}, "
        f"data_dtype={case.data_dtype}, gate_dtype={case.gate_dtype}"
    )

    if golden_only:
        for name, tensor in zip(("dk", "dv", "dbeta", "dg"), expected):
            print(f"         {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        return

    actual = run_npu_operator(
        inputs,
        case.chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    assert_close_outputs(actual, expected)
    print(f"[PASS]   {case.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="prepare_wy_repr_bwd generated tests")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="只运行指定用例名，可重复传入；默认运行全部用例。",
    )
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="只生成 CPU golden，不调用 NPU 算子。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_npu()
    if not args.golden_only and not HAS_NPU_RUNTIME:
        raise RuntimeError("torch_npu/fla_npu is not available; rerun with --golden-only")

    for case in select_cases(args.cases):
        run_case(case, args.golden_only)


if __name__ == "__main__":
    main()

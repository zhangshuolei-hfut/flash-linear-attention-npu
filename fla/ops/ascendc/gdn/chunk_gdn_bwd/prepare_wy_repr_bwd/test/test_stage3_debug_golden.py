import argparse
import contextlib
import gc
import importlib
import io
import random
import time
from dataclasses import dataclass
from typing import Iterable

import ct
import torch
import torch_npu

from fla_npu.ops import ascendc as fla_ascendc


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def release_aclnn_keepalive():
    try:
        runtime_mod = importlib.import_module("fla_npu.ops.ascendc._runtime")
        runtime_mod._RECENT_LAUNCH_STORAGE.clear()
    except Exception:
        pass


@dataclass(frozen=True)
class Stage3Case:
    name: str
    B: int
    KH: int
    VH: int
    T: int
    K: int
    V: int
    chunk_size: int
    ktype: str
    gtype: str
    cu_seqlens_len: int | None = None


@dataclass(frozen=True)
class TaskRange:
    batch: int
    begin: int
    end: int


def prepare_cu_seqlens(T: int, L: int, seed: int = 42) -> list[int]:
    if T < 1:
        raise ValueError("T must be at least 1.")
    if L < 2 or L > T + 1:
        raise ValueError(f"L must satisfy 2 <= L <= T + 1, got L={L}, T={T}.")
    random.seed(seed)
    if L == 2:
        return [0, T]
    middle_points = random.sample(range(1, T), L - 2)
    middle_points.sort()
    return [0] + middle_points + [T]


def prepare_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


FULL_GDN_CASES: tuple[Stage3Case, ...] = (
    Stage3Case("F1", 64, 8, 8, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage3Case("F2", 32, 16, 16, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage3Case("F3", 16, 32, 32, 4096, 128, 128, 128, "fp16", "fp32"),
    Stage3Case("F4", 8, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage3Case("F5", 128, 4, 4, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage3Case("F6", 64, 8, 8, 2048, 128, 128, 64, "bf16", "fp32"),
    Stage3Case("F7", 32, 16, 16, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage3Case("F8", 16, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage3Case("F9", 64, 8, 8, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage3Case("F10", 32, 16, 16, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage3Case("F11", 16, 32, 32, 16384, 128, 128, 64, "fp16", "fp32"),
    Stage3Case("F12", 8, 32, 32, 32768, 128, 128, 64, "bf16", "bf16"),
    Stage3Case("F13", 64, 8, 8, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage3Case("F14", 32, 16, 16, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage3Case("F15", 16, 32, 32, 4096, 128, 128, 128, "fp16", "fp32"),
    Stage3Case("F16", 8, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage3Case("F17", 64, 8, 8, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage3Case("F18", 32, 16, 16, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage3Case("L1", 1, 32, 32, 65536, 128, 128, 64, "bf16", "bf16", 64),
    Stage3Case("L2", 1, 16, 16, 65536, 128, 128, 64, "fp16", "fp16", 33),
    Stage3Case("L3", 1, 8, 8, 131072, 128, 128, 64, "bf16", "bf16", 333),
    Stage3Case("L4", 1, 4, 4, 262144, 128, 128, 64, "fp16", "fp32", 567),
    Stage3Case("L5", 1, 16, 16, 32768, 128, 128, 64, "bf16", "bf16", 7),
    Stage3Case("L6", 1, 8, 8, 65536, 128, 128, 64, "fp16", "fp16", 25),
    Stage3Case("L7", 1, 16, 32, 16384, 128, 256, 64, "fp16", "fp32", 128),
    Stage3Case("L8", 1, 21, 63, 16384, 128, 256, 64, "bf16", "fp32", 2),
    Stage3Case("L9", 1, 8, 32, 65536, 128, 256, 128, "bf16", "fp32", 172),
    Stage3Case("L10", 1, 16, 32, 65536, 128, 128, 64, "fp16", "fp32", 668),
    Stage3Case("L11", 1, 4, 32, 65536, 128, 128, 128, "bf16", "fp32", 17),
    Stage3Case("L12", 1, 2, 64, 262144, 128, 256, 64, "fp16", "fp32", 32),
    Stage3Case("F19", 1, 16, 32, 4096, 128, 256, 64, "fp16", "fp32"),
    Stage3Case("F20", 16, 21, 63, 2048, 128, 256, 64, "bf16", "fp32"),
    Stage3Case("F21", 711, 4, 32, 196, 128, 128, 128, "fp16", "fp32"),
    Stage3Case("F22", 176, 2, 64, 24, 128, 256, 64, "bf16", "fp32"),
)


SMOKE_CASE_NAMES = {"F1", "F3", "F19", "F20", "L9"}


def rand_symmetric(shape: tuple[int, ...], device: str, dtype: torch.dtype, input_scale: float):
    return (torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0) * input_scale


def make_inputs(case: Stage3Case, device: str, seed: int, input_scale: float):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    ktype = DTYPES[case.ktype]
    gtype = DTYPES[case.gtype]
    k = rand_symmetric((case.B, case.KH, case.T, case.K), device, ktype, input_scale)
    v = rand_symmetric((case.B, case.VH, case.T, case.V), device, ktype, input_scale)
    beta = rand_symmetric((case.B, case.VH, case.T), device, gtype, input_scale)
    A = rand_symmetric((case.B, case.VH, case.T, case.chunk_size), device, ktype, input_scale)
    dw = rand_symmetric((case.B, case.VH, case.T, case.K), device, ktype, input_scale)
    du = rand_symmetric((case.B, case.VH, case.T, case.V), device, ktype, input_scale)
    g = rand_symmetric((case.B, case.VH, case.T), device, gtype, input_scale)
    return k, v, beta, A, dw, du, g


def build_tasks(case: Stage3Case, cu_seqlens: list[int] | None, chunk_indices: list[int] | None) -> list[TaskRange]:
    tasks: list[TaskRange] = []
    if cu_seqlens is None:
        chunk_num_per_b = (case.T + case.chunk_size - 1) // case.chunk_size
        for batch in range(case.B):
            for chunk_idx in range(chunk_num_per_b):
                begin = chunk_idx * case.chunk_size
                end = min(begin + case.chunk_size, case.T)
                tasks.append(TaskRange(batch, begin, end))
        return tasks

    assert chunk_indices is not None
    for idx in range(0, len(chunk_indices), 2):
        seq_idx = chunk_indices[idx]
        chunk_idx = chunk_indices[idx + 1]
        seq_begin = cu_seqlens[seq_idx]
        seq_end = cu_seqlens[seq_idx + 1]
        begin = seq_begin + chunk_idx * case.chunk_size
        end = min(begin + case.chunk_size, seq_end)
        tasks.append(TaskRange(0, begin, end))
    return tasks


def ct_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).removeprefix("torch.")


def ct_success(result) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(result)


def ct_failure_summary(result) -> str:
    if not isinstance(result, dict):
        return "no detail"
    metrics = result.get("metrics")
    if metrics is None:
        return "no metrics"
    return (
        f"fail_count={getattr(metrics, 'fail_count', 'NA')}, "
        f"pass_rate={getattr(metrics, 'pass_rate', 'NA')}, "
        f"max_diff={getattr(metrics, 'max_diff', 'NA')}, "
        f"max_re={getattr(metrics, 'max_re_calc', 'NA')}"
    )


def compare_with_ct(name: str, actual: torch.Tensor, expected: torch.Tensor, failures: list[str], verbose_ct: bool):
    try:
        if verbose_ct:
            result = ct.single(actual.detach().cpu(), expected.detach().cpu(), dtype=ct_dtype(actual.dtype))
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = ct.single(actual.detach().cpu(), expected.detach().cpu(), dtype=ct_dtype(actual.dtype))
    except Exception as exc:
        failures.append(f"{name}: ct.single failed: {exc}")
    else:
        if not ct_success(result):
            failures.append(f"{name}: ct.single failed ({ct_failure_summary(result)})")


def run_case(case: Stage3Case, args) -> list[str]:
    k, v, beta, A, dw, du, g = make_inputs(case, args.device, args.seed, args.input_scale)
    cu_seqlens = None
    chunk_indices = None
    if case.cu_seqlens_len is not None:
        cu_seqlens = prepare_cu_seqlens(case.T, case.cu_seqlens_len)
        chunk_indices = prepare_chunk_indices(cu_seqlens, case.chunk_size)
    tasks = build_tasks(case, cu_seqlens, chunk_indices)

    with torch.no_grad():
        case_start = time.perf_counter()
        kernel_start = time.perf_counter()
        outputs = fla_ascendc.prepare_wy_repr_bwd_stage3_debug(
            k, v, beta, A, dw, du, g, case.chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )
        torch.npu.synchronize()
        release_aclnn_keepalive()
        kernel_seconds = time.perf_counter() - kernel_start
        print(f"  phase_time kernel_sync={kernel_seconds:.6f}s", flush=True)
        if args.kernel_only:
            del k, v, beta, A, dw, du, g, outputs
            gc.collect()
            torch.npu.empty_cache()
            return []

        golden_start = time.perf_counter()
        expect_d = fla_ascendc.prepare_wy_repr_bwd_da(
            k, v, beta, A, dw, du, g, chunk_size=case.chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )
        torch.npu.synchronize()
        release_aclnn_keepalive()
        golden_seconds = time.perf_counter() - golden_start
        print(f"  phase_time golden_npu={golden_seconds:.6f}s", flush=True)

        compare_start = time.perf_counter()
        debug_dkb = outputs[7]
        debug_dk = outputs[8]
        failures: list[str] = []
        checked_lines = 0
        group_size = case.VH // case.KH

        for task_idx, task in enumerate(tasks):
            cur = task.end - task.begin
            checked_lines += 2 * case.VH
            expect_dkb_heads = []
            expect_dk_heads = []
            for hv in range(case.VH):
                hk = hv // group_size
                d_chunk = expect_d[task.batch, hv, task.begin:task.end, :cur]
                k_chunk = k[task.batch, hk, task.begin:task.end, :]
                beta_chunk = beta[task.batch, hv, task.begin:task.end]
                kbeta_chunk = (k_chunk.to(torch.float32) * beta_chunk.to(torch.float32)[:, None]).to(k.dtype)
                expect_dkb_heads.append((d_chunk.transpose(0, 1) @ k_chunk).to(k.dtype))
                expect_dk_heads.append((d_chunk @ kbeta_chunk).to(k.dtype))
            expect_dkb = torch.stack(expect_dkb_heads, dim=0)
            expect_dk = torch.stack(expect_dk_heads, dim=0)
            torch.npu.synchronize()

            compare_with_ct(
                f"{case.name}.task{task_idx}.Dkb",
                debug_dkb[task_idx, :, :cur, :],
                expect_dkb,
                failures,
                args.verbose_ct,
            )
            compare_with_ct(
                f"{case.name}.task{task_idx}.DK",
                debug_dk[task_idx, :, :cur, :],
                expect_dk,
                failures,
                args.verbose_ct,
            )
            if args.progress_interval > 0 and (
                (task_idx + 1) % args.progress_interval == 0 or task_idx + 1 == len(tasks)
            ):
                print(f"  compare_progress tasks={task_idx + 1}/{len(tasks)}", flush=True)

        torch.npu.synchronize()
        compare_seconds = time.perf_counter() - compare_start
        total_seconds = time.perf_counter() - case_start
        print(f"  phase_time golden_compare={compare_seconds:.6f}s total={total_seconds:.6f}s", flush=True)

    print(f"  checked_chunks={len(tasks)} checked_lines={checked_lines}")

    del k, v, beta, A, dw, du, g, outputs, expect_d
    gc.collect()
    torch.npu.empty_cache()
    return failures


def select_cases(args) -> list[Stage3Case]:
    if args.cases:
        wanted = {case.strip() for case in args.cases.split(",") if case.strip()}
        return [case for case in FULL_GDN_CASES if case.name in wanted]
    if args.suite == "smoke":
        return [case for case in FULL_GDN_CASES if case.name in SMOKE_CASE_NAMES]
    if args.suite == "fixed":
        return [case for case in FULL_GDN_CASES if case.cu_seqlens_len is None]
    if args.suite == "varlen":
        return [case for case in FULL_GDN_CASES if case.cu_seqlens_len is not None]
    return list(FULL_GDN_CASES)


def run_cases(cases: Iterable[Stage3Case], args) -> int:
    all_failures: list[str] = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}] {case.name}: B={case.B}, KH={case.KH}, VH={case.VH}, T={case.T}, "
            f"K={case.K}, V={case.V}, chunk={case.chunk_size}, ktype={case.ktype}, gtype={case.gtype}, "
            f"varlen_L={case.cu_seqlens_len}",
            flush=True,
        )
        failures = run_case(case, args)
        if failures:
            all_failures.extend(failures)
            print(f"  RESULT: FAIL ({len(failures)} outputs)", flush=True)
            if args.stop_on_fail:
                break
        elif args.kernel_only:
            print("  RESULT: KERNEL_ONLY_PASS (precision not checked)", flush=True)
        else:
            print("  RESULT: PASS", flush=True)

    if all_failures:
        print("FAILED CASES:")
        for failure in all_failures:
            print(f"  {failure}")
        return 1
    if args.kernel_only:
        print("ALL CASES KERNEL RETURNED")
        return 0
    print("ALL CASES PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="prepare_wy_repr_bwd stage3 Dkb/DK golden precision test")
    parser.add_argument("--suite", choices=("smoke", "fixed", "varlen", "all"), default="smoke")
    parser.add_argument("--cases", default="")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Half-width of the symmetric random input range. Default 1.0 generates values in [-1, 1].",
    )
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--kernel-only", action="store_true")
    parser.add_argument("--verbose-ct", action="store_true", help="Print every ct.single report.")
    parser.add_argument("--progress-interval", type=int, default=64, help="Task interval for compare progress logs.")
    args = parser.parse_args()
    selected = select_cases(args)
    if not selected:
        raise RuntimeError("No cases selected.")
    return run_cases(selected, args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

gpu_root=${GDN_GPU_ROOT:-/mnt/d/golden}
repo_root=${GDN_GPU_NPU_REPO:-$gpu_root/flash-linear-attention-npu}
triton_root=${GDN_GPU_TRITON_REPO:-$gpu_root/flash-linear-attention}
python_bin=${GDN_GPU_PYTHON:-/home/zsl/.venvs/fla/bin/python3}
port=${GDN_GPU_PORT:-9090}
device=${GDN_GPU_DEVICE:-0}
use_qk_l2norm=${GDN_ATK_USE_QK_L2NORM:-false}
output_dir=${GDN_GPU_OUTPUT_DIR:-$gpu_root/atk_output/chunk_gated_delta_rule_fwd}
op_dir=$repo_root/tests/atk/chunk_gated_delta_rule_fwd

case "$repo_root" in "$gpu_root"/*) ;; *) echo "GPU 测试仓必须位于 $gpu_root 内" >&2; exit 2;; esac
case "$triton_root" in "$gpu_root"/*) ;; *) echo "Triton 仓必须位于 $gpu_root 内" >&2; exit 2;; esac
case "$output_dir" in "$gpu_root"/*) ;; *) echo "GPU 输出必须位于 $gpu_root 内" >&2; exit 2;; esac
[[ -f "$op_dir/executor_chunk_gated_delta_rule_fwd.py" ]] || {
    echo "找不到 GPU executor：$op_dir" >&2
    exit 2
}
[[ -d "$triton_root/fla" ]] || { echo "找不到 Triton FLA：$triton_root" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "找不到 GPU Python：$python_bin" >&2; exit 2; }
[[ "$use_qk_l2norm" == "true" || "$use_qk_l2norm" == "false" ]] || {
    echo "GDN_ATK_USE_QK_L2NORM 必须为 true 或 false" >&2
    exit 2
}

mkdir -p "$output_dir"
mkdir -p "$output_dir/tmp" "$output_dir/cache" "$output_dir/triton_cache"
export CUDA_VISIBLE_DEVICES=$device
export PYTHONPATH=$gpu_root/atk_pkg:$triton_root:$op_dir:${PYTHONPATH:-}
export GDN_ATK_TARGET=pr540
export GDN_ATK_USE_QK_L2NORM=$use_qk_l2norm
export TMPDIR=$output_dir/tmp
export XDG_CACHE_HOME=$output_dir/cache
export TORCHINDUCTOR_CACHE_DIR=$output_dir/cache/torchinductor
export TRITON_CACHE_DIR=$output_dir/triton_cache
cd "$op_dir"
exec "$python_bin" -m atk server \
    --host 0.0.0.0 \
    --port "$port" \
    --devices 0 \
    --name gpu_reference \
    --output_path "$output_dir" \
    --plugin_path ./executor_chunk_gated_delta_rule_fwd.py \
    --bind_cpu_type 1 \
    --timeout 8000

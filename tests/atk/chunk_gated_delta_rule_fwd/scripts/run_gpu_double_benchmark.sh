#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
op_dir=$(cd -- "$script_dir/.." && pwd)
device=${1:-0}
atk_bin=${ATK_BIN:-atk}
case_json=${GDN_ATK_CASE_JSON:-$op_dir/atk_chunk_gated_delta_rule_fwd.json}
gpu_host=${GDN_GPU_HOST:?必须设置 GDN_GPU_HOST}
gpu_port=${GDN_GPU_PORT:-9090}
gpu_device=${GDN_GPU_LOGICAL_DEVICE:-0}
output_root=${ATK_OUTPUT_ROOT:-$op_dir/atk_output/gpu_double_benchmark}
timeout=${GDN_ATK_TIMEOUT:-2000}
use_qk_l2norm=${GDN_ATK_USE_QK_L2NORM:-false}

[[ "$device" =~ ^[0-9]+$ ]] || { echo "device 必须是非负整数" >&2; exit 2; }
[[ -f "$case_json" ]] || { echo "找不到用例：$case_json" >&2; exit 2; }
command -v "$atk_bin" >/dev/null 2>&1 || { echo "找不到 ATK 命令：$atk_bin" >&2; exit 2; }
[[ "$use_qk_l2norm" == "true" || "$use_qk_l2norm" == "false" ]] || {
    echo "GDN_ATK_USE_QK_L2NORM 必须为 true 或 false" >&2
    exit 2
}

case_ids=${GDN_ATK_CASE_IDS:-$(python3 - "$case_json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cases = json.load(handle)

selected = []
for case in cases:
    values = {
        item.get("name"): item.get("range_values")
        for item in case.get("inputs", [])
        if isinstance(item, dict)
    }
    q = next(item for item in case["inputs"] if item.get("name") == "q")
    v = next(item for item in case["inputs"] if item.get("name") == "v")
    if (
        values.get("chunk_size") == 64
        and q.get("dtype") == "bf16"
        and q.get("shape", [0])[-1] == 128
        and v.get("shape", [0])[-1] == 128
        and v["shape"][1] % q["shape"][1] == 0
        and v["shape"][1] // q["shape"][1] <= 4
    ):
        selected.append(int(case["id"]))
print(selected)
PY
)}
[[ "$case_ids" != "[]" ]] || { echo "没有可用的 chunk_size=64 GPU case" >&2; exit 2; }
export GDN_ATK_TARGET=pr540
export GDN_ATK_USE_QK_L2NORM=$use_qk_l2norm
export PYTHONPATH=$op_dir:${PYTHONPATH:-}

timestamp=$(date +%Y%m%d_%H%M%S)
run_dir=$output_root/$timestamp
node_output_path=./atk_distributed_output/$timestamp
mkdir -p "$run_dir"

printf '%q ' "$atk_bin" node --name npu_dut --backend npu --devices "$device" \
    --output_path "$node_output_path" node --name gpu_reference --backend gpu \
    --host "$gpu_host" --port "$gpu_port" --devices "$gpu_device" \
    --is_compare true --output_path "$node_output_path" task -c "$case_json" \
    --task accuracy --bm_device gpu -p "$op_dir/executor_chunk_gated_delta_rule_fwd.py" \
    -wl "$case_ids" --save_data output --syc_dataset -mt 1 -to "$timeout" \
    >"$run_dir/command.txt"
printf '\n' >>"$run_dir/command.txt"

cd "$run_dir"
exec "$atk_bin" node --name npu_dut --backend npu --devices "$device" \
    --output_path "$node_output_path" \
  node --name gpu_reference --backend gpu \
    --host "$gpu_host" --port "$gpu_port" --devices "$gpu_device" \
    --is_compare true --output_path "$node_output_path" \
  task -c "$case_json" --task accuracy --bm_device gpu \
    -p "$op_dir/executor_chunk_gated_delta_rule_fwd.py" \
    -wl "$case_ids" --save_data output --syc_dataset -mt 1 -to "$timeout"

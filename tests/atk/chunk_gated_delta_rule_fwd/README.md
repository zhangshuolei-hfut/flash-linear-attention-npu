# chunk_gated_delta_rule_fwd ATK 验证

本目录验证公开接口 `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`。融合算子在 A2/A3
使用私有 `arch22` 实现，在 A5 使用私有 `arch35` 实现；A3 当前只具备注册和编译验收环境。

## 支持范围

- `q/k`：`[B,Hk,T,128]`，BF16/FP16。
- `v`：`[B,Hv,T,V]`，BF16/FP16，`V=128/256`，`Hv % Hk == 0`。
- `g/beta`：`[B,T,Hv]`；`g` 为 FP32，`beta` 与 q/k/v 同 dtype。
- `chunk_size`：64 或 128。
- 支持定长、变长、GVA、可选初始状态和可选最终状态。
- SoC：A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`)。

## 精度标杆

精度使用 ATK 原生 `cv_fused_double_benchmark`：

1. NPU DUT：`chunk_gated_delta_rule_fwd`；
2. NPU benchmark：公开算子链 `chunk_local_cumsum`、`chunk_scaled_dot_kkt`、`solve_tri`、
   `recompute_w_u_fwd`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o`；
3. CPU golden：相同冻结输入上的 FP64 recurrence。

公开 `solve_tri` 来自 main 已合入的 PR 398；融合 kernel 不引用该公开实现，只有双标杆链路调用它。
比较阈值为最大相对误差比例 5、平均相对误差比例 1.5、均方根误差比例 1.5。

也支持与 KDA 一致、专门看护 PR #540 A5 大融合新路径的远端 GPU 双标杆拓扑：
NPU 运行融合 DUT，GPU benchmark task
运行 Torch FP64 recurrence，GPU control task 运行统一 FLA Triton forward。三路使用同一
份 ATK 同步输入。DUT 固定使用 `layout=BSND`、`use_exp2=false`、
默认 `use_qk_l2norm_in_kernel=false`，通过 BSND layout 进入 Prepare + H + O 拼接路径；脚本只选择
该路径支持的 BF16、`K=V=128`、`chunk_size=64`、`HV/HK<=4` 冻结 case。

## 用例

- `atk_chunk_gated_delta_rule_fwd.json`：既有泛化 500 条冻结矩阵，五种场景各 100 条，
  覆盖 BF16/FP16、MHA/GVA、V128/V256、chunk 64/128、定长/变长及状态组合。
- `scripts/cases/legacy500_adapted.json`：既有 BF16/MHA 历史 500 条回归矩阵，不作为默认入口。
- `atk_chunk_gated_delta_rule_fwd_perf.json`：A5 两条模型 case：
  - 推理：`B=1,Hk=16,Hv=32,T=11274,K=V=128,chunk=64`，变长并输出最终状态；
  - 训练：`B=2,Hk=Hv=32,T=8192,K=V=128,chunk=64`，定长无状态输出。
- `atk_chunk_gated_delta_rule_fwd_mss.json`：从冻结矩阵抽取 6 条精简用例，覆盖 V128/V256、
  chunk 64/128、定长/变长、FP16/BF16 和状态输出。

A5 模型 shape 分别来源于 `推理model.csv` 和 `训练model.csv`，原文件 SHA256 为
`a8f21a5ddc23b824b2b5ccc625d33db95dd9441b33f2b0c0e3e313e72aeaa363` 与
`87e9bb1027c44eaf8cc2f5fc4d24256b22e0ff00c16162b5fb11aaaf24aca8ca`。

## TilingKey 覆盖

| TilingKey | 选择条件 | 普通/边界用例 | SoC | 实际选择证据 |
| --- | --- | --- | --- | --- |
| 1 | `V=128`，未进入 A5 推理模型特化 | MSS 0、2、4 | A2/A3/A5 | 本 PR 硬件门禁补录 |
| 2 | `V=256` | MSS 1、3、5 | A2/A3/A5 | 本 PR 硬件门禁补录 |
| 301 | A5 性能 case 0 的推理模式，初态为 BF16/FP32 | MSS 尚未覆盖 | A5 | 原模型 shape 已实测命中；MSS 待补 |

## 执行

先执行不依赖 NPU/ATK 的 ACLNN ABI 合同，确认公开参数顺序、ctypes 类型和默认路径映射：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```

公开 `aclnnChunkGatedDeltaRuleFwd` 保留完整扩展 ABI。当前 Phase6 默认路径使用
`layout=BNSD`、`useExp2=false`、`allowNegEigval=false`、`stateVFirst=false`，且
`aLog/dtBias` 与扩展中间输出为空；`finalStateOutOptional` 是否为空决定是否输出 final state。
尚未实现的扩展组合会显式返回参数错误，不会静默忽略。

正式 500 条双标杆精度使用可恢复分片入口；默认每 25 条启动一个 fresh ATK 进程，避免
六算子 benchmark 长进程状态累积：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

冒烟或单分片可直接使用三节点入口（默认 `-mt 5`）：

```bash
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_double_benchmark.sh 0
```

### GPU Triton 双标杆

GPU 侧的仓库、脚本、运行目录和输出必须全部位于 `/mnt/d/golden`。GPU 机器准备
`/mnt/d/golden/flash-linear-attention` 和
`/mnt/d/golden/flash-linear-attention-npu` 后，在 ATK 环境中启动服务：

```bash
cd /mnt/d/golden/flash-linear-attention-npu
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/start_gpu_server.sh
```

在 A5 发起端设置 GPU 宿主机可达地址，然后执行。默认自动选择冻结矩阵内 PR #540
支持集内的 case；烟测时通过 `GDN_ATK_CASE_IDS='[0]'` 指定单条：

```bash
export GDN_GPU_HOST=<gpu_host>
export GDN_GPU_PORT=<gpu_host_port>
GDN_ATK_CASE_IDS='[0]' \
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_gpu_double_benchmark.sh 0
```

默认验证 PR #540 新增的不做 Q/K L2Norm 分支。需要回归 L2Norm 分支时，GPU server 与
A5 发起端必须同时设置 `GDN_ATK_USE_QK_L2NORM=true` 后重新启动并执行。

远端模式必须保留 `--syc_dataset`，不得使用 `-sp`，并保持 `-mt 1`。同一 case 的日志
应分别出现 `dut`、`benchmark`、`golden`，其中目标依次为融合 NPU、统一 Triton GPU、
GPU FP64 recurrence。GPU 与 A5 上的 case JSON、executor、reference 和 role contract
必须做 SHA256 一致性检查。两个节点必须使用相同的相对 `output_path`，由各自服务的工作
目录解析；否则 ATK 26.7.8 派生 GPU golden 任务时可能把发起端绝对路径传到 GPU。

复跑历史矩阵：

```bash
GDN_ATK_CASE_JSON="$PWD/tests/atk/chunk_gated_delta_rule_fwd/scripts/cases/legacy500_adapted.json" \
bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh 0
```

性能、确定性和内存检测仍使用仓内统一入口：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=mssanitizer
```

正式结论必须记录代码 commit、ATK/CANN 版本、SoC、实际加载的 OPP、case JSON 哈希和原始报告。

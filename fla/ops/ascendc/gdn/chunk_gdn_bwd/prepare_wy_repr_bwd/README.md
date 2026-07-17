# PrepareWyReprBwd 算子说明

`PrepareWyReprBwd` 是 Chunk Gated Delta Rule 反向中 WY 表示准备阶段的独立融合入口。公开接口只接收原始输入 `k/v/beta/A/dw/du/g` 以及可选的 `cu_seqlens/chunk_indices`，一次返回 `dk/dv/dbeta/dg`。

本版按照新的算子结构生成：host tiling 直接描述该算子的 chunk、head、tile 和 workspace；kernel 内部构造 chunk 级伴随矩阵缓存，并在同一个 AiCore launch 中写回四个公开梯度输出。内部缓存不暴露给 PyTorch 或 ACLNN 调用方。

## 支持范围

当前只按 A2/A3 生成，OpDef 仅注册：

- `ascend910b`
- `ascend910_93`

其他架构需要单独评估核类型、workspace 切分与同步策略，不在本次范围内。

## ACLNN 接口

```cpp
aclnnStatus aclnnPrepareWyReprBwdGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *A,
    const aclTensor *dw,
    const aclTensor *du,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *dkOut,
    const aclTensor *dvOut,
    const aclTensor *dbetaOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnPrepareWyReprBwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

PyTorch 侧入口：

```python
dk, dv, dbeta, dg = torch.ops.npu.npu_prepare_wy_repr_bwd(
    k, v, beta, A, dw, du, g, chunk_size,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
)
```

## Kernel 结构

`op_kernel/prepare_wy_repr_bwd.cpp` 是独立 A2/A3 kernel 骨架，按 mixed AIC/AIV 拆分：

1. host tiling 使用 `GET_TPL_TILING_KEY(IS_VARLEN, k_dtype, gate_dtype, V)` 生成模板化 tiling key。
2. AIV 侧 `prepare_wy_repr_bwd_vector.h` 使用 AscendC 基础 API 生成 `kbg/vb/kbeta`，并处理 mask、exp、reduce 与公开输出写回。
3. AIC 侧 `prepare_wy_repr_bwd_cube.h` 使用 Catlass tile 层接口显式执行所有矩阵乘，包括 DA 阶段和 full 阶段的 `A.T@*`、`dA@*`、`K@K.T`。
4. AIC/AIV 之间先用全参与同步表达阶段边界，后续性能化再替换为细粒度 ready/free flag。
5. 任务粒度按每轮两个 chunk 遍历，并在 chunk 内顺序覆盖全部 value head；`dk` 对同一 key head 的多个 value head 贡献在同一阶段内累加。

这版重点是先在小 shape 上打通实际执行代码与精度验证；后续可在保持 `PrepareWyReprBwdTilingData` 主字段稳定的基础上，把同步策略替换为细粒度 flag 和双 slot 流水。

## Tiling 结构

`PrepareWyReprBwdTilingData` 包含：

- 主维度：`B/HV/HK/T/K/V`
- chunk 信息：`chunkSize/chunkNum/isVariable/headGroup`
- AIV 行粒度：`rowTileInput/rowTileDa/rowTileFullK/rowTileFullV/rowTileFullKkt`
- workspace：`kbg/vb/kbeta/da*/full*` 各阶段 offset、`kWorkspaceBytes/vWorkspaceBytes/aWorkspaceBytes/userWorkspaceBytes`

shape 校验覆盖：

- `k` 为 `[B, HK, T, K]`
- `v/du` 为 `[B, HV, T, V]`
- `beta/g` 为 `[B, HV, T]`
- `A` 为 `[B, HV, T, chunk_size]`
- `dw` 为 `[B, HV, T, K]`
- `HV` 必须是 `HK` 的正整数倍
- `chunk_size` 仅支持 `64/128`
- `V` 仅支持 `128/256`
- 可变长模式要求同时提供 `cu_seqlens` 与 `chunk_indices`，并要求 `B=1`

## 验证状态

本次按“只生成，不跑通”的要求提交，未执行 CANN 构建、单算子精度或性能测试。

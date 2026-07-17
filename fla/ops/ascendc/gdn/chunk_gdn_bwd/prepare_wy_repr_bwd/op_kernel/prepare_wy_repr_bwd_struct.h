/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_STRUCT_H
#define PREPARE_WY_REPR_BWD_STRUCT_H

#include <cstdint>
#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

#define PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE 0
#define PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE 1
#define PREPARE_WY_REPR_BWD_TPL_BF16 10
#define PREPARE_WY_REPR_BWD_TPL_FP16 20
#define PREPARE_WY_REPR_BWD_TPL_FP32 30

namespace GDN {

static constexpr bool IS_VARLEN_FALSE = PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE;
static constexpr bool IS_VARLEN_TRUE = PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE;

static constexpr int32_t TPL_BF16 = PREPARE_WY_REPR_BWD_TPL_BF16;
static constexpr int32_t TPL_FP16 = PREPARE_WY_REPR_BWD_TPL_FP16;
static constexpr int32_t TPL_FP32 = PREPARE_WY_REPR_BWD_TPL_FP32;
static constexpr uint64_t SYNC_FLAG_2 = 2;
static constexpr uint64_t SYNC_FLAG_3 = 3;
static constexpr uint64_t SYNC_FLAG_4 = 4;
static constexpr uint64_t SYNC_FLAG_5 = 5;
static constexpr uint64_t BWD_TASK_PAIR_SLOT_COUNT = 2;
static constexpr int64_t BWD_TASK_PAIR_INDEPENDENT_UNIT = 0;
static constexpr int64_t BWD_TASK_PAIR_SAME_OWNER_HV = 1;

struct PrepareWyReprBwdTilingData {
    // 张量主维度。k: [B, HK, T, K]，v/du: [B, HV, T, V]，beta/g: [B, HV, T]。
    int64_t B;
    int64_t HV;
    int64_t HK;
    int64_t T;
    int64_t K;
    int64_t V;

    // chunkSize 只支持 64/128；chunkNum 表示 batch 或可变长序列展开后的 chunk 总数。
    int64_t chunkSize;
    int64_t chunkNum;
    int64_t seqNum;
    int64_t isVariable;

    // 每个 key head 对应的 value head 数。kernel 用它做 GQA 映射与写回归属划分。
    int64_t headGroup;

    // A5 vector 双任务调度策略。非 GVA(headGroup=1) 时两个 slot 取两个独立 owner unit；
    // GVA(headGroup>1) 时两个 slot 取同一个 (chunk,hk) owner 下的两个 hv，避免 dk 跨核竞争。
    int64_t taskPairMode;
    int64_t ownerUnitNum;

    // AIV 基础 API 的行粒度。AIC 只做矩阵乘，所有逐元素准备、mask、规约和输出写回
    // 由 AIV 按这些行粒度切片处理。
    int64_t rowTileInput;
    int64_t rowTileDa;
    int64_t rowTileFullK;
    int64_t rowTileFullV;
    int64_t rowTileFullKkt;

    // AIV 手工 UB arena 布局，单位 byte。tiling 根据实际 dtype、K/V、chunkSize 和 row tile 计算，
    // kernel 只按这些 offset 切片复用 UB。
    int64_t vectorIoInOffset;
    int64_t vectorIoOutOffset;
    int64_t vectorPersistentOffset;
    int64_t vectorGateCacheOffset;
    int64_t vectorTempOffset;
    int64_t vectorMaskOffset;
    int64_t vectorZeroOffset;
    int64_t vectorUbBytes;
    int64_t a5UseSlotWork;
    int64_t a5UseVPrefetch;
    int64_t a5UseL1Resident;
    int64_t a5VectorRowTile;

    // 实际下发的 AIC/AIV core 组数。kernel 按 chunk 做 stride 分发，chunk 内每轮处理两个 value head。
    int64_t usedCoreNum;

    // workspace 使用 GetUserWorkspace 后的 slot 内偏移，单位 byte。每个 core 使用双 buffer slot，
    // 每个 slot 覆盖一个 chunk 的全部 value heads：
    //   K 类: [HV, chunkSize, K]
    //   V 类: [HV, chunkSize, V]
    //   A 类: [HV, chunkSize, chunkSize]
    int64_t kbgOffset;
    int64_t vbOffset;
    int64_t kbetaOffset;
    int64_t da1Offset;
    int64_t da2Offset;
    int64_t da4Offset;
    int64_t da5Offset;
    int64_t daOffset;
    int64_t fullDkOffset;
    int64_t fullDkbOffset;
    int64_t fullDkbgOffset;
    int64_t fullDvbOffset;
    int64_t fullKktOffset;
    int64_t kWorkspaceBytes;
    int64_t vWorkspaceBytes;
    int64_t aWorkspaceBytes;
    int64_t workspaceSlotBytes;
    int64_t workspaceBufferCount;
    int64_t userWorkspaceBytes;
};

} // namespace GDN

using PrepareWyReprBwdTilingData = GDN::PrepareWyReprBwdTilingData;

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(PrepareWyReprBwd,
    ASCENDC_TPL_BOOL_DECL(IS_VARLEN, PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE,
                          PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE),
    ASCENDC_TPL_DTYPE_DECL(D_T_K, PREPARE_WY_REPR_BWD_TPL_BF16, PREPARE_WY_REPR_BWD_TPL_FP16),
    ASCENDC_TPL_DTYPE_DECL(D_T_GATE, PREPARE_WY_REPR_BWD_TPL_BF16,
                           PREPARE_WY_REPR_BWD_TPL_FP16, PREPARE_WY_REPR_BWD_TPL_FP32),
    ASCENDC_TPL_UINT_DECL(V, 1, ASCENDC_TPL_UI_LIST, 128, 256),
    ASCENDC_TPL_UINT_DECL(CHUNK_SIZE, 1, ASCENDC_TPL_UI_LIST, 64, 128),
);

#define PREPARE_WY_REPR_BWD_SEL_ENTRY_CHUNK(                                                        \
    IS_VARLEN_VALUE, K_DTYPE, GATE_DTYPE, VALUE_DIM, CHUNK_DIM)                                     \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(IS_VARLEN, IS_VARLEN_VALUE),                           \
                         ASCENDC_TPL_DTYPE_SEL(D_T_K, K_DTYPE),                                      \
                         ASCENDC_TPL_DTYPE_SEL(D_T_GATE, GATE_DTYPE),                                \
                         ASCENDC_TPL_UINT_SEL(V, ASCENDC_TPL_UI_LIST, VALUE_DIM),                    \
                         ASCENDC_TPL_UINT_SEL(CHUNK_SIZE, ASCENDC_TPL_UI_LIST, CHUNK_DIM),           \
                         ASCENDC_TPL_TILING_STRUCT_SEL(PrepareWyReprBwdTilingData))

#define PREPARE_WY_REPR_BWD_SEL_ENTRY(IS_VARLEN_VALUE, K_DTYPE, GATE_DTYPE, VALUE_DIM)               \
    PREPARE_WY_REPR_BWD_SEL_ENTRY_CHUNK(IS_VARLEN_VALUE, K_DTYPE, GATE_DTYPE, VALUE_DIM, 64),        \
    PREPARE_WY_REPR_BWD_SEL_ENTRY_CHUNK(IS_VARLEN_VALUE, K_DTYPE, GATE_DTYPE, VALUE_DIM, 128)

ASCENDC_TPL_SEL(
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE, PREPARE_WY_REPR_BWD_TPL_FP16,
                                  PREPARE_WY_REPR_BWD_TPL_FP16, 128),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE, PREPARE_WY_REPR_BWD_TPL_FP16,
                                  PREPARE_WY_REPR_BWD_TPL_FP16, 256),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE, PREPARE_WY_REPR_BWD_TPL_BF16,
                                  PREPARE_WY_REPR_BWD_TPL_BF16, 128),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE, PREPARE_WY_REPR_BWD_TPL_BF16,
                                  PREPARE_WY_REPR_BWD_TPL_BF16, 256),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE, PREPARE_WY_REPR_BWD_TPL_FP16,
                                  PREPARE_WY_REPR_BWD_TPL_FP32, 128),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE, PREPARE_WY_REPR_BWD_TPL_FP16,
                                  PREPARE_WY_REPR_BWD_TPL_FP32, 256),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE, PREPARE_WY_REPR_BWD_TPL_BF16,
                                  PREPARE_WY_REPR_BWD_TPL_FP32, 128),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_FALSE, PREPARE_WY_REPR_BWD_TPL_BF16,
                                  PREPARE_WY_REPR_BWD_TPL_FP32, 256),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE, PREPARE_WY_REPR_BWD_TPL_FP16,
                                  PREPARE_WY_REPR_BWD_TPL_FP16, 128),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE, PREPARE_WY_REPR_BWD_TPL_FP16,
                                  PREPARE_WY_REPR_BWD_TPL_FP16, 256),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE, PREPARE_WY_REPR_BWD_TPL_BF16,
                                  PREPARE_WY_REPR_BWD_TPL_BF16, 128),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE, PREPARE_WY_REPR_BWD_TPL_BF16,
                                  PREPARE_WY_REPR_BWD_TPL_BF16, 256),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE, PREPARE_WY_REPR_BWD_TPL_FP16,
                                  PREPARE_WY_REPR_BWD_TPL_FP32, 128),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE, PREPARE_WY_REPR_BWD_TPL_FP16,
                                  PREPARE_WY_REPR_BWD_TPL_FP32, 256),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE, PREPARE_WY_REPR_BWD_TPL_BF16,
                                  PREPARE_WY_REPR_BWD_TPL_FP32, 128),
    PREPARE_WY_REPR_BWD_SEL_ENTRY(PREPARE_WY_REPR_BWD_TPL_VARLEN_TRUE, PREPARE_WY_REPR_BWD_TPL_BF16,
                                  PREPARE_WY_REPR_BWD_TPL_FP32, 256),
);

#undef PREPARE_WY_REPR_BWD_SEL_ENTRY
#undef PREPARE_WY_REPR_BWD_SEL_ENTRY_CHUNK
#endif

#endif // PREPARE_WY_REPR_BWD_STRUCT_H

/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd.cpp
 * \brief Standalone prepare_wy_repr_bwd kernel entry.
 */

#include "kernel_operator.h"
#include "arch35/prepare_wy_repr_bwd_cube.h"
#include "arch35/prepare_wy_repr_bwd_vector.h"
#include "prepare_wy_repr_bwd_struct.h"

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

namespace GDN {

template <typename T, typename GateT, bool IS_VARLEN, int V, int CHUNK_SIZE>
__aicore__ inline void PrepareWyReprBwdKernelImpl(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dw,
                                                  GM_ADDR du, GM_ADDR g, GM_ADDR cuSeqlens,
                                                  GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta,
                                                  GM_ADDR dg, GM_ADDR workspace,
                                                  const PrepareWyReprBwdTilingData *tiling)
{
    if ASCEND_IS_AIV {
        PrepareWyReprBwdVector<T, GateT, IS_VARLEN, V, CHUNK_SIZE> vector;
        vector.Init(k, v, beta, A, dw, du, g, cuSeqlens, chunkIndices, dk, dv, dbeta, dg, workspace, tiling);
        vector.ProcessPipeline();
    }

    if ASCEND_IS_AIC {
        PrepareWyReprBwdCube<T, IS_VARLEN, V, CHUNK_SIZE> cube;
        cube.Init(k, v, A, dw, du, cuSeqlens, chunkIndices, workspace, tiling);
        cube.ProcessPipeline();
    }
}

} // namespace GDN

#ifndef TORCH_MODE
template <bool IS_VARLEN, int D_T_K, int D_T_GATE, int V, int CHUNK_SIZE>
__global__ __aicore__ void prepare_wy_repr_bwd(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dw, GM_ADDR du, GM_ADDR g,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GDN::PrepareWyReprBwdTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::PrepareWyReprBwdTilingData, tilingData, tiling);

    GM_ADDR userWorkspace = workspace;
    if (workspace != nullptr) {
        userWorkspace = AscendC::GetUserWorkspace(workspace);
    }

    // 模板化 tiling 只注册四类 dtype 组合：K 为 fp16/bf16，gate 与 K 同类型或为 fp32。
    // 这里直接使用 if constexpr 做编译期分流，避免再通过额外 trait 间接映射类型。
    if constexpr (D_T_K == GDN::TPL_BF16 && D_T_GATE == GDN::TPL_BF16) {
        GDN::PrepareWyReprBwdKernelImpl<bfloat16_t, bfloat16_t, IS_VARLEN, V, CHUNK_SIZE>(
            k, v, beta, A, dw, du, g, cuSeqlens, chunkIndices, dk, dv, dbeta, dg, userWorkspace, &tilingData);
    } else if constexpr (D_T_K == GDN::TPL_BF16 && D_T_GATE == GDN::TPL_FP32) {
        GDN::PrepareWyReprBwdKernelImpl<bfloat16_t, float, IS_VARLEN, V, CHUNK_SIZE>(
            k, v, beta, A, dw, du, g, cuSeqlens, chunkIndices, dk, dv, dbeta, dg, userWorkspace, &tilingData);
    } else if constexpr (D_T_K == GDN::TPL_FP16 && D_T_GATE == GDN::TPL_FP16) {
        GDN::PrepareWyReprBwdKernelImpl<half, half, IS_VARLEN, V, CHUNK_SIZE>(
            k, v, beta, A, dw, du, g, cuSeqlens, chunkIndices, dk, dv, dbeta, dg, userWorkspace, &tilingData);
    } else if constexpr (D_T_K == GDN::TPL_FP16 && D_T_GATE == GDN::TPL_FP32) {
        GDN::PrepareWyReprBwdKernelImpl<half, float, IS_VARLEN, V, CHUNK_SIZE>(
            k, v, beta, A, dw, du, g, cuSeqlens, chunkIndices, dk, dv, dbeta, dg, userWorkspace, &tilingData);
    }
}
#endif

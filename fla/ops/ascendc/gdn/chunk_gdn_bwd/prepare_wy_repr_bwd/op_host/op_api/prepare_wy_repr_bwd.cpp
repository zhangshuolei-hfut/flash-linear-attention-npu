/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "prepare_wy_repr_bwd.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(PrepareWyReprBwd);

const std::array<const aclTensor *, 4> PrepareWyReprBwd(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
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
    aclOpExecutor *executor)
{
    L0_DFX(PrepareWyReprBwd, k, v, beta, a, dw, du, g, cuSeqlensOptional, chunkIndicesOptional, chunkSize,
           dkOut, dvOut, dbetaOut, dgOut);

    // 可变长输入在 OpDef 中是 OPTIONAL + ValueDepend。这里把 aclIntArray 转成 ND tensor，
    // 让 host tiling 能直接读取序列边界与 chunk 映射。
    const aclTensor *actualCuSeqLen = nullptr;
    if (cuSeqlensOptional) {
        actualCuSeqLen = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqLen)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    // 单次 AiCore launch 完成内部伴随矩阵构造和四个公开梯度写回。伴随矩阵只存在于
    // kernel workspace 中，不作为 executor 临时 tensor 暴露给调用方。
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PrepareWyReprBwd,
        OP_INPUT(k, v, beta, a, dw, du, g, actualCuSeqLen, actualChunkIndices),
        OP_OUTPUT(dkOut, dvOut, dbetaOut, dgOut),
        OP_ATTR(chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {dkOut, dvOut, dbetaOut, dgOut};
}

} // namespace l0op

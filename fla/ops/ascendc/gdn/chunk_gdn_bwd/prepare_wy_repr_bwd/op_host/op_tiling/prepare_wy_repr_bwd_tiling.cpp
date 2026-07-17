/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_tiling.cpp
 * \brief Tiling registration for prepare_wy_repr_bwd.
 */

#include "prepare_wy_repr_bwd_tiling.h"
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {
namespace {

int64_t ToPrepareWyReprBwdDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return PREPARE_WY_REPR_BWD_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT) {
        return PREPARE_WY_REPR_BWD_DTYPE_FP32;
    }
    return PREPARE_WY_REPR_BWD_DTYPE_FP16;
}

int32_t ToPrepareWyReprBwdTplDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return GDN::TPL_BF16;
    }
    if (dtype == ge::DT_FLOAT) {
        return GDN::TPL_FP32;
    }
    return GDN::TPL_FP16;
}

void PrepareWyReprBwdTilingDataPrint(gert::TilingContext *context, const PrepareWyReprBwdTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print PrepareWyReprBwd tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.B);
    OP_LOGD(nodeName, "=== HV: %ld", tiling.HV);
    OP_LOGD(nodeName, "=== HK: %ld", tiling.HK);
    OP_LOGD(nodeName, "=== T: %ld", tiling.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling.V);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== chunkNum: %ld", tiling.chunkNum);
    OP_LOGD(nodeName, "=== isVariable: %ld", tiling.isVariable);
    OP_LOGD(nodeName, "=== headGroup: %ld", tiling.headGroup);
    OP_LOGD(nodeName, "=== taskPairMode: %ld", tiling.taskPairMode);
    OP_LOGD(nodeName, "=== ownerUnitNum: %ld", tiling.ownerUnitNum);
    OP_LOGD(nodeName, "=== rowTileInput: %ld", tiling.rowTileInput);
    OP_LOGD(nodeName, "=== rowTileDa: %ld", tiling.rowTileDa);
    OP_LOGD(nodeName, "=== rowTileFullK: %ld", tiling.rowTileFullK);
    OP_LOGD(nodeName, "=== rowTileFullV: %ld", tiling.rowTileFullV);
    OP_LOGD(nodeName, "=== rowTileFullKkt: %ld", tiling.rowTileFullKkt);
    OP_LOGD(nodeName, "=== vectorIoInOffset: %ld", tiling.vectorIoInOffset);
    OP_LOGD(nodeName, "=== vectorIoOutOffset: %ld", tiling.vectorIoOutOffset);
    OP_LOGD(nodeName, "=== vectorPersistentOffset: %ld", tiling.vectorPersistentOffset);
    OP_LOGD(nodeName, "=== vectorGateCacheOffset: %ld", tiling.vectorGateCacheOffset);
    OP_LOGD(nodeName, "=== vectorTempOffset: %ld", tiling.vectorTempOffset);
    OP_LOGD(nodeName, "=== vectorMaskOffset: %ld", tiling.vectorMaskOffset);
    OP_LOGD(nodeName, "=== vectorZeroOffset: %ld", tiling.vectorZeroOffset);
    OP_LOGD(nodeName, "=== vectorUbBytes: %ld", tiling.vectorUbBytes);
    OP_LOGD(nodeName, "=== usedCoreNum: %ld", tiling.usedCoreNum);
    OP_LOGD(nodeName, "=== kbgOffset: %ld", tiling.kbgOffset);
    OP_LOGD(nodeName, "=== vbOffset: %ld", tiling.vbOffset);
    OP_LOGD(nodeName, "=== kbetaOffset: %ld", tiling.kbetaOffset);
    OP_LOGD(nodeName, "=== da1Offset: %ld", tiling.da1Offset);
    OP_LOGD(nodeName, "=== da2Offset: %ld", tiling.da2Offset);
    OP_LOGD(nodeName, "=== da4Offset: %ld", tiling.da4Offset);
    OP_LOGD(nodeName, "=== da5Offset: %ld", tiling.da5Offset);
    OP_LOGD(nodeName, "=== daOffset: %ld", tiling.daOffset);
    OP_LOGD(nodeName, "=== fullDkOffset: %ld", tiling.fullDkOffset);
    OP_LOGD(nodeName, "=== fullDkbOffset: %ld", tiling.fullDkbOffset);
    OP_LOGD(nodeName, "=== fullDkbgOffset: %ld", tiling.fullDkbgOffset);
    OP_LOGD(nodeName, "=== fullDvbOffset: %ld", tiling.fullDvbOffset);
    OP_LOGD(nodeName, "=== fullKktOffset: %ld", tiling.fullKktOffset);
    OP_LOGD(nodeName, "=== kWorkspaceBytes: %ld", tiling.kWorkspaceBytes);
    OP_LOGD(nodeName, "=== vWorkspaceBytes: %ld", tiling.vWorkspaceBytes);
    OP_LOGD(nodeName, "=== aWorkspaceBytes: %ld", tiling.aWorkspaceBytes);
    OP_LOGD(nodeName, "=== workspaceSlotBytes: %ld", tiling.workspaceSlotBytes);
    OP_LOGD(nodeName, "=== workspaceBufferCount: %ld", tiling.workspaceBufferCount);
    OP_LOGD(nodeName, "=== userWorkspaceBytes: %ld", tiling.userWorkspaceBytes);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print PrepareWyReprBwd tiling data end <<<<<<<<<<<<<<<<");
}

} // namespace

ge::graphStatus Tiling4PrepareWyReprBwd(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwd start.");
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    PrepareWyReprBwdTilingData *tiling = context->GetTilingData<PrepareWyReprBwdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto kDesc = context->GetInputDesc(BWD_INPUT_K_IDX);
    auto betaDesc = context->GetInputDesc(BWD_INPUT_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aicCoreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    if (aicCoreNum == 0) {
        aicCoreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
    }

    auto cuSeqlensTensor = context->GetOptionalInputTensor(BWD_INPUT_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(BWD_INPUT_CHUNK_INDICES_IDX);
    const int64_t *cuSeqlensData = cuSeqlensTensor == nullptr ? nullptr : cuSeqlensTensor->GetData<int64_t>();
    const int64_t *chunkIndicesData =
        chunkIndicesTensor == nullptr ? nullptr : chunkIndicesTensor->GetData<int64_t>();

    PrepareWyReprBwdTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(BWD_INPUT_K_IDX),
        context->GetRequiredInputShape(BWD_INPUT_V_IDX),
        context->GetRequiredInputShape(BWD_INPUT_BETA_IDX),
        context->GetRequiredInputShape(BWD_INPUT_A_IDX),
        context->GetRequiredInputShape(BWD_INPUT_DW_IDX),
        context->GetRequiredInputShape(BWD_INPUT_DU_IDX),
        context->GetRequiredInputShape(BWD_INPUT_G_IDX),
        context->GetOptionalInputShape(BWD_INPUT_SEQLENS_IDX),
        context->GetOptionalInputShape(BWD_INPUT_CHUNK_INDICES_IDX),
        cuSeqlensData,
        chunkIndicesData,
        static_cast<int64_t>(*(attrPtr->GetAttrPointer<int32_t>(BWD_ATTR_CHUNK_SIZE_IDX))),
        ToPrepareWyReprBwdDtype(kDesc->GetDataType()),
        ToPrepareWyReprBwdDtype(betaDesc->GetDataType()),
        ubSize,
        aicCoreNum,
        static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()),
    };

    PrepareWyReprBwdTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    const bool isVarLenKey = processor.IsVariableLength();
    const int32_t kTplDtype = ToPrepareWyReprBwdTplDtype(kDesc->GetDataType());
    const int32_t gateTplDtype = ToPrepareWyReprBwdTplDtype(betaDesc->GetDataType());
    const uint64_t cubeTileChunkSize = tiling->V == 128 ? 128 : static_cast<uint64_t>(tiling->chunkSize);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(isVarLenKey, kTplDtype, gateTplDtype,
                                                  static_cast<uint64_t>(tiling->V),
                                                  cubeTileChunkSize);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(processor.GetBlockDim());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);

    OP_LOGD(context->GetNodeName(), "tilingKey: %lu", static_cast<uint64_t>(context->GetTilingKey()));
    PrepareWyReprBwdTilingDataPrint(context, *tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwd end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForPrepareWyReprBwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PrepareWyReprBwd)
    .Tiling(Tiling4PrepareWyReprBwd)
    .TilingParse<PrepareWyReprBwdCompileInfo>(TilingPrepareForPrepareWyReprBwd);

} // namespace optiling

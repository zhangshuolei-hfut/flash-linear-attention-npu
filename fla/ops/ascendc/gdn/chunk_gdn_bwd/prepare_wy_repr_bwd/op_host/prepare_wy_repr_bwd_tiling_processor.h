/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_TILING_PROCESSOR_H
#define PREPARE_WY_REPR_BWD_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/prepare_wy_repr_bwd_struct.h"

#include <exe_graph/runtime/storage_shape.h>
#include "log/log.h"
#include <register/op_impl_registry.h>

using GDN::PrepareWyReprBwdTilingData;

namespace optiling {

static constexpr size_t BWD_INPUT_K_IDX = 0;
static constexpr size_t BWD_INPUT_V_IDX = 1;
static constexpr size_t BWD_INPUT_BETA_IDX = 2;
static constexpr size_t BWD_INPUT_A_IDX = 3;
static constexpr size_t BWD_INPUT_DW_IDX = 4;
static constexpr size_t BWD_INPUT_DU_IDX = 5;
static constexpr size_t BWD_INPUT_G_IDX = 6;
static constexpr size_t BWD_INPUT_SEQLENS_IDX = 7;
static constexpr size_t BWD_INPUT_CHUNK_INDICES_IDX = 8;
static constexpr size_t BWD_ATTR_CHUNK_SIZE_IDX = 0;

static constexpr size_t BWD_DIM_NUM_1 = 1;
static constexpr size_t BWD_DIM_NUM_3 = 3;
static constexpr size_t BWD_DIM_NUM_4 = 4;
static constexpr size_t BWD_DIM_0 = 0;
static constexpr size_t BWD_DIM_1 = 1;
static constexpr size_t BWD_DIM_2 = 2;
static constexpr size_t BWD_DIM_3 = 3;

static constexpr int64_t BWD_CHUNK_SIZE_64 = 64;
static constexpr int64_t BWD_CHUNK_SIZE_128 = 128;
static constexpr int64_t BWD_K_DIM_128 = 128;
static constexpr int64_t BWD_V_DIM_128 = 128;
static constexpr int64_t BWD_V_DIM_256 = 256;
static constexpr int64_t BWD_VAR_LEN_B_DIM_1 = 1;

static constexpr int64_t PREPARE_WY_REPR_BWD_DTYPE_FP16 = 0;
static constexpr int64_t PREPARE_WY_REPR_BWD_DTYPE_BF16 = 1;
static constexpr int64_t PREPARE_WY_REPR_BWD_DTYPE_FP32 = 2;

static constexpr uint64_t BWD_SIZE_HALF = 2;
static constexpr uint64_t BWD_SIZE_FP32 = 4;
static constexpr uint64_t BWD_ONE_BLOCK_32 = 32;
static constexpr uint64_t BWD_TILE_MIN_ROW = 8;
static constexpr uint64_t BWD_VECTOR_IO_BUFFER_COUNT = 2;
static constexpr uint64_t BWD_BIT_NUM_FOR_UINT8 = 8;
static constexpr uint64_t BWD_A5_UB_SIZE = 248 * 1024;
static constexpr uint64_t BWD_A5_L1_SIZE = 512 * 1024;

struct PrepareWyReprBwdTilingContext {
    const char *nodeName;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *betaShape;
    const gert::StorageShape *aShape;
    const gert::StorageShape *dwShape;
    const gert::StorageShape *duShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    const int64_t *cuSeqlensData;
    const int64_t *chunkIndicesData;
    int64_t chunkSize;
    int64_t kDataType;
    int64_t betaDataType;
    uint64_t ubSize;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
};

class PrepareWyReprBwdTilingProcessor {
    PrepareWyReprBwdTilingContext &ctx_;
    PrepareWyReprBwdTilingData &tiling_;
    size_t workspaceSize_ = 0;
    uint32_t blockDim_ = 0;

public:
    explicit PrepareWyReprBwdTilingProcessor(PrepareWyReprBwdTilingContext &ctx, PrepareWyReprBwdTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    size_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

    uint32_t GetBlockDim() const
    {
        return blockDim_;
    }

    bool IsVariableLength() const
    {
        return ctx_.cuSeqlensShape != nullptr || ctx_.chunkIndicesShape != nullptr;
    }

    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                             const char *inputName)
    {
        OP_CHECK_IF(curShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.", inputName),
                    return ge::GRAPH_FAILED);
        size_t dimNum = curShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != validDimNum,
                    OP_LOGE(ctx_.nodeName,
                            "Check input %s shape failed, dim num should be %zu, but get %zu.", inputName,
                            validDimNum, dimNum),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PreCheck()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, BWD_DIM_NUM_4, "k") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.vShape, BWD_DIM_NUM_4, "v") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.betaShape, BWD_DIM_NUM_3, "beta") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.aShape, BWD_DIM_NUM_4, "A") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.dwShape, BWD_DIM_NUM_4, "dw") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.duShape, BWD_DIM_NUM_4, "du") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, BWD_DIM_NUM_3, "g") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                                 const char *inputName2, size_t compareDimNum)
    {
        for (size_t dimIndex = 0; dimIndex < compareDimNum; ++dimIndex) {
            size_t shapeDim1 = shape1.GetDim(dimIndex);
            size_t shapeDim2 = shape2.GetDim(dimIndex);
            OP_CHECK_IF(shapeDim1 != shapeDim2,
                        OP_LOGE(ctx_.nodeName,
                                "Compare input shape of %s and %s failed, dim %zu should be same, got %zu and %zu.",
                                inputName1, inputName2, dimIndex, shapeDim1, shapeDim2),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    uint64_t AlignUp(uint64_t value, uint64_t align) const
    {
        return (value + align - 1) / align * align;
    }

    int64_t CeilDiv(int64_t a, int64_t b) const
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    uint64_t DtypeSize(int64_t dtype) const
    {
        return dtype == PREPARE_WY_REPR_BWD_DTYPE_FP32 ? BWD_SIZE_FP32 : BWD_SIZE_HALF;
    }

    struct VectorUbLayout {
        uint64_t ioInOffset = 0;
        uint64_t ioOutOffset = 0;
        uint64_t persistentOffset = 0;
        uint64_t gateCacheOffset = 0;
        uint64_t tempOffset = 0;
        uint64_t maskOffset = 0;
        uint64_t zeroOffset = 0;
        uint64_t totalBytes = 0;
    };

    uint64_t VectorGateCacheStride() const
    {
        return AlignUp(static_cast<uint64_t>(tiling_.chunkSize) * BWD_SIZE_FP32, BWD_ONE_BLOCK_32) /
               BWD_SIZE_FP32;
    }

    bool EnableVectorInputCache() const
    {
        return tiling_.chunkSize == BWD_CHUNK_SIZE_64 && tiling_.K == BWD_K_DIM_128 &&
               tiling_.V == BWD_V_DIM_128;
    }

    uint64_t CalcVectorUbLayout(uint64_t rowNum, uint64_t elemBytes, uint64_t gateBytes,
                                VectorUbLayout *layout = nullptr) const
    {
        uint64_t chunkSize = static_cast<uint64_t>(tiling_.chunkSize);
        uint64_t kDim = static_cast<uint64_t>(tiling_.K);
        uint64_t vDim = static_cast<uint64_t>(tiling_.V);
        uint64_t maxDim = std::max<uint64_t>(static_cast<uint64_t>(tiling_.K), static_cast<uint64_t>(tiling_.V));
        maxDim = std::max<uint64_t>(maxDim, chunkSize);
        uint64_t ioBytes = rowNum * maxDim * elemBytes;
        uint64_t gateIoBytes = chunkSize * gateBytes;
        ioBytes = std::max<uint64_t>(ioBytes, gateIoBytes);

        uint64_t gateCacheBytes = 4 * VectorGateCacheStride() * BWD_SIZE_FP32;
        // Two head slots each retain the rows owned by one of the two AIV subblocks.
        uint64_t finalDaCacheBytes = chunkSize * chunkSize * elemBytes;
        uint64_t inputCacheBytes = EnableVectorInputCache() ?
                                   chunkSize * (kDim + vDim) * BWD_SIZE_FP32 : 0;
        uint64_t persistentBytes = gateCacheBytes + finalDaCacheBytes + inputCacheBytes;
        uint64_t brcbBytes = rowNum * BWD_ONE_BLOCK_32;
        uint64_t kbgTileCount = EnableVectorInputCache() ? 2 : 3;
        uint64_t kbgTempBytes = kbgTileCount * rowNum * maxDim * BWD_SIZE_FP32 +
                                2 * rowNum * BWD_SIZE_FP32 + 2 * brcbBytes;
        uint64_t da4TempBytes = 3 * rowNum * chunkSize * BWD_SIZE_FP32;
        uint64_t finalDaTempBytes = (2 * rowNum * chunkSize + rowNum) * BWD_SIZE_FP32 + brcbBytes;
        uint64_t outputKTileCount = EnableVectorInputCache() ? 4 : 5;
        uint64_t outputVTileCount = EnableVectorInputCache() ? 2 : 3;
        uint64_t outputKTempBytes = outputKTileCount * rowNum * kDim * BWD_SIZE_FP32 + 3 * brcbBytes;
        uint64_t outputVTempBytes = outputVTileCount * rowNum * vDim * BWD_SIZE_FP32 + 2 * brcbBytes;
        uint64_t kktTempBytes = chunkSize * chunkSize * BWD_SIZE_FP32 +
                                rowNum * chunkSize * BWD_SIZE_FP32 +
                                rowNum * BWD_SIZE_FP32 +
                                chunkSize * BWD_ONE_BLOCK_32;
        uint64_t singletonTempBytes = (3 * kDim + 3 * vDim + 2 * (BWD_ONE_BLOCK_32 / BWD_SIZE_FP32) + 1) *
                                      BWD_SIZE_FP32;
        uint64_t tempBytes = std::max<uint64_t>(kbgTempBytes, da4TempBytes);
        tempBytes = std::max<uint64_t>(tempBytes, finalDaTempBytes);
        tempBytes = std::max<uint64_t>(tempBytes, outputKTempBytes);
        tempBytes = std::max<uint64_t>(tempBytes, outputVTempBytes);
        tempBytes = std::max<uint64_t>(tempBytes, kktTempBytes);
        tempBytes = std::max<uint64_t>(tempBytes, singletonTempBytes);
        uint64_t maskBlocksPerRow = CeilDiv(chunkSize, BWD_BIT_NUM_FOR_UINT8);
        uint64_t maskBytes = 2 * chunkSize * maskBlocksPerRow;

        VectorUbLayout localLayout;
        uint64_t offset = 0;
        localLayout.ioInOffset = offset;
        offset += BWD_VECTOR_IO_BUFFER_COUNT * AlignUp(ioBytes, BWD_ONE_BLOCK_32);
        localLayout.ioOutOffset = offset;
        offset += BWD_VECTOR_IO_BUFFER_COUNT * AlignUp(ioBytes, BWD_ONE_BLOCK_32);
        localLayout.persistentOffset = offset;
        offset += AlignUp(persistentBytes, BWD_ONE_BLOCK_32);
        localLayout.gateCacheOffset = offset;
        offset += AlignUp(gateCacheBytes, BWD_ONE_BLOCK_32);
        localLayout.tempOffset = offset;
        offset += AlignUp(tempBytes, BWD_ONE_BLOCK_32);
        localLayout.maskOffset = offset;
        offset += AlignUp(maskBytes, BWD_ONE_BLOCK_32);
        localLayout.zeroOffset = offset;
        offset += BWD_ONE_BLOCK_32;
        localLayout.totalBytes = offset;

        if (layout != nullptr) {
            *layout = localLayout;
        }
        return localLayout.totalBytes;
    }

    uint64_t SelectRowTile(uint64_t ubSize) const
    {
        uint64_t rowNum = static_cast<uint64_t>(tiling_.chunkSize) / 2;
        uint64_t elemBytes = DtypeSize(ctx_.kDataType);
        uint64_t gateBytes = DtypeSize(ctx_.betaDataType);
        while (rowNum >= BWD_TILE_MIN_ROW) {
            if (CalcVectorUbLayout(rowNum, elemBytes, gateBytes) <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        return std::max<uint64_t>(rowNum, BWD_TILE_MIN_ROW);
    }

    void SetVectorUbLayout(uint64_t rowNum)
    {
        VectorUbLayout layout;
        CalcVectorUbLayout(rowNum, DtypeSize(ctx_.kDataType), DtypeSize(ctx_.betaDataType), &layout);
        tiling_.vectorIoInOffset = static_cast<int64_t>(layout.ioInOffset);
        tiling_.vectorIoOutOffset = static_cast<int64_t>(layout.ioOutOffset);
        tiling_.vectorPersistentOffset = static_cast<int64_t>(layout.persistentOffset);
        tiling_.vectorGateCacheOffset = static_cast<int64_t>(layout.gateCacheOffset);
        tiling_.vectorTempOffset = static_cast<int64_t>(layout.tempOffset);
        tiling_.vectorMaskOffset = static_cast<int64_t>(layout.maskOffset);
        tiling_.vectorZeroOffset = static_cast<int64_t>(layout.zeroOffset);
        tiling_.vectorUbBytes = static_cast<int64_t>(layout.totalBytes);
    }

    uint64_t CalcA5VectorUbBytes(uint64_t workRows, bool useSlotWork, bool useVPrefetch) const
    {
        uint64_t chunkSize = static_cast<uint64_t>(tiling_.chunkSize);
        uint64_t localRows = chunkSize / 2;
        uint64_t elemBytes = DtypeSize(ctx_.kDataType);
        uint64_t gateBytes = DtypeSize(ctx_.betaDataType);
        uint64_t maxWidth = std::max<uint64_t>(static_cast<uint64_t>(tiling_.V),
                                               static_cast<uint64_t>(BWD_K_DIM_128));
        uint64_t splitResultBytes = localRows * maxWidth * elemBytes;
        uint64_t broadcastResultBytes = chunkSize * chunkSize * elemBytes;
        uint64_t resultSlotBytes = std::max<uint64_t>(splitResultBytes, broadcastResultBytes);
        uint64_t resultBytes = 2 * resultSlotBytes;
        uint64_t finalDaBytes = chunkSize * chunkSize * elemBytes;
        uint64_t workElements = std::max<uint64_t>(workRows * maxWidth,
                                                   localRows * static_cast<uint64_t>(tiling_.K));
        uint64_t workSlotBytes = workElements * elemBytes;
        uint64_t workSlotCount = useSlotWork ? 2 : 1;
        uint64_t gateStrideBytes = AlignUp(chunkSize * gateBytes, BWD_ONE_BLOCK_32);
        uint64_t gateSlotBytes = 4 * gateStrideBytes;
        uint64_t reduceBytes = 2 * chunkSize * BWD_SIZE_FP32;
        uint64_t ubBytes = resultBytes + 2 * finalDaBytes + 3 * workSlotCount * workSlotBytes +
                           2 * gateSlotBytes + reduceBytes;
        if (useVPrefetch) {
            ubBytes += workRows * static_cast<uint64_t>(tiling_.V) * elemBytes;
        }
        return ubBytes;
    }

    uint64_t SelectA5VectorRowTile() const
    {
        uint64_t localRows = static_cast<uint64_t>(tiling_.chunkSize) / 2;
        uint64_t rowTile = tiling_.chunkSize == BWD_CHUNK_SIZE_128 ? localRows / 2 : localRows;
        rowTile = std::max<uint64_t>(rowTile, BWD_TILE_MIN_ROW);
        while (rowTile > BWD_TILE_MIN_ROW) {
            if (CalcA5VectorUbBytes(rowTile, true, true) <= ctx_.ubSize &&
                CalcA5VectorUbBytes(rowTile, true, true) <= BWD_A5_UB_SIZE) {
                return rowTile;
            }
            rowTile /= 2;
        }
        return BWD_TILE_MIN_ROW;
    }

    bool CanUseA5L1Resident() const
    {
        uint64_t chunkSize = static_cast<uint64_t>(tiling_.chunkSize);
        uint64_t elemBytes = DtypeSize(ctx_.kDataType);
        uint64_t kBytes = chunkSize * static_cast<uint64_t>(BWD_K_DIM_128) * elemBytes;
        uint64_t vBytes = chunkSize * static_cast<uint64_t>(tiling_.V) * elemBytes;
        uint64_t aBytes = chunkSize * chunkSize * elemBytes;
        uint64_t scratchBytes = std::max<uint64_t>(3 * kBytes, 2 * vBytes);
        uint64_t residentBytes = scratchBytes + 2 * aBytes + 2 * kBytes + 2 * vBytes + 2 * aBytes + 2 * kBytes;
        return residentBytes <= BWD_A5_L1_SIZE;
    }

    void SetA5PipelinePolicy()
    {
        uint64_t a5RowTile = SelectA5VectorRowTile();
        bool useSlotWork = CalcA5VectorUbBytes(a5RowTile, true, false) <= ctx_.ubSize &&
                           CalcA5VectorUbBytes(a5RowTile, true, false) <= BWD_A5_UB_SIZE;
        bool useVPrefetch = useSlotWork &&
                            CalcA5VectorUbBytes(a5RowTile, true, true) <= ctx_.ubSize &&
                            CalcA5VectorUbBytes(a5RowTile, true, true) <= BWD_A5_UB_SIZE;
        tiling_.a5UseSlotWork = useSlotWork ? 1 : 0;
        tiling_.a5UseVPrefetch = useVPrefetch ? 1 : 0;
        tiling_.a5UseL1Resident = CanUseA5L1Resident() ? 1 : 0;
        tiling_.a5VectorRowTile = static_cast<int64_t>(a5RowTile);
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape kStorageShape = ctx_.kShape->GetStorageShape();
        const gert::Shape vStorageShape = ctx_.vShape->GetStorageShape();
        const gert::Shape betaStorageShape = ctx_.betaShape->GetStorageShape();
        const gert::Shape aStorageShape = ctx_.aShape->GetStorageShape();
        const gert::Shape dwStorageShape = ctx_.dwShape->GetStorageShape();
        const gert::Shape duStorageShape = ctx_.duShape->GetStorageShape();
        const gert::Shape gStorageShape = ctx_.gShape->GetStorageShape();

        OP_CHECK_IF(CompareShape(vStorageShape, duStorageShape, "v", "du", BWD_DIM_NUM_4) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(betaStorageShape, gStorageShape, "beta", "g", BWD_DIM_NUM_3) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(vStorageShape, betaStorageShape, "v", "beta", BWD_DIM_NUM_3) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(vStorageShape, aStorageShape, "v", "A", BWD_DIM_NUM_3) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(vStorageShape, dwStorageShape, "v", "dw", BWD_DIM_NUM_3) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(BWD_DIM_0) != vStorageShape.GetDim(BWD_DIM_0) ||
                        kStorageShape.GetDim(BWD_DIM_2) != vStorageShape.GetDim(BWD_DIM_2),
                    OP_LOGE(ctx_.nodeName, "k and v must share batch and time dimensions."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dwStorageShape.GetDim(BWD_DIM_3) != kStorageShape.GetDim(BWD_DIM_3),
                    OP_LOGE(ctx_.nodeName, "dw last dimension must match k last dimension."),
                    return ge::GRAPH_FAILED);

        tiling_.B = static_cast<int64_t>(vStorageShape.GetDim(BWD_DIM_0));
        tiling_.HV = static_cast<int64_t>(vStorageShape.GetDim(BWD_DIM_1));
        tiling_.HK = static_cast<int64_t>(kStorageShape.GetDim(BWD_DIM_1));
        tiling_.T = static_cast<int64_t>(vStorageShape.GetDim(BWD_DIM_2));
        tiling_.K = static_cast<int64_t>(kStorageShape.GetDim(BWD_DIM_3));
        tiling_.V = static_cast<int64_t>(vStorageShape.GetDim(BWD_DIM_3));

        OP_CHECK_IF(tiling_.HK <= 0 || tiling_.HV <= 0 || tiling_.HV % tiling_.HK != 0,
                    OP_LOGE(ctx_.nodeName, "HV (%ld) must be a positive multiple of HK (%ld).", tiling_.HV,
                            tiling_.HK),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling_.K != BWD_K_DIM_128,
                    OP_LOGE(ctx_.nodeName, "Only K=128 is supported, but get %ld.", tiling_.K),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling_.V != BWD_V_DIM_128 && tiling_.V != BWD_V_DIM_256,
                    OP_LOGE(ctx_.nodeName, "Only V=128 or V=256 is supported, but get %ld.", tiling_.V),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkSize != BWD_CHUNK_SIZE_64 && ctx_.chunkSize != BWD_CHUNK_SIZE_128,
                    OP_LOGE(ctx_.nodeName, "chunk_size should be 64 or 128, but get %ld.", ctx_.chunkSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(aStorageShape.GetDim(BWD_DIM_3) != static_cast<size_t>(ctx_.chunkSize),
                    OP_LOGE(ctx_.nodeName, "A last dimension should equal chunk_size (%ld), but get %zu.",
                            ctx_.chunkSize, aStorageShape.GetDim(BWD_DIM_3)),
                    return ge::GRAPH_FAILED);

        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.headGroup = tiling_.HV / tiling_.HK;
        tiling_.taskPairMode = tiling_.headGroup == 1 ? GDN::BWD_TASK_PAIR_INDEPENDENT_UNIT :
                                                        GDN::BWD_TASK_PAIR_SAME_OWNER_HV;

        uint64_t rowTile = SelectRowTile(ctx_.ubSize);
        uint64_t rowTileInput = rowTile;
        uint64_t rowTileDa = rowTile;
        uint64_t rowTileFull = rowTile;
        tiling_.rowTileInput = static_cast<int64_t>(rowTileInput);
        tiling_.rowTileDa = static_cast<int64_t>(rowTileDa);
        tiling_.rowTileFullK = static_cast<int64_t>(rowTileFull);
        tiling_.rowTileFullV = static_cast<int64_t>(rowTileFull);
        tiling_.rowTileFullKkt = static_cast<int64_t>(rowTileFull);
        SetVectorUbLayout(rowTile);
        SetA5PipelinePolicy();
        OP_CHECK_IF(tiling_.a5UseSlotWork == 0,
                    OP_LOGE(ctx_.nodeName, "PrepareWyReprBwd A5 vector slot-work layout does not fit."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling_.a5UseL1Resident == 0,
                    OP_LOGE(ctx_.nodeName, "PrepareWyReprBwd A5 L1 resident layout does not fit."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint64_t>(tiling_.vectorUbBytes) > ctx_.ubSize,
                    OP_LOGE(ctx_.nodeName, "PrepareWyReprBwd vector UB requires %ld bytes, but UB size is %lu.",
                            tiling_.vectorUbBytes, ctx_.ubSize),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus FixLenTiling()
    {
        tiling_.chunkNum = tiling_.B * CeilDiv(tiling_.T, tiling_.chunkSize);
        tiling_.seqNum = tiling_.B;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.chunkIndicesShape, BWD_DIM_NUM_1, "chunk_indices") !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, BWD_DIM_NUM_1, "cu_seqlens") != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.cuSeqlensData == nullptr || ctx_.chunkIndicesData == nullptr,
                    OP_LOGE(ctx_.nodeName,
                            "Variable-length tiling requires host-visible cu_seqlens and chunk_indices data."),
                    return ge::GRAPH_FAILED);

        const gert::Shape seqlensStorageShape = ctx_.cuSeqlensShape->GetStorageShape();
        int64_t seqlensDim0 = static_cast<int64_t>(seqlensStorageShape.GetDim(BWD_DIM_0));
        OP_CHECK_IF(seqlensDim0 < 2,
                    OP_LOGE(ctx_.nodeName, "cu_seqlens dim 0 should be larger than 1, but get %ld.", seqlensDim0),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.cuSeqlensData[0] != 0,
                    OP_LOGE(ctx_.nodeName, "cu_seqlens[0] should be 0, but get %ld.", ctx_.cuSeqlensData[0]),
                    return ge::GRAPH_FAILED);

        std::vector<int64_t> expectChunkIndices;
        for (int64_t seq = 1; seq < seqlensDim0; ++seq) {
            int64_t curSeqLen = ctx_.cuSeqlensData[seq] - ctx_.cuSeqlensData[seq - 1];
            OP_CHECK_IF(curSeqLen <= 0,
                        OP_LOGE(ctx_.nodeName, "cu_seqlens should be strictly increasing."),
                        return ge::GRAPH_FAILED);
            for (int64_t token = 0; token < curSeqLen; token += tiling_.chunkSize) {
                expectChunkIndices.push_back(seq - 1);
                expectChunkIndices.push_back(token / tiling_.chunkSize);
            }
        }

        const gert::Shape chunkIndicesStorageShape = ctx_.chunkIndicesShape->GetStorageShape();
        int64_t chunkIndicesDim0 = static_cast<int64_t>(chunkIndicesStorageShape.GetDim(BWD_DIM_0));
        OP_CHECK_IF(chunkIndicesDim0 != static_cast<int64_t>(expectChunkIndices.size()),
                    OP_LOGE(ctx_.nodeName,
                            "chunk_indices length should be %zu, but get %ld.", expectChunkIndices.size(),
                            chunkIndicesDim0),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < static_cast<int64_t>(expectChunkIndices.size()); ++i) {
            OP_CHECK_IF(expectChunkIndices[i] != ctx_.chunkIndicesData[i],
                        OP_LOGE(ctx_.nodeName, "chunk_indices[%ld] should be %ld, but get %ld.", i,
                                expectChunkIndices[i], ctx_.chunkIndicesData[i]),
                        return ge::GRAPH_FAILED);
        }
        tiling_.chunkNum = chunkIndicesDim0 / 2;
        tiling_.seqNum = seqlensDim0 - 1;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus WorkspaceTiling()
    {
        OP_CHECK_IF(ctx_.aicCoreNum == 0,
                    OP_LOGE(ctx_.nodeName, "AIC core num should be larger than 0."),
                    return ge::GRAPH_FAILED);
        uint64_t ownerUnitNum = static_cast<uint64_t>(tiling_.chunkNum) * static_cast<uint64_t>(tiling_.HK);
        blockDim_ = ctx_.aicCoreNum;
        tiling_.usedCoreNum = static_cast<int64_t>(blockDim_);
        tiling_.ownerUnitNum = static_cast<int64_t>(ownerUnitNum);

        uint64_t elemBytes = DtypeSize(ctx_.kDataType);
        uint64_t slotRows = static_cast<uint64_t>(tiling_.HV) * static_cast<uint64_t>(tiling_.chunkSize);
        uint64_t kBytes = AlignUp(slotRows * static_cast<uint64_t>(tiling_.K) * elemBytes, BWD_ONE_BLOCK_32);
        uint64_t vBytes = AlignUp(slotRows * static_cast<uint64_t>(tiling_.V) * elemBytes, BWD_ONE_BLOCK_32);
        uint64_t aBytes = AlignUp(slotRows * static_cast<uint64_t>(tiling_.chunkSize) * elemBytes,
                                  BWD_ONE_BLOCK_32);

        uint64_t offset = 0;
        tiling_.kbgOffset = static_cast<int64_t>(offset);
        offset += kBytes;
        tiling_.vbOffset = static_cast<int64_t>(offset);
        offset += vBytes;
        tiling_.kbetaOffset = static_cast<int64_t>(offset);
        offset += kBytes;
        tiling_.da1Offset = static_cast<int64_t>(offset);
        offset += aBytes;
        tiling_.da2Offset = static_cast<int64_t>(offset);
        offset += aBytes;
        tiling_.da4Offset = static_cast<int64_t>(offset);
        offset += aBytes;
        tiling_.da5Offset = static_cast<int64_t>(offset);
        offset += aBytes;
        tiling_.daOffset = static_cast<int64_t>(offset);
        offset += aBytes;
        tiling_.fullDkOffset = static_cast<int64_t>(offset);
        offset += kBytes;
        tiling_.fullDkbOffset = static_cast<int64_t>(offset);
        offset += kBytes;
        tiling_.fullDkbgOffset = static_cast<int64_t>(offset);
        offset += kBytes;
        tiling_.fullDvbOffset = static_cast<int64_t>(offset);
        offset += vBytes;
        tiling_.fullKktOffset = static_cast<int64_t>(offset);
        offset += aBytes;

        tiling_.kWorkspaceBytes = static_cast<int64_t>(kBytes);
        tiling_.vWorkspaceBytes = static_cast<int64_t>(vBytes);
        tiling_.aWorkspaceBytes = static_cast<int64_t>(aBytes);
        tiling_.workspaceSlotBytes = static_cast<int64_t>(offset);
        tiling_.workspaceBufferCount = 2;
        tiling_.userWorkspaceBytes =
            static_cast<int64_t>(static_cast<uint64_t>(tiling_.usedCoreNum) *
                                 static_cast<uint64_t>(tiling_.workspaceBufferCount) * offset);
        workspaceSize_ = ctx_.sysWorkspaceSize + static_cast<size_t>(tiling_.userWorkspaceBytes);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        if (IsVariableLength()) {
            OP_CHECK_IF(ctx_.cuSeqlensShape == nullptr || ctx_.chunkIndicesShape == nullptr,
                        OP_LOGE(ctx_.nodeName,
                                "Variable-length mode requires both cu_seqlens and chunk_indices."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(tiling_.B != BWD_VAR_LEN_B_DIM_1,
                        OP_LOGE(ctx_.nodeName, "Variable-length mode requires B=1, but get %ld.", tiling_.B),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 1;
        } else {
            OP_CHECK_IF(FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 0;
        }
        OP_CHECK_IF(WorkspaceTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // PREPARE_WY_REPR_BWD_TILING_PROCESSOR_H

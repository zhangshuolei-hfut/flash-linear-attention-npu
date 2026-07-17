/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_VECTOR_H
#define PREPARE_WY_REPR_BWD_VECTOR_H

#include <type_traits>

#include "adv_api/index/arithprogression.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"
#include "prepare_wy_repr_bwd_struct.h"

namespace GDN {

static constexpr uint32_t BWD_VEC_ONE_BLOCK_32 = 32;
static constexpr uint32_t BWD_VEC_FP32_PER_BLOCK_8 = 8;
static constexpr uint32_t BWD_VEC_FP32_PER_REPEAT_64 = 64;
static constexpr uint32_t BWD_VEC_BIT_NUM_FOR_UINT8 = 8;
static constexpr uint32_t BWD_VEC_SIZE_FLOAT = 4;
static constexpr uint32_t BWD_VEC_CAL_NUM_FLOAT = 64;
static constexpr uint32_t BWD_VEC_IO_BUFFER_COUNT = 2;

template <typename T, typename GateT, bool IS_VARLEN, int V>
class PrepareWyReprBwdVector {
public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dw, GM_ADDR du, GM_ADDR g,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta,
                                GM_ADDR dg, GM_ADDR workspace, const PrepareWyReprBwdTilingData *tiling,
                                AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        coreIdx_ = static_cast<uint64_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
        coreNum_ = static_cast<uint64_t>(tiling_->usedCoreNum);
        subBlockIdx_ = static_cast<uint64_t>(AscendC::GetSubBlockIdx());
        subBlockNum_ = static_cast<uint64_t>(AscendC::GetSubBlockNum());

        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(A));
        dwGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dw));
        duGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(du));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(beta));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dk));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dv));
        dbetaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(dbeta));
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(dg));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
        if (cuSeqlens != nullptr) {
            cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(cuSeqlens));
        }
        if (chunkIndices != nullptr) {
            chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(chunkIndices));
        }
    }

    __aicore__ inline void ProcessPipeline()
    {
        constexpr uint64_t slotCount = 2;
        InitPipelineBuffers();
        AscendC::Duplicate<float>(ZeroTensor(), 0.0f, BWD_VEC_ONE_BLOCK_32 / BWD_VEC_SIZE_FLOAT);
        AscendC::PipeBarrier<PIPE_V>();
        InitTriangleMasks();

        const uint64_t coreLoops = static_cast<uint64_t>(tiling_->chunkNum);
        for (uint64_t chunkLinear = coreIdx_; chunkLinear < coreLoops; chunkLinear += coreNum_) {
            uint64_t batch = 0;
            uint64_t chunkBegin = 0;
            uint64_t chunkEnd = 0;
            ResolveChunk(chunkLinear, batch, chunkBegin, chunkEnd);
            if (chunkEnd <= chunkBegin) {
                continue;
            }
            uint32_t chunkLen = static_cast<uint32_t>(chunkEnd - chunkBegin);
            for (uint64_t headBase = 0; headBase < static_cast<uint64_t>(tiling_->HV); headBase += slotCount) {
                // Advance both head slots one dependency stage at a time to overlap AIC(slot 1) with AIV(slot 0).
                for (uint32_t pipelineStage = 0; pipelineStage < 4; ++pipelineStage) {
                    for (uint64_t slot = 0; slot < slotCount; ++slot) {
                    uint64_t valueHead = 0;
                    uint64_t keyHead = 0;
                    if (!ResolvePipelineHeadSlot(headBase, slot, valueHead, keyHead)) {
                        continue;
                    }
                    if (pipelineStage == 0) {
                    uint32_t kbgRowNum = static_cast<uint32_t>(tiling_->rowTileInput);
                    uint32_t kbgKDim = static_cast<uint32_t>(tiling_->K);
                    uint32_t kbgVDim = static_cast<uint32_t>(tiling_->V);
                    uint32_t kbgMaxDim = kbgKDim > kbgVDim ? kbgKDim : kbgVDim;
                    kbgMaxDim = kbgMaxDim > static_cast<uint32_t>(tiling_->chunkSize) ? kbgMaxDim :
                                static_cast<uint32_t>(tiling_->chunkSize);
                    auto kbgBetaCache = SlotBetaCache(slot);
                    auto kbgGCache = SlotGCache(slot);
                    auto kbgGExpCache = SlotGExpCache(slot);
                    LoadGateToFp32(kbgBetaCache, GateOffset(batch, valueHead, chunkBegin), chunkLen);
                    LoadGToFp32(kbgGCache, GateOffset(batch, valueHead, chunkBegin), chunkLen);

                    auto kbgTemp = TempTensor();
                    auto kbgKScaled = kbgTemp;
                    auto kbgVScaled = kbgKScaled[kbgRowNum * kbgMaxDim];
                    auto kbgInputTile = kbgVScaled[kbgRowNum * kbgMaxDim];
                    uint32_t kbgTempTileCount = UseInputCache() ? 2U : 3U;
                    auto kbgBetaExpFp32 = kbgTemp[kbgTempTileCount * kbgRowNum * kbgMaxDim];
                    auto kbgBetaBrcb = kbgBetaExpFp32[kbgRowNum];
                    auto kbgBetaExpBrcb = kbgBetaBrcb[kbgRowNum * BWD_VEC_FP32_PER_BLOCK_8];

                    uint32_t kbgRowStep = kbgRowNum * static_cast<uint32_t>(subBlockNum_);
                    uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * kbgRowNum;
                    uint32_t kIoSlot = 0;
                    if (rowOffset < chunkLen) {
                        uint32_t firstRowNum = rowOffset + kbgRowNum > chunkLen ? chunkLen - rowOffset : kbgRowNum;
                        kIoSlot = BeginLoadGmTile(
                            kGm_, KeyOffset(batch, keyHead, chunkBegin + rowOffset, 0), kbgKDim * firstRowNum);
                    }
                    for (; rowOffset < chunkLen; rowOffset += kbgRowStep) {
                        uint32_t curRowNum = rowOffset + kbgRowNum > chunkLen ? chunkLen - rowOffset : kbgRowNum;
                        uint64_t token = chunkBegin + rowOffset;
                        AscendC::Adds(kbgGExpCache[rowOffset], kbgGCache[rowOffset], 0.0f, curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Exp(kbgGExpCache[rowOffset], kbgGExpCache[rowOffset], curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(kbgBetaExpFp32, kbgGExpCache[rowOffset], kbgBetaCache[rowOffset], curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Brcb(kbgBetaBrcb, kbgBetaCache[rowOffset],
                                      static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                        AscendC::Brcb(kbgBetaExpBrcb, kbgBetaExpFp32,
                                      static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                        auto kbgKInput = UseInputCache() ?
                                         SlotKCache(slot)[InputCacheElemOffset(rowOffset, kbgKDim)] :
                                         kbgInputTile;
                        FinishLoadTile(kbgKInput, kIoSlot, kbgKDim * curRowNum);
                        uint32_t vIoSlot = BeginLoadGmTile(
                            vGm_, ValueOffset(batch, valueHead, token, 0), kbgVDim * curRowNum);
                        uint64_t perCol = 0;
                        uint8_t kRepeatStride = kbgKDim * sizeof(float32_t) / BWD_VEC_ONE_BLOCK_32;
                        while (perCol < kbgKDim) {
                            AscendC::Mul(kbgKScaled[perCol], kbgKInput[perCol], kbgBetaExpBrcb,
                                         BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                         {1, 1, 0, kRepeatStride, kRepeatStride, 1});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        StoreGmTile(workspaceGm_, SlotWorkspaceElem(
                                    slot, tiling_->kbgOffset, SlotValueKeyLikeOffset(valueHead, rowOffset, 0)),
                                    kbgKScaled, kbgKDim * curRowNum);
                        perCol = 0;
                        while (perCol < kbgKDim) {
                            AscendC::Mul(kbgKScaled[perCol], kbgKInput[perCol], kbgBetaBrcb,
                                         BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                         {1, 1, 0, kRepeatStride, kRepeatStride, 1});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        StoreGmTile(workspaceGm_, SlotWorkspaceElem(
                                    slot, tiling_->kbetaOffset, SlotValueKeyLikeOffset(valueHead, rowOffset, 0)),
                                    kbgKScaled, kbgKDim * curRowNum);
                        auto kbgVInput = UseInputCache() ?
                                         SlotVCache(slot)[InputCacheElemOffset(rowOffset, kbgVDim)] :
                                         kbgInputTile;
                        FinishLoadTile(kbgVInput, vIoSlot, kbgVDim * curRowNum);
                        uint32_t nextKIoSlot = 0;
                        uint32_t nextRowOffset = rowOffset + kbgRowStep;
                        if (nextRowOffset < chunkLen) {
                            uint32_t nextRowNum = nextRowOffset + kbgRowNum > chunkLen ? chunkLen - nextRowOffset :
                                                  kbgRowNum;
                            nextKIoSlot = BeginLoadGmTile(
                                kGm_, KeyOffset(batch, keyHead, chunkBegin + nextRowOffset, 0),
                                kbgKDim * nextRowNum);
                        }
                        perCol = 0;
                        uint8_t vRepeatStride = kbgVDim * sizeof(float32_t) / BWD_VEC_ONE_BLOCK_32;
                        while (perCol < kbgVDim) {
                            AscendC::Mul(kbgVScaled[perCol], kbgVInput[perCol], kbgBetaBrcb,
                                         BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                         {1, 1, 0, vRepeatStride, vRepeatStride, 1});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        StoreGmTile(workspaceGm_, SlotWorkspaceElem(
                                    slot, tiling_->vbOffset, SlotValueOffset(valueHead, rowOffset, 0)),
                                    kbgVScaled, kbgVDim * curRowNum);
                        kIoSlot = nextKIoSlot;
                    }
                    AscendC::PipeBarrier<PIPE_MTE3>();
                    SignalAivFinishStore();

                    } else if (pipelineStage == 1) {
                    WaitAicFinishStore();

                    uint32_t da4RowNum = static_cast<uint32_t>(tiling_->rowTileDa);
                    uint32_t da4ChunkSize = static_cast<uint32_t>(tiling_->chunkSize);
                    auto da4Temp = TempTensor();
                    auto da4Da1 = da4Temp;
                    auto da4Da2 = da4Temp[da4RowNum * da4ChunkSize];
                    auto da4Sum = da4Temp[2 * da4RowNum * da4ChunkSize];
                    auto da4Mask = LowerMaskTensor();
                    auto da4Zero = ZeroTensor();
                    for (uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * da4RowNum; rowOffset < chunkLen;
                         rowOffset += da4RowNum * static_cast<uint32_t>(subBlockNum_)) {
                        uint32_t curRowNum = rowOffset + da4RowNum > chunkLen ? chunkLen - rowOffset : da4RowNum;
                        LoadMatrixTile(da4Da1, slot, tiling_->da1Offset, SlotMatrixOffset(valueHead, rowOffset, 0),
                                       da4ChunkSize * curRowNum);
                        LoadMatrixTile(da4Da2, slot, tiling_->da2Offset, SlotMatrixOffset(valueHead, rowOffset, 0),
                                       da4ChunkSize * curRowNum);
                        AscendC::Add(da4Sum, da4Da1, da4Da2, da4ChunkSize * curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
                        AscendC::Select(da4Sum, da4Mask[rowOffset * da4ChunkSize / BWD_VEC_BIT_NUM_FOR_UINT8],
                                        da4Zero, da4Sum, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                        BWD_VEC_CAL_NUM_FLOAT,
                                        curRowNum * da4ChunkSize / BWD_VEC_CAL_NUM_FLOAT, repeatParams);
                        StoreGmTile(workspaceGm_, SlotWorkspaceElem(
                                    slot, tiling_->da4Offset, SlotMatrixOffset(valueHead, rowOffset, 0)),
                                    da4Sum, da4ChunkSize * curRowNum);
                    }
                    AscendC::PipeBarrier<PIPE_MTE3>();
                    SignalAivFinishStore();

                    } else if (pipelineStage == 2) {
                    uint32_t finalDaRowNum = static_cast<uint32_t>(tiling_->rowTileDa);
                    uint32_t finalDaChunkSize = static_cast<uint32_t>(tiling_->chunkSize);
                    auto finalDaTemp = TempTensor();
                    auto finalDa = finalDaTemp;
                    auto finalDaGAll = SlotGCache(slot);
                    auto finalDaGBrcb = finalDaTemp[finalDaRowNum * finalDaChunkSize];
                    auto finalDaFactor = finalDaGBrcb[finalDaRowNum * BWD_VEC_FP32_PER_BLOCK_8];
                    auto finalDaMask = UpperMaskTensor();
                    auto finalDaZero = ZeroTensor();
                    WaitAicFinishStore();

                    for (uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * finalDaRowNum;
                         rowOffset < chunkLen; rowOffset += finalDaRowNum * static_cast<uint32_t>(subBlockNum_)) {
                        uint32_t curRowNum = rowOffset + finalDaRowNum > chunkLen ? chunkLen - rowOffset :
                                             finalDaRowNum;
                        uint32_t finalDaIoSlot = BeginLoadMatrixTile(
                            slot, tiling_->daOffset, SlotMatrixOffset(valueHead, rowOffset, 0),
                            finalDaChunkSize * curRowNum);

                        uint64_t perCol = 0;
                        uint8_t repeatStride = finalDaChunkSize * sizeof(float32_t) / BWD_VEC_ONE_BLOCK_32;
                        while (perCol < finalDaChunkSize) {
                            AscendC::Copy(finalDaFactor[perCol], finalDaGAll[perCol],
                                          BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                          {1, 1, repeatStride, 0});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        AscendC::Brcb(finalDaGBrcb, finalDaGAll[rowOffset],
                                      static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                        AscendC::PipeBarrier<PIPE_V>();
                        perCol = 0;
                        while (perCol < finalDaChunkSize) {
                            AscendC::Sub(finalDaFactor[perCol], finalDaFactor[perCol], finalDaGBrcb,
                                         BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                         {1, 1, 0, repeatStride, repeatStride, 1});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mins(finalDaFactor, finalDaFactor, 0.0f, curRowNum * finalDaChunkSize);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Exp(finalDaFactor, finalDaFactor, curRowNum * finalDaChunkSize);
                        AscendC::PipeBarrier<PIPE_V>();
                        FinishLoadTile(finalDa, finalDaIoSlot, finalDaChunkSize * curRowNum);
                        AscendC::Muls(finalDa, finalDa, -1.0f, curRowNum * finalDaChunkSize);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(finalDa, finalDa, finalDaFactor, curRowNum * finalDaChunkSize);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
                        AscendC::Select(finalDa,
                                        finalDaMask[rowOffset * finalDaChunkSize / BWD_VEC_BIT_NUM_FOR_UINT8],
                                        finalDaZero, finalDa, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                        BWD_VEC_CAL_NUM_FLOAT,
                                        curRowNum * finalDaChunkSize / BWD_VEC_CAL_NUM_FLOAT, repeatParams);
                        StoreFinalDaTile(slot, rowOffset, workspaceGm_, SlotWorkspaceElem(
                                         slot, tiling_->daOffset, SlotMatrixOffset(valueHead, rowOffset, 0)),
                                         finalDa, finalDaChunkSize * curRowNum);
                    }
                    AscendC::PipeBarrier<PIPE_MTE3>();
                    SignalAivFinishStore();

                    } else {
                    uint32_t outputKDim = static_cast<uint32_t>(tiling_->K);
                    uint32_t outputVDim = static_cast<uint32_t>(tiling_->V);
                    uint32_t outputChunkSize = static_cast<uint32_t>(tiling_->chunkSize);
                    uint32_t outputRowK = static_cast<uint32_t>(tiling_->rowTileFullK);
                    uint32_t outputRowV = static_cast<uint32_t>(tiling_->rowTileFullV);
                    uint32_t outputRowKkt = static_cast<uint32_t>(tiling_->rowTileFullKkt);
                    uint32_t outputRowOwned = outputRowK < outputRowV ? outputRowK : outputRowV;
                    uint32_t wholeReduceKCnt =
                        static_cast<uint32_t>(CeilDiv(outputKDim, BWD_VEC_FP32_PER_REPEAT_64));
                    uint32_t wholeReduceVCnt =
                        static_cast<uint32_t>(CeilDiv(outputVDim, BWD_VEC_FP32_PER_REPEAT_64));
                    uint32_t wholeReduceChunkCnt =
                        static_cast<uint32_t>(CeilDiv(chunkLen, BWD_VEC_FP32_PER_REPEAT_64));
                    uint64_t persistentStride =
                        CeilDiv(tiling_->chunkSize * sizeof(float32_t), BWD_VEC_ONE_BLOCK_32) *
                        BWD_VEC_ONE_BLOCK_32 / sizeof(float32_t);
                    auto outputPersistent = PersistentTensor();
                    auto outputBetaFp32 = SlotBetaCache(slot);
                    auto outputGExpFp32 = SlotGExpCache(slot);
                    auto outputDbetaAcc = outputPersistent[2 * persistentStride];
                    auto outputDgAcc = outputPersistent[3 * persistentStride];
                    WaitAicFinishFullStore();

                    bool waitedFullDkbReady = false;
                    bool waitedFullDkbgReady = false;

                    auto kOutTemp = TempTensor();
                    for (uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * outputRowOwned;
                         rowOffset < chunkLen; rowOffset += outputRowOwned * static_cast<uint32_t>(subBlockNum_)) {
                        uint32_t curRowNum = rowOffset + outputRowOwned > chunkLen ? chunkLen - rowOffset :
                                             outputRowOwned;
                        uint64_t rowElem = outputRowOwned * outputKDim;
                        auto kOutDkb = kOutTemp;
                        auto kOutDkbg = kOutTemp[rowElem];
                        auto kOutProd = kOutTemp[2 * rowElem];
                        auto kOutDk = kOutTemp[3 * rowElem];
                        auto kOutK = UseInputCache() ?
                                     SlotKCache(slot)[InputCacheElemOffset(rowOffset, outputKDim)] :
                                     kOutTemp[4 * rowElem];
                        uint32_t kOutTempTileCount = UseInputCache() ? 4U : 5U;
                        auto kOutBetaBrcb = kOutTemp[kOutTempTileCount * rowElem];
                        auto kOutGBrcb = kOutBetaBrcb[outputRowOwned * BWD_VEC_FP32_PER_BLOCK_8];
                        auto kOutReduce = kOutGBrcb[outputRowOwned * BWD_VEC_FP32_PER_BLOCK_8];
                        uint64_t token = chunkBegin + rowOffset;

                        uint32_t dkIoSlot = BeginLoadMatrixTile(
                            slot, tiling_->fullDkOffset, SlotValueKeyLikeOffset(valueHead, rowOffset, 0),
                            outputKDim * curRowNum);
                        uint32_t kIoSlot = 0;
                        if (!UseInputCache()) {
                            kIoSlot = BeginLoadGmTile(
                                kGm_, KeyOffset(batch, keyHead, token, 0), outputKDim * curRowNum);
                        }
                        AscendC::Brcb(kOutBetaBrcb, outputBetaFp32[rowOffset],
                                      static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                        AscendC::Brcb(kOutGBrcb, outputGExpFp32[rowOffset],
                                      static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(kOutGBrcb, kOutGBrcb, kOutBetaBrcb,
                                     curRowNum * BWD_VEC_FP32_PER_BLOCK_8);
                        uint32_t dkbIoSlot = 0;
                        if (UseInputCache()) {
                            if (!waitedFullDkbReady) {
                                WaitAicFinishFullStore();
                                waitedFullDkbReady = true;
                            }
                            dkbIoSlot = BeginLoadMatrixTile(
                                slot, tiling_->fullDkbOffset,
                                SlotValueKeyLikeOffset(valueHead, rowOffset, 0), outputKDim * curRowNum);
                        }
                        FinishLoadTile(kOutDk, dkIoSlot, outputKDim * curRowNum);
                        if (!UseInputCache()) {
                            FinishLoadTile(kOutK, kIoSlot, outputKDim * curRowNum);
                            if (!waitedFullDkbReady) {
                                WaitAicFinishFullStore();
                                waitedFullDkbReady = true;
                            }
                            dkbIoSlot = BeginLoadMatrixTile(
                                slot, tiling_->fullDkbOffset,
                                SlotValueKeyLikeOffset(valueHead, rowOffset, 0), outputKDim * curRowNum);
                        }
                        FinishLoadTile(kOutDkb, dkbIoSlot, outputKDim * curRowNum);
                        AscendC::Mul(kOutProd, kOutDkb, kOutK, outputKDim * curRowNum);
                        ReduceRowsToVector(kOutProd, kOutReduce, outputKDim, curRowNum, wholeReduceKCnt);
                        AscendC::Adds(outputDbetaAcc[rowOffset], kOutProd, 0.0f, curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        uint64_t perCol = 0;
                        uint8_t repeatStride = outputKDim * sizeof(float32_t) / BWD_VEC_ONE_BLOCK_32;
                        while (perCol < outputKDim) {
                            AscendC::Mul(kOutDkb[perCol], kOutDkb[perCol], kOutBetaBrcb,
                                         BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                         {1, 1, 0, repeatStride, repeatStride, 1});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        if (!waitedFullDkbgReady) {
                            WaitAicFinishFullStore();
                            waitedFullDkbgReady = true;
                        }
                        uint32_t dkbgIoSlot = BeginLoadMatrixTile(
                            slot, tiling_->fullDkbgOffset,
                            SlotValueKeyLikeOffset(valueHead, rowOffset, 0), outputKDim * curRowNum);
                        FinishLoadTile(kOutDkbg, dkbgIoSlot, outputKDim * curRowNum);
                        AscendC::Mul(kOutProd, kOutDkbg, kOutK, outputKDim * curRowNum);
                        ReduceRowsToVector(kOutProd, kOutReduce, outputKDim, curRowNum, wholeReduceKCnt);
                        AscendC::Mul(kOutProd, kOutProd, outputGExpFp32[rowOffset], curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(outputDbetaAcc[rowOffset], outputDbetaAcc[rowOffset], kOutProd, curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(kOutProd, kOutProd, outputBetaFp32[rowOffset], curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Adds(outputDgAcc[rowOffset], kOutProd, 0.0f, curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        perCol = 0;
                        while (perCol < outputKDim) {
                            AscendC::Mul(kOutDkbg[perCol], kOutDkbg[perCol], kOutGBrcb,
                                         BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                         {1, 1, 0, repeatStride, repeatStride, 1});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(kOutDkb, kOutDkb, kOutDkbg, outputKDim * curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(kOutDk, kOutDk, kOutDkb, outputKDim * curRowNum);
                        if (valueHead % static_cast<uint64_t>(tiling_->headGroup) != 0) {
                            LoadGmTile(kOutProd, dkGm_, KeyOffset(batch, keyHead, token, 0),
                                       outputKDim * curRowNum);
                            AscendC::Add(kOutDk, kOutDk, kOutProd, outputKDim * curRowNum);
                        }
                        StoreGmTile(dkGm_, KeyOffset(batch, keyHead, token, 0), kOutDk,
                                    outputKDim * curRowNum);
                    }
                    if (!waitedFullDkbReady) {
                        WaitAicFinishFullStore();
                    }
                    if (!waitedFullDkbgReady) {
                        WaitAicFinishFullStore();
                    }

                    WaitAicFinishFullStore();
                        auto vOutTemp = TempTensor();
                    for (uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * outputRowOwned;
                         rowOffset < chunkLen; rowOffset += outputRowOwned * static_cast<uint32_t>(subBlockNum_)) {
                        uint32_t curRowNum = rowOffset + outputRowOwned > chunkLen ? chunkLen - rowOffset :
                                             outputRowOwned;
                        uint64_t rowElem = outputRowOwned * outputVDim;
                        auto vOutDvb = vOutTemp;
                        auto vOutProd = vOutTemp[rowElem];
                        auto vOutV = UseInputCache() ?
                                     SlotVCache(slot)[InputCacheElemOffset(rowOffset, outputVDim)] :
                                     vOutTemp[2 * rowElem];
                        uint32_t vOutTempTileCount = UseInputCache() ? 2U : 3U;
                        auto vOutBetaBrcb = vOutTemp[vOutTempTileCount * rowElem];
                        auto vOutReduce = vOutBetaBrcb[outputRowOwned * BWD_VEC_FP32_PER_BLOCK_8];
                        uint64_t token = chunkBegin + rowOffset;
                        uint32_t dvbIoSlot = BeginLoadMatrixTile(
                            slot, tiling_->fullDvbOffset, SlotValueOffset(valueHead, rowOffset, 0),
                            outputVDim * curRowNum);
                        uint32_t vIoSlot = 0;
                        if (!UseInputCache()) {
                            vIoSlot = BeginLoadGmTile(
                                vGm_, ValueOffset(batch, valueHead, token, 0), outputVDim * curRowNum);
                        }
                        AscendC::Brcb(vOutBetaBrcb, outputBetaFp32[rowOffset],
                                      static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                        FinishLoadTile(vOutDvb, dvbIoSlot, outputVDim * curRowNum);
                        if (!UseInputCache()) {
                            FinishLoadTile(vOutV, vIoSlot, outputVDim * curRowNum);
                        }
                        AscendC::Mul(vOutProd, vOutV, vOutDvb, outputVDim * curRowNum);
                        ReduceRowsToVector(vOutProd, vOutReduce, outputVDim, curRowNum, wholeReduceVCnt);
                        AscendC::Add(outputDbetaAcc[rowOffset], outputDbetaAcc[rowOffset], vOutProd, curRowNum);
                        uint64_t perCol = 0;
                        uint8_t repeatStride = outputVDim * sizeof(float32_t) / BWD_VEC_ONE_BLOCK_32;
                        while (perCol < outputVDim) {
                            AscendC::Mul(vOutDvb[perCol], vOutDvb[perCol], vOutBetaBrcb,
                                         BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                         {1, 1, 0, repeatStride, repeatStride, 1});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        StoreGmTile(dvGm_, ValueOffset(batch, valueHead, token, 0), vOutDvb,
                                    outputVDim * curRowNum);
                    }
                    WaitAicFinishFullStore();
                    auto kktTemp = TempTensor();
                    auto kktAll = kktTemp;
                    auto kktDa = kktTemp[outputChunkSize * outputChunkSize];
                    auto kktRowSum = kktDa[outputRowKkt * outputChunkSize];
                    auto kktReduce = kktRowSum[outputRowKkt];
                    uint32_t kktRemainCnt = chunkLen % BWD_VEC_FP32_PER_REPEAT_64;
                    bool hasKktTail = kktRemainCnt > 0;
                    uint32_t kktTailOffset = wholeReduceChunkCnt * BWD_VEC_FP32_PER_REPEAT_64 -
                                             BWD_VEC_FP32_PER_REPEAT_64;
                    uint64_t kktTailMask[1] = {0xffffffffffffffff};
                    if (hasKktTail) {
                        kktTailMask[0] <<= kktRemainCnt;
                    }
                    uint32_t kktIoSlot = BeginLoadMatrixTile(
                        slot, tiling_->fullKktOffset, SlotMatrixOffset(valueHead, 0, 0),
                        outputChunkSize * (outputRowKkt > chunkLen ? chunkLen : outputRowKkt));
                    bool daFromCache = static_cast<uint32_t>(subBlockIdx_) == 0U;
                    uint32_t daIoSlot = 0;
                    if (!daFromCache) {
                        daIoSlot = BeginLoadMatrixTile(
                            slot, tiling_->daOffset, SlotMatrixOffset(valueHead, 0, 0),
                            outputChunkSize * (outputRowKkt > chunkLen ? chunkLen : outputRowKkt));
                    }
                    for (uint32_t rowOffset = 0; rowOffset < chunkLen; rowOffset += outputRowKkt) {
                        uint32_t curRowNum = rowOffset + outputRowKkt > chunkLen ? chunkLen - rowOffset :
                                             outputRowKkt;
                        if (daFromCache) {
                            LoadFinalDaCache(kktDa, slot, rowOffset, outputChunkSize * curRowNum);
                        } else {
                            FinishLoadTile(kktDa, daIoSlot, outputChunkSize * curRowNum);
                        }
                        FinishLoadTile(kktAll[rowOffset * outputChunkSize], kktIoSlot,
                                       outputChunkSize * curRowNum);
                        uint32_t nextRowOffset = rowOffset + outputRowKkt;
                        uint32_t nextKktIoSlot = 0;
                        uint32_t nextDaIoSlot = 0;
                        bool nextDaFromCache = false;
                        if (nextRowOffset < chunkLen) {
                            uint32_t nextRowNum = nextRowOffset + outputRowKkt > chunkLen ?
                                                  chunkLen - nextRowOffset : outputRowKkt;
                            nextDaFromCache =
                                (nextRowOffset / static_cast<uint32_t>(tiling_->rowTileDa)) %
                                    static_cast<uint32_t>(subBlockNum_) ==
                                static_cast<uint32_t>(subBlockIdx_);
                            if (!nextDaFromCache) {
                                nextDaIoSlot = BeginLoadMatrixTile(
                                    slot, tiling_->daOffset,
                                    SlotMatrixOffset(valueHead, nextRowOffset, 0),
                                    outputChunkSize * nextRowNum);
                            }
                            nextKktIoSlot = BeginLoadMatrixTile(
                                slot, tiling_->fullKktOffset,
                                SlotMatrixOffset(valueHead, nextRowOffset, 0),
                                outputChunkSize * nextRowNum);
                        }
                        uint64_t perCol = 0;
                        uint8_t repeatStride = outputChunkSize * sizeof(float32_t) / BWD_VEC_ONE_BLOCK_32;
                        while (perCol < chunkLen) {
                            AscendC::Mul(kktAll[rowOffset * outputChunkSize + perCol],
                                         kktAll[rowOffset * outputChunkSize + perCol],
                                         outputBetaFp32[perCol], BWD_VEC_FP32_PER_REPEAT_64, curRowNum,
                                         {1, 1, 1, repeatStride, repeatStride, 0});
                            perCol += BWD_VEC_FP32_PER_REPEAT_64;
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(kktAll[rowOffset * outputChunkSize], kktDa,
                                     kktAll[rowOffset * outputChunkSize], outputChunkSize * curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        if (hasKktTail) {
                            for (uint32_t row = 0; row < curRowNum; ++row) {
                                AscendC::Duplicate(kktAll[(rowOffset + row) * outputChunkSize + kktTailOffset],
                                                   0.0f, kktTailMask, 1, 1, 8);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        kktIoSlot = nextKktIoSlot;
                        daIoSlot = nextDaIoSlot;
                        daFromCache = nextDaFromCache;
                    }
                    for (uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * outputRowOwned;
                         rowOffset < chunkLen; rowOffset += outputRowOwned * static_cast<uint32_t>(subBlockNum_)) {
                        uint32_t curRowNum = rowOffset + outputRowOwned > chunkLen ? chunkLen - rowOffset :
                                             outputRowOwned;
                        for (uint32_t row = 0; row < curRowNum; ++row) {
                            AscendC::WholeReduceSum(kktReduce[(rowOffset + row) * BWD_VEC_FP32_PER_BLOCK_8],
                                                    kktAll[(rowOffset + row) * outputChunkSize],
                                                    BWD_VEC_FP32_PER_REPEAT_64, wholeReduceChunkCnt, 1, 1, 8);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::WholeReduceSum(kktRowSum, kktReduce[rowOffset * BWD_VEC_FP32_PER_BLOCK_8],
                                                wholeReduceChunkCnt, curRowNum, 1, 1, 1);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(kktRowSum, kktRowSum, -1.0f, curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(outputDgAcc[rowOffset], outputDgAcc[rowOffset], kktRowSum, curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    uint32_t remainRow = chunkLen;
                    while (remainRow > 1) {
                        uint32_t calcCnt = (remainRow / 2) * outputChunkSize;
                        remainRow = static_cast<uint32_t>(CeilDiv(remainRow, 2));
                        uint32_t offset = remainRow * outputChunkSize;
                        AscendC::Add(kktAll, kktAll, kktAll[offset], calcCnt);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    for (uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * outputRowOwned;
                         rowOffset < chunkLen; rowOffset += outputRowOwned * static_cast<uint32_t>(subBlockNum_)) {
                        uint32_t curRowNum = rowOffset + outputRowOwned > chunkLen ? chunkLen - rowOffset :
                                             outputRowOwned;
                        AscendC::Add(outputDgAcc[rowOffset], outputDgAcc[rowOffset], kktAll[rowOffset], curRowNum);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    for (uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * outputRowOwned;
                         rowOffset < chunkLen; rowOffset += outputRowOwned * static_cast<uint32_t>(subBlockNum_)) {
                        uint32_t curRowNum = rowOffset + outputRowOwned > chunkLen ? chunkLen - rowOffset :
                                             outputRowOwned;
                        StoreGateTile(dbetaGm_, GateOffset(batch, valueHead, chunkBegin + rowOffset),
                                      outputDbetaAcc[rowOffset], curRowNum);
                        StoreGateTile(dgGm_, GateOffset(batch, valueHead, chunkBegin + rowOffset),
                                      outputDgAcc[rowOffset], curRowNum);
                    }
                    AscendC::PipeBarrier<PIPE_MTE3>();
                    }
                    }
                }
            }
        }
        DrainPipelineBuffers();
    }

private:
    __aicore__ inline bool ResolvePipelineHeadSlot(uint64_t headBase, uint64_t slot,
                                                   uint64_t &valueHead, uint64_t &keyHead) const
    {
        valueHead = headBase + slot;
        if (valueHead >= static_cast<uint64_t>(tiling_->HV)) {
            return false;
        }
        keyHead = valueHead / static_cast<uint64_t>(tiling_->headGroup);
        return true;
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline int64_t CeilDiv(int64_t lhs, int64_t rhs) const
    {
        return rhs == 0 ? 0 : (lhs + rhs - 1) / rhs;
    }

    template <typename D>
    __aicore__ inline AscendC::LocalTensor<D> UbTensor(uint64_t byteOffset)
    {
        return ubBuf_.Get<D>()[byteOffset / sizeof(D)];
    }

    template <typename D>
    __aicore__ inline AscendC::LocalTensor<D> IoInTensor(uint32_t ioSlot)
    {
        return UbTensor<D>(ioInOffset_ + static_cast<uint64_t>(ioSlot) * ioInStride_);
    }

    template <typename D>
    __aicore__ inline AscendC::LocalTensor<D> IoOutTensor(uint32_t ioSlot)
    {
        return UbTensor<D>(ioOutOffset_ + static_cast<uint64_t>(ioSlot) * ioOutStride_);
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> PersistentTensor()
    {
        return UbTensor<float32_t>(persistentOffset_);
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> GateCacheTensor()
    {
        return UbTensor<float32_t>(gateCacheOffset_);
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> TempTensor()
    {
        return UbTensor<float32_t>(tempOffset_);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> MaskTensor()
    {
        return UbTensor<uint8_t>(maskOffset_);
    }

    __aicore__ inline uint32_t MaskBlocksPerRow() const
    {
        return (static_cast<uint32_t>(tiling_->chunkSize) + BWD_VEC_BIT_NUM_FOR_UINT8 - 1) /
               BWD_VEC_BIT_NUM_FOR_UINT8;
    }

    __aicore__ inline uint32_t MaskElemCount() const
    {
        return static_cast<uint32_t>(tiling_->chunkSize) * MaskBlocksPerRow();
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> LowerMaskTensor()
    {
        return MaskTensor();
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> UpperMaskTensor()
    {
        return MaskTensor()[MaskElemCount()];
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> ZeroTensor()
    {
        return UbTensor<float32_t>(zeroOffset_);
    }

    __aicore__ inline uint32_t NextIoInSlot()
    {
        uint32_t ioSlot = ioInSlot_;
        ioInSlot_ ^= 1U;
        return ioSlot;
    }

    __aicore__ inline uint32_t NextIoOutSlot()
    {
        uint32_t ioSlot = ioOutSlot_;
        ioOutSlot_ ^= 1U;
        return ioSlot;
    }

    __aicore__ inline int32_t IoEventId(uint32_t ioSlot) const
    {
        return static_cast<int32_t>(ioSlot);
    }

    __aicore__ inline void WaitIoInFree(uint32_t ioSlot) const
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(IoEventId(ioSlot));
    }

    __aicore__ inline void ReleaseIoIn(uint32_t ioSlot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(IoEventId(ioSlot));
    }

    __aicore__ inline void WaitIoOutFree(uint32_t ioSlot) const
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(IoEventId(ioSlot));
    }

    __aicore__ inline void ReleaseIoOut(uint32_t ioSlot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(IoEventId(ioSlot));
    }

    __aicore__ inline void SyncMte2ToV(uint32_t ioSlot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoEventId(ioSlot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoEventId(ioSlot));
    }

    __aicore__ inline void SyncVToMte3(uint32_t ioSlot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoEventId(ioSlot));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoEventId(ioSlot));
    }

    __aicore__ inline uint64_t GateOffset(uint64_t batch, uint64_t head, uint64_t token) const
    {
        return (batch * static_cast<uint64_t>(tiling_->HV) + head) * static_cast<uint64_t>(tiling_->T) + token;
    }

    __aicore__ inline uint64_t KeyOffset(uint64_t batch, uint64_t head, uint64_t token, uint64_t col) const
    {
        return (((batch * static_cast<uint64_t>(tiling_->HK) + head) * static_cast<uint64_t>(tiling_->T) + token) *
                static_cast<uint64_t>(tiling_->K)) + col;
    }

    __aicore__ inline uint64_t ValueOffset(uint64_t batch, uint64_t head, uint64_t token, uint64_t col) const
    {
        return (((batch * static_cast<uint64_t>(tiling_->HV) + head) * static_cast<uint64_t>(tiling_->T) + token) *
                static_cast<uint64_t>(tiling_->V)) + col;
    }

    __aicore__ inline uint64_t ValueKeyLikeOffset(uint64_t batch, uint64_t head, uint64_t token, uint64_t col) const
    {
        return (((batch * static_cast<uint64_t>(tiling_->HV) + head) * static_cast<uint64_t>(tiling_->T) + token) *
                static_cast<uint64_t>(tiling_->K)) + col;
    }

    __aicore__ inline uint64_t MatrixOffset(uint64_t batch, uint64_t head, uint64_t token, uint64_t localCol) const
    {
        return (((batch * static_cast<uint64_t>(tiling_->HV) + head) * static_cast<uint64_t>(tiling_->T) + token) *
                static_cast<uint64_t>(tiling_->chunkSize)) + localCol;
    }

    __aicore__ inline uint64_t WorkspaceElem(int64_t byteOffset, uint64_t elemOffset) const
    {
        return static_cast<uint64_t>(byteOffset) / sizeof(T) + elemOffset;
    }

    __aicore__ inline uint64_t SlotBaseElem(uint64_t slot) const
    {
        return ((coreIdx_ * static_cast<uint64_t>(tiling_->workspaceBufferCount) + slot) *
                static_cast<uint64_t>(tiling_->workspaceSlotBytes)) / sizeof(T);
    }

    __aicore__ inline uint64_t SlotWorkspaceElem(uint64_t slot, int64_t byteOffset, uint64_t elemOffset) const
    {
        return SlotBaseElem(slot) + WorkspaceElem(byteOffset, elemOffset);
    }

    __aicore__ inline uint64_t SlotValueKeyLikeOffset(uint64_t head, uint64_t localRow, uint64_t col) const
    {
        return ((head * static_cast<uint64_t>(tiling_->chunkSize) + localRow) *
                static_cast<uint64_t>(tiling_->K)) + col;
    }

    __aicore__ inline uint64_t SlotValueOffset(uint64_t head, uint64_t localRow, uint64_t col) const
    {
        return ((head * static_cast<uint64_t>(tiling_->chunkSize) + localRow) *
                static_cast<uint64_t>(tiling_->V)) + col;
    }

    __aicore__ inline uint64_t SlotMatrixOffset(uint64_t head, uint64_t localRow, uint64_t localCol) const
    {
        return ((head * static_cast<uint64_t>(tiling_->chunkSize) + localRow) *
                static_cast<uint64_t>(tiling_->chunkSize)) + localCol;
    }

    __aicore__ inline void ResolveChunk(uint64_t chunkLinear, uint64_t &batch, uint64_t &chunkBegin,
                                        uint64_t &chunkEnd)
    {
        const uint64_t chunkSize = static_cast<uint64_t>(tiling_->chunkSize);
        if constexpr (IS_VARLEN) {
            uint64_t seqIdx = static_cast<uint64_t>(chunkIndicesGm_.GetValue(chunkLinear * 2));
            uint64_t chunkInSeq = static_cast<uint64_t>(chunkIndicesGm_.GetValue(chunkLinear * 2 + 1));
            uint64_t seqBegin = static_cast<uint64_t>(cuSeqlensGm_.GetValue(seqIdx));
            uint64_t seqEnd = static_cast<uint64_t>(cuSeqlensGm_.GetValue(seqIdx + 1));
            batch = 0;
            chunkBegin = seqBegin + chunkInSeq * chunkSize;
            chunkEnd = Min(chunkBegin + chunkSize, seqEnd);
            return;
        }

        const uint64_t chunksPerBatch = (static_cast<uint64_t>(tiling_->T) + chunkSize - 1) / chunkSize;
        batch = chunkLinear / chunksPerBatch;
        uint64_t chunkInBatch = chunkLinear - batch * chunksPerBatch;
        chunkBegin = chunkInBatch * chunkSize;
        chunkEnd = Min(chunkBegin + chunkSize, static_cast<uint64_t>(tiling_->T));
    }

    __aicore__ inline void InitPipelineBuffers()
    {
        ioInOffset_ = static_cast<uint64_t>(tiling_->vectorIoInOffset);
        ioOutOffset_ = static_cast<uint64_t>(tiling_->vectorIoOutOffset);
        persistentOffset_ = static_cast<uint64_t>(tiling_->vectorPersistentOffset);
        gateCacheOffset_ = static_cast<uint64_t>(tiling_->vectorGateCacheOffset);
        tempOffset_ = static_cast<uint64_t>(tiling_->vectorTempOffset);
        maskOffset_ = static_cast<uint64_t>(tiling_->vectorMaskOffset);
        zeroOffset_ = static_cast<uint64_t>(tiling_->vectorZeroOffset);
        ubBytes_ = static_cast<uint64_t>(tiling_->vectorUbBytes);
        ioInStride_ = (ioOutOffset_ - ioInOffset_) / BWD_VEC_IO_BUFFER_COUNT;
        ioOutStride_ = (persistentOffset_ - ioOutOffset_) / BWD_VEC_IO_BUFFER_COUNT;
        ioInSlot_ = 0;
        ioOutSlot_ = 0;
        pipe_->InitBuffer(ubBuf_, ubBytes_);
        for (uint32_t ioSlot = 0; ioSlot < BWD_VEC_IO_BUFFER_COUNT; ++ioSlot) {
            ReleaseIoIn(ioSlot);
            ReleaseIoOut(ioSlot);
        }
    }

    __aicore__ inline void DrainPipelineBuffers() const
    {
        for (uint32_t ioSlot = 0; ioSlot < BWD_VEC_IO_BUFFER_COUNT; ++ioSlot) {
            WaitIoInFree(ioSlot);
            WaitIoOutFree(ioSlot);
        }
    }

    __aicore__ inline void InitTriangleMasks()
    {
        uint32_t maskChunkSize = static_cast<uint32_t>(tiling_->chunkSize);
        uint32_t maskBlocksPerRow = MaskBlocksPerRow();
        auto lowerMask = LowerMaskTensor();
        auto upperMask = UpperMaskTensor();
        for (uint32_t row = 0; row < maskChunkSize; ++row) {
            for (uint32_t block = 0; block < maskBlocksPerRow; ++block) {
                uint32_t colStart = block * BWD_VEC_BIT_NUM_FOR_UINT8;
                uint8_t lowerMaskVal = 0;
                uint8_t upperMaskVal = 0;
                for (uint32_t bit = 0; bit < BWD_VEC_BIT_NUM_FOR_UINT8 && colStart + bit < maskChunkSize; ++bit) {
                    uint32_t col = colStart + bit;
                    if (col >= row) {
                        lowerMaskVal |= static_cast<uint8_t>(1U << bit);
                    }
                    if (col <= row) {
                        upperMaskVal |= static_cast<uint8_t>(1U << bit);
                    }
                }
                uint32_t maskOffset = row * maskBlocksPerRow + block;
                lowerMask.SetValue(maskOffset, lowerMaskVal);
                upperMask.SetValue(maskOffset, upperMaskVal);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SignalAivFinishStore()
    {
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
    }

    __aicore__ inline void WaitAicFinishStore()
    {
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagAicFinishStore);
    }

    __aicore__ inline void WaitAicFinishFullStore()
    {
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagAicFinishStore);
    }

    __aicore__ inline uint64_t GateCacheStride() const
    {
        return CeilDiv(tiling_->chunkSize * sizeof(float32_t), BWD_VEC_ONE_BLOCK_32) *
               BWD_VEC_ONE_BLOCK_32 / sizeof(float32_t);
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> SlotBetaCache(uint64_t slot)
    {
        auto cache = GateCacheTensor();
        uint64_t stride = GateCacheStride();
        return cache[slot * 2U * stride];
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> SlotGCache(uint64_t slot)
    {
        auto cache = GateCacheTensor();
        uint64_t stride = GateCacheStride();
        return cache[slot * 2U * stride + stride];
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> SlotGExpCache(uint64_t slot)
    {
        return PersistentTensor()[slot * GateCacheStride()];
    }

    __aicore__ inline bool UseInputCache() const
    {
        return tiling_->chunkSize == 64 && tiling_->K == 128 && tiling_->V == 128;
    }

    __aicore__ inline uint64_t LocalOwnedRowOffset(uint32_t rowOffset) const
    {
        uint64_t rowTile = static_cast<uint64_t>(tiling_->rowTileDa);
        return (static_cast<uint64_t>(rowOffset) / (rowTile * subBlockNum_)) * rowTile;
    }

    __aicore__ inline AscendC::LocalTensor<T> SlotFinalDaCache(uint64_t slot)
    {
        uint64_t gatePersistentBytes = 4U * GateCacheStride() * sizeof(float32_t);
        uint64_t subBlockMatrixBytes = static_cast<uint64_t>(tiling_->chunkSize) *
                                       static_cast<uint64_t>(tiling_->chunkSize) * sizeof(T) /
                                       subBlockNum_;
        return UbTensor<T>(persistentOffset_ + gatePersistentBytes + slot * subBlockMatrixBytes);
    }

    __aicore__ inline uint64_t FinalDaCacheElemOffset(uint32_t rowOffset) const
    {
        return LocalOwnedRowOffset(rowOffset) * static_cast<uint64_t>(tiling_->chunkSize);
    }

    __aicore__ inline uint64_t InputCacheElemOffset(uint32_t rowOffset, uint32_t width) const
    {
        return LocalOwnedRowOffset(rowOffset) * static_cast<uint64_t>(width);
    }

    __aicore__ inline uint64_t InputCacheBaseByteOffset() const
    {
        uint64_t gatePersistentBytes = 4U * GateCacheStride() * sizeof(float32_t);
        uint64_t finalDaCacheBytes = static_cast<uint64_t>(tiling_->chunkSize) *
                                     static_cast<uint64_t>(tiling_->chunkSize) * sizeof(T);
        return persistentOffset_ + gatePersistentBytes + finalDaCacheBytes;
    }

    __aicore__ inline uint64_t InputCacheRowsPerSlot() const
    {
        return static_cast<uint64_t>(CeilDiv(tiling_->chunkSize, subBlockNum_));
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> SlotKCache(uint64_t slot)
    {
        uint64_t rows = InputCacheRowsPerSlot();
        uint64_t slotStride = rows * static_cast<uint64_t>(tiling_->K + tiling_->V);
        return UbTensor<float32_t>(InputCacheBaseByteOffset() + slot * slotStride * sizeof(float32_t));
    }

    __aicore__ inline AscendC::LocalTensor<float32_t> SlotVCache(uint64_t slot)
    {
        uint64_t rows = InputCacheRowsPerSlot();
        return SlotKCache(slot)[rows * static_cast<uint64_t>(tiling_->K)];
    }

    __aicore__ inline void LoadGateToFp32(AscendC::LocalTensor<float32_t> dst, uint64_t offset, uint32_t count)
    {
        uint32_t ioSlot = NextIoInSlot();
        auto gate = IoInTensor<GateT>(ioSlot);
        WaitIoInFree(ioSlot);
        AscendC::DataCopyPad(gate, betaGm_[offset], {1, count * static_cast<uint32_t>(sizeof(GateT)), 0, 0, 0},
                             {false, 0, 0, 0});
        SyncMte2ToV(ioSlot);
        if constexpr (std::is_same<GateT, float32_t>::value) {
            AscendC::Adds(dst, gate, 0.0f, count);
        } else {
            AscendC::Cast(dst, gate, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReleaseIoIn(ioSlot);
    }

    __aicore__ inline void LoadGToFp32(AscendC::LocalTensor<float32_t> dst, uint64_t offset, uint32_t count)
    {
        uint32_t ioSlot = NextIoInSlot();
        auto gate = IoInTensor<GateT>(ioSlot);
        WaitIoInFree(ioSlot);
        AscendC::DataCopyPad(gate, gGm_[offset], {1, count * static_cast<uint32_t>(sizeof(GateT)), 0, 0, 0},
                             {false, 0, 0, 0});
        SyncMte2ToV(ioSlot);
        if constexpr (std::is_same<GateT, float32_t>::value) {
            AscendC::Adds(dst, gate, 0.0f, count);
        } else {
            AscendC::Cast(dst, gate, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReleaseIoIn(ioSlot);
    }

    __aicore__ inline void ReduceRowsToVector(AscendC::LocalTensor<float32_t> matrix,
                                              AscendC::LocalTensor<float32_t> reduce,
                                              uint32_t width, uint32_t rows, uint32_t wholeReduceCnt)
    {
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::WholeReduceSum(reduce[row * BWD_VEC_FP32_PER_BLOCK_8], matrix[row * width],
                                    BWD_VEC_FP32_PER_REPEAT_64, wholeReduceCnt, 1, 1, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(matrix, reduce, wholeReduceCnt, rows, 1, 1, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadMatrixTile(AscendC::LocalTensor<float32_t> dst, uint64_t slot,
                                          int64_t baseOffset, uint64_t elemOffset, uint32_t count)
    {
        uint32_t ioSlot = NextIoInSlot();
        auto tensor = IoInTensor<T>(ioSlot);
        WaitIoInFree(ioSlot);
        AscendC::DataCopy(tensor, workspaceGm_[SlotWorkspaceElem(slot, baseOffset, elemOffset)], count);
        SyncMte2ToV(ioSlot);
        if constexpr (std::is_same<T, float32_t>::value) {
            AscendC::DataCopy(dst, tensor, count);
        } else {
            AscendC::Cast(dst, tensor, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReleaseIoIn(ioSlot);
    }

    __aicore__ inline uint32_t BeginLoadGmTile(AscendC::GlobalTensor<T> &src,
                                               uint64_t elemOffset, uint32_t count)
    {
        uint32_t ioSlot = NextIoInSlot();
        auto tensor = IoInTensor<T>(ioSlot);
        WaitIoInFree(ioSlot);
        AscendC::DataCopy(tensor, src[elemOffset], count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoEventId(ioSlot));
        return ioSlot;
    }

    __aicore__ inline uint32_t BeginLoadMatrixTile(uint64_t slot, int64_t baseOffset,
                                                   uint64_t elemOffset, uint32_t count)
    {
        uint32_t ioSlot = NextIoInSlot();
        auto tensor = IoInTensor<T>(ioSlot);
        WaitIoInFree(ioSlot);
        AscendC::DataCopy(tensor, workspaceGm_[SlotWorkspaceElem(slot, baseOffset, elemOffset)], count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoEventId(ioSlot));
        return ioSlot;
    }

    __aicore__ inline void FinishLoadTile(AscendC::LocalTensor<float32_t> dst,
                                          uint32_t ioSlot, uint32_t count)
    {
        auto tensor = IoInTensor<T>(ioSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoEventId(ioSlot));
        if constexpr (std::is_same<T, float32_t>::value) {
            AscendC::DataCopy(dst, tensor, count);
        } else {
            AscendC::Cast(dst, tensor, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReleaseIoIn(ioSlot);
    }

    __aicore__ inline void LoadGmTile(AscendC::LocalTensor<float32_t> dst, AscendC::GlobalTensor<T> &src,
                                      uint64_t elemOffset, uint32_t count)
    {
        uint32_t ioSlot = NextIoInSlot();
        auto tensor = IoInTensor<T>(ioSlot);
        WaitIoInFree(ioSlot);
        AscendC::DataCopy(tensor, src[elemOffset], count);
        SyncMte2ToV(ioSlot);
        if constexpr (std::is_same<T, float32_t>::value) {
            AscendC::DataCopy(dst, tensor, count);
        } else {
            AscendC::Cast(dst, tensor, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReleaseIoIn(ioSlot);
    }

    __aicore__ inline void StoreGmTile(AscendC::GlobalTensor<T> &dst, uint64_t elemOffset,
                                       AscendC::LocalTensor<float32_t> src, uint32_t count)
    {
        uint32_t ioSlot = NextIoOutSlot();
        auto tensor = IoOutTensor<T>(ioSlot);
        WaitIoOutFree(ioSlot);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<T, float32_t>::value) {
            AscendC::DataCopy(tensor, src, count);
        } else {
            AscendC::Cast(tensor, src, AscendC::RoundMode::CAST_RINT, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SyncVToMte3(ioSlot);
        AscendC::DataCopy(dst[elemOffset], tensor, count);
        ReleaseIoOut(ioSlot);
    }

    __aicore__ inline void StoreFinalDaTile(uint64_t slot, uint32_t rowOffset,
                                            AscendC::GlobalTensor<T> &dst, uint64_t elemOffset,
                                            AscendC::LocalTensor<float32_t> src, uint32_t count)
    {
        uint32_t ioSlot = NextIoOutSlot();
        auto cache = SlotFinalDaCache(slot)[FinalDaCacheElemOffset(rowOffset)];
        WaitIoOutFree(ioSlot);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<T, float32_t>::value) {
            AscendC::DataCopy(cache, src, count);
        } else {
            AscendC::Cast(cache, src, AscendC::RoundMode::CAST_RINT, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SyncVToMte3(ioSlot);
        AscendC::DataCopy(dst[elemOffset], cache, count);
        ReleaseIoOut(ioSlot);
    }

    __aicore__ inline void LoadFinalDaCache(AscendC::LocalTensor<float32_t> dst, uint64_t slot,
                                            uint32_t rowOffset, uint32_t count)
    {
        auto cache = SlotFinalDaCache(slot)[FinalDaCacheElemOffset(rowOffset)];
        if constexpr (std::is_same<T, float32_t>::value) {
            AscendC::DataCopy(dst, cache, count);
        } else {
            AscendC::Cast(dst, cache, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreGateTile(AscendC::GlobalTensor<GateT> &dst, uint64_t elemOffset,
                                         AscendC::LocalTensor<float32_t> src, uint32_t count)
    {
        uint32_t ioSlot = NextIoOutSlot();
        auto out = IoOutTensor<GateT>(ioSlot);
        WaitIoOutFree(ioSlot);
        if constexpr (std::is_same<GateT, float32_t>::value) {
            AscendC::Adds(out, src, 0.0f, count);
        } else {
            AscendC::Cast(out, src, AscendC::RoundMode::CAST_RINT, count);
        }
        SyncVToMte3(ioSlot);
        AscendC::DataCopyPad(dst[elemOffset], out, {1, count * static_cast<uint32_t>(sizeof(GateT)), 0, 0, 0});
        ReleaseIoOut(ioSlot);
    }

    const PrepareWyReprBwdTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    uint64_t coreIdx_ = 0;
    uint64_t coreNum_ = 1;
    uint64_t subBlockIdx_ = 0;
    uint64_t subBlockNum_ = 1;

    AscendC::GlobalTensor<T> kGm_;
    AscendC::GlobalTensor<T> vGm_;
    AscendC::GlobalTensor<T> aGm_;
    AscendC::GlobalTensor<T> dwGm_;
    AscendC::GlobalTensor<T> duGm_;
    AscendC::GlobalTensor<GateT> betaGm_;
    AscendC::GlobalTensor<GateT> gGm_;
    AscendC::GlobalTensor<T> dkGm_;
    AscendC::GlobalTensor<T> dvGm_;
    AscendC::GlobalTensor<GateT> dbetaGm_;
    AscendC::GlobalTensor<GateT> dgGm_;
    AscendC::GlobalTensor<T> workspaceGm_;
    AscendC::GlobalTensor<uint64_t> cuSeqlensGm_;
    AscendC::GlobalTensor<uint64_t> chunkIndicesGm_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    Catlass::Arch::CrossCoreFlagWithReverse<14> flagAicFinishStore{SYNC_FLAG_2, SYNC_FLAG_3};
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_FLAG_4, SYNC_FLAG_5};
    uint64_t ioInOffset_ = 0;
    uint64_t ioOutOffset_ = 0;
    uint64_t ioInStride_ = 0;
    uint64_t ioOutStride_ = 0;
    uint64_t persistentOffset_ = 0;
    uint64_t gateCacheOffset_ = 0;
    uint64_t tempOffset_ = 0;
    uint64_t maskOffset_ = 0;
    uint64_t zeroOffset_ = 0;
    uint64_t ubBytes_ = 0;
    uint32_t ioInSlot_ = 0;
    uint32_t ioOutSlot_ = 0;
};

} // namespace GDN

#endif // PREPARE_WY_REPR_BWD_VECTOR_H

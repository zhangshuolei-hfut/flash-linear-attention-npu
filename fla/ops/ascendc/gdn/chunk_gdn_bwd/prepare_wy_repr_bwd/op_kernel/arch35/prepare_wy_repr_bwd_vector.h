/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_ARCH35_VECTOR_H
#define PREPARE_WY_REPR_BWD_ARCH35_VECTOR_H

#include <type_traits>

#include "../prepare_wy_repr_bwd_struct.h"
#include "prepare_wy_repr_bwd_cube_resource.h"
#include "kernel_operator.h"

namespace GDN {

static constexpr uint32_t BWD_A5_SLOT_COUNT = 2;
static constexpr uint32_t BWD_A5_ONE_BLOCK_BYTES = 32;
static constexpr uint32_t BWD_A5_FP32_PER_BLOCK = 8;
static constexpr uint32_t BWD_A5_FP32_PER_REPEAT = 64;

template <typename T, typename GateT, bool IS_VARLEN, int V, int CHUNK_SIZE>
class PrepareWyReprBwdVector {
    struct TaskDesc {
        bool valid = false;
        uint64_t ownerUnit = 0;
        uint64_t chunkLinear = 0;
        uint64_t batch = 0;
        uint64_t chunkBegin = 0;
        uint64_t chunkEnd = 0;
        uint64_t keyHead = 0;
        uint64_t valueHead = 0;
        uint64_t hvLocal = 0;
        bool firstValueHead = true;
    };

public:
    /**
     * @brief 初始化 A5 vector kernel 的 GM tensor、tiling、core 和 subblock 信息。
     */
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dw, GM_ADDR du, GM_ADDR g,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta,
                                GM_ADDR dg, GM_ADDR workspace, const PrepareWyReprBwdTilingData *tiling)
    {
        (void)A;
        (void)dw;
        (void)du;
        (void)workspace;
        tiling_ = tiling;
        coreIdx_ = static_cast<uint64_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
        coreNum_ = static_cast<uint64_t>(tiling_->usedCoreNum) > 0 ? static_cast<uint64_t>(tiling_->usedCoreNum) : 1;
        subBlockIdx_ = static_cast<uint64_t>(AscendC::GetSubBlockIdx());
        subBlockNum_ = static_cast<uint64_t>(AscendC::GetSubBlockNum());

        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(beta));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dk));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dv));
        dbetaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(dbeta));
        dgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(dg));
        if (cuSeqlens != nullptr) {
            cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(cuSeqlens));
        }
        if (chunkIndices != nullptr) {
            chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(chunkIndices));
        }
    }

    /**
     * @brief 执行 A5 vector 侧双任务流水。
     *
     * 非 GVA 使用两个独立 owner unit 填满 task slot；GVA 使用同一 owner 下两个 hv。
     * 当前 CUBE 侧直接返回，本函数独立跑通 MTE2/VEC/MTE3 双 buffer profile 路径。
     */
    __aicore__ inline void ProcessPipeline()
    {
        InitUbArena();
        if (tiling_->taskPairMode == BWD_TASK_PAIR_INDEPENDENT_UNIT) {
            ProcessIndependentUnitPairs();
        } else {
            ProcessSameOwnerHeadPairs();
        }
        DrainIoBuffers();
    }

private:
    /**
     * @brief 初始化 UB arena 和输入/输出 buffer 的可复用事件状态。
     */
    __aicore__ inline void InitUbArena()
    {
        ioInOffset_ = static_cast<uint64_t>(tiling_->vectorIoInOffset);
        ioOutOffset_ = static_cast<uint64_t>(tiling_->vectorIoOutOffset);
        persistentOffset_ = static_cast<uint64_t>(tiling_->vectorPersistentOffset);
        gateCacheOffset_ = static_cast<uint64_t>(tiling_->vectorGateCacheOffset);
        tempOffset_ = static_cast<uint64_t>(tiling_->vectorTempOffset);
        ubBytes_ = static_cast<uint64_t>(tiling_->vectorUbBytes);
        ioInStride_ = (static_cast<uint64_t>(tiling_->vectorIoOutOffset) - ioInOffset_) / BWD_A5_SLOT_COUNT;
        ioOutStride_ = (static_cast<uint64_t>(tiling_->vectorPersistentOffset) -
                        static_cast<uint64_t>(tiling_->vectorIoOutOffset)) / BWD_A5_SLOT_COUNT;
        for (uint32_t slot = 0; slot < BWD_A5_SLOT_COUNT; ++slot) {
            ReleaseInputSlot(slot);
            ReleaseOutputSlot(slot);
        }
    }

    /**
     * @brief 非 GVA 调度：两个 task slot 分别消费两个独立 owner unit。
     */
    __aicore__ inline void ProcessIndependentUnitPairs()
    {
        const uint64_t ownerUnitNum = OwnerUnitNum();
        const uint64_t step = coreNum_ * BWD_A5_SLOT_COUNT;
        for (uint64_t baseUnit = coreIdx_; baseUnit < ownerUnitNum; baseUnit += step) {
            TaskDesc task0 = BuildIndependentTask(baseUnit);
            TaskDesc task1 = BuildIndependentTask(baseUnit + coreNum_);
            ProcessTaskPair(task0, task1);
        }
    }

    /**
     * @brief GVA 调度：一个 owner unit 内每轮处理两个相邻 value head。
     */
    __aicore__ inline void ProcessSameOwnerHeadPairs()
    {
        const uint64_t ownerUnitNum = OwnerUnitNum();
        const uint64_t headGroup = static_cast<uint64_t>(tiling_->headGroup);
        for (uint64_t ownerUnit = coreIdx_; ownerUnit < ownerUnitNum; ownerUnit += coreNum_) {
            for (uint64_t hvLocal = 0; hvLocal < headGroup; hvLocal += BWD_A5_SLOT_COUNT) {
                TaskDesc task0 = BuildGroupedHeadTask(ownerUnit, hvLocal);
                TaskDesc task1 = BuildGroupedHeadTask(ownerUnit, hvLocal + 1);
                ProcessTaskPair(task0, task1);
            }
        }
    }

    /**
     * @brief 执行一个双 slot 逻辑循环，先缓存两个 task 的 gate，再分别写回公开输出。
     */
    __aicore__ inline void ProcessTaskPair(const TaskDesc &task0, const TaskDesc &task1)
    {
        LoadTaskGate(0, task0);
        LoadTaskGate(1, task1);
        ProcessTaskOutputs(0, task0);
        ProcessTaskOutputs(1, task1);
    }

    /**
     * @brief 从独立 owner unit 解码非 GVA task。
     */
    __aicore__ inline TaskDesc BuildIndependentTask(uint64_t ownerUnit)
    {
        TaskDesc task;
        if (ownerUnit >= OwnerUnitNum()) {
            return task;
        }
        DecodeOwnerUnit(ownerUnit, task);
        task.valueHead = task.keyHead;
        task.hvLocal = 0;
        task.firstValueHead = true;
        task.valid = task.chunkEnd > task.chunkBegin;
        return task;
    }

    /**
     * @brief 从 owner unit 和局部 hv 序号解码 GVA task。
     */
    __aicore__ inline TaskDesc BuildGroupedHeadTask(uint64_t ownerUnit, uint64_t hvLocal)
    {
        TaskDesc task;
        const uint64_t headGroup = static_cast<uint64_t>(tiling_->headGroup);
        if (ownerUnit >= OwnerUnitNum() || hvLocal >= headGroup) {
            return task;
        }
        DecodeOwnerUnit(ownerUnit, task);
        task.hvLocal = hvLocal;
        task.valueHead = task.keyHead * headGroup + hvLocal;
        task.firstValueHead = hvLocal == 0;
        task.valid = task.valueHead < static_cast<uint64_t>(tiling_->HV) && task.chunkEnd > task.chunkBegin;
        return task;
    }

    /**
     * @brief 将 owner unit 解码成 chunk、batch、token 范围和 key head。
     */
    __aicore__ inline void DecodeOwnerUnit(uint64_t ownerUnit, TaskDesc &task)
    {
        const uint64_t hk = static_cast<uint64_t>(tiling_->HK);
        task.ownerUnit = ownerUnit;
        task.chunkLinear = ownerUnit / hk;
        task.keyHead = ownerUnit - task.chunkLinear * hk;
        ResolveChunk(task.chunkLinear, task.batch, task.chunkBegin, task.chunkEnd);
    }

    /**
     * @brief 解析固定长或 varlen 模式下的 chunk token 范围。
     */
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

        const uint64_t chunksPerBatch = CeilDiv(static_cast<uint64_t>(tiling_->T), chunkSize);
        batch = chunkLinear / chunksPerBatch;
        uint64_t chunkInBatch = chunkLinear - batch * chunksPerBatch;
        chunkBegin = chunkInBatch * chunkSize;
        chunkEnd = Min(chunkBegin + chunkSize, static_cast<uint64_t>(tiling_->T));
    }

    /**
     * @brief 将一个 task 的 beta、g、exp(g) 缓存在 UB 中。
     */
    __aicore__ inline void LoadTaskGate(uint32_t slot, const TaskDesc &task)
    {
        if (!task.valid) {
            return;
        }
        const uint32_t count = static_cast<uint32_t>(task.chunkEnd - task.chunkBegin);
        auto beta = SlotBetaCache(slot);
        auto gate = SlotGCache(slot);
        auto gateExp = SlotGExpCache(slot);
        const uint32_t calcCount = AlignedFp32Count(count);
        LoadGateVector(beta, betaGm_, GateOffset(task.batch, task.valueHead, task.chunkBegin), count);
        LoadGateVector(gate, gGm_, GateOffset(task.batch, task.valueHead, task.chunkBegin), count);
        AscendC::Adds(gateExp, gate, 0.0f, calcCount);
        AscendC::PipeBarrier<PIPE_V>();
        ExpGateVector(gateExp, calcCount);
    }

    /**
     * @brief 处理一个 task 的行 tile 输出，覆盖 dk、dv、dbeta、dg 四个公开张量。
     */
    __aicore__ inline void ProcessTaskOutputs(uint32_t slot, const TaskDesc &task)
    {
        if (!task.valid) {
            return;
        }
        const uint32_t chunkLen = static_cast<uint32_t>(task.chunkEnd - task.chunkBegin);
        const uint32_t rowTile = VectorRowTile();
        const uint32_t rowStep = rowTile * static_cast<uint32_t>(subBlockNum_);
        if (rowStep == 0) {
            return;
        }
        for (uint32_t rowOffset = static_cast<uint32_t>(subBlockIdx_) * rowTile;
             rowOffset < chunkLen; rowOffset += rowStep) {
            uint32_t rows = rowOffset + rowTile > chunkLen ? chunkLen - rowOffset : rowTile;
            uint64_t token = task.chunkBegin + rowOffset;
            ProcessKeyRows(slot, task, rowOffset, token, rows);
            ProcessValueRows(slot, task, rowOffset, token, rows);
            ProcessGateRows(slot, task, rowOffset, token, rows);
        }
    }

    /**
     * @brief 按 64 元素粒度计算 gate exp，避免 chunk=128 时单条矢量指令跨度过大。
     */
    __aicore__ inline void ExpGateVector(AscendC::LocalTensor<float32_t> gateExp, uint32_t count)
    {
        for (uint32_t offset = 0; offset < count; offset += BWD_A5_FP32_PER_REPEAT) {
            uint32_t curCount = offset + BWD_A5_FP32_PER_REPEAT > count ?
                                count - offset : BWD_A5_FP32_PER_REPEAT;
            AscendC::Exp(gateExp[offset], gateExp[offset], curCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 用 beta 缩放 k 行并写入 dk，用于验证 K 方向 MTE2/VEC/MTE3 流水。
     */
    __aicore__ inline void ProcessKeyRows(uint32_t slot, const TaskDesc &task, uint32_t rowOffset,
                                          uint64_t token, uint32_t rows)
    {
        const uint32_t width = static_cast<uint32_t>(tiling_->K);
        auto inFp32 = TempTensor();
        auto outFp32 = inFp32[static_cast<uint64_t>(rows) * width];
        auto betaBrcb = outFp32[static_cast<uint64_t>(rows) * width];
        uint32_t ioSlot = BeginLoadTensor(kGm_, KeyOffset(task.batch, task.keyHead, token, 0), rows * width);
        FinishLoadTensor(inFp32, ioSlot, rows * width);
        BroadcastGate(betaBrcb, SlotBetaCache(slot)[rowOffset], rows);
        ScaleRows(outFp32, inFp32, betaBrcb, rows, width);
        StoreTensor(dkGm_, KeyOffset(task.batch, task.keyHead, token, 0), outFp32, rows * width);
    }

    /**
     * @brief 用 beta 缩放 v 行并写入 dv，用于验证 V 方向 MTE2/VEC/MTE3 流水。
     */
    __aicore__ inline void ProcessValueRows(uint32_t slot, const TaskDesc &task, uint32_t rowOffset,
                                            uint64_t token, uint32_t rows)
    {
        const uint32_t width = static_cast<uint32_t>(tiling_->V);
        auto inFp32 = TempTensor();
        auto outFp32 = inFp32[static_cast<uint64_t>(rows) * width];
        auto betaBrcb = outFp32[static_cast<uint64_t>(rows) * width];
        uint32_t ioSlot = BeginLoadTensor(vGm_, ValueOffset(task.batch, task.valueHead, token, 0), rows * width);
        FinishLoadTensor(inFp32, ioSlot, rows * width);
        BroadcastGate(betaBrcb, SlotBetaCache(slot)[rowOffset], rows);
        ScaleRows(outFp32, inFp32, betaBrcb, rows, width);
        StoreTensor(dvGm_, ValueOffset(task.batch, task.valueHead, token, 0), outFp32, rows * width);
    }

    /**
     * @brief 生成 dbeta、dg 的轻量 gate 输出，保证 gate dtype 的 MTE3 路径也被覆盖。
     */
    __aicore__ inline void ProcessGateRows(uint32_t slot, const TaskDesc &task, uint32_t rowOffset,
                                           uint64_t token, uint32_t rows)
    {
        auto tmp = TempTensor();
        const uint32_t calcRows = AlignedFp32Count(rows);
        AscendC::Add(tmp, SlotBetaCache(slot)[rowOffset], SlotGCache(slot)[rowOffset], calcRows);
        AscendC::PipeBarrier<PIPE_V>();
        StoreGateVector(dbetaGm_, GateOffset(task.batch, task.valueHead, token), tmp, rows);
        StoreGateVector(dgGm_, GateOffset(task.batch, task.valueHead, token), SlotGExpCache(slot)[rowOffset], rows);
    }

    /**
     * @brief 从 GM 读取 gate 向量并转换为 fp32。
     */
    __aicore__ inline void LoadGateVector(AscendC::LocalTensor<float32_t> dst,
                                          AscendC::GlobalTensor<GateT> &src,
                                          uint64_t elemOffset, uint32_t count)
    {
        uint32_t ioSlot = NextInputSlot();
        auto input = IoInTensor<GateT>(ioSlot);
        WaitInputSlot(ioSlot);
        AscendC::DataCopyPad(input, src[elemOffset], {1, count * static_cast<uint32_t>(sizeof(GateT)), 0, 0, 0},
                             {false, 0, 0, 0});
        SyncMte2ToV(ioSlot);
        const uint32_t calcCount = AlignedFp32Count(count);
        if constexpr (std::is_same<GateT, float32_t>::value) {
            AscendC::Adds(dst, input, 0.0f, calcCount);
        } else {
            AscendC::Cast(dst, input, AscendC::RoundMode::CAST_NONE, calcCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReleaseInputSlot(ioSlot);
    }

    /**
     * @brief 异步开始读取一个 T 类型 GM tile。
     */
    __aicore__ inline uint32_t BeginLoadTensor(AscendC::GlobalTensor<T> &src, uint64_t elemOffset, uint32_t count)
    {
        uint32_t ioSlot = NextInputSlot();
        auto input = IoInTensor<T>(ioSlot);
        WaitInputSlot(ioSlot);
        AscendC::DataCopy(input, src[elemOffset], count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(InputEventId(ioSlot));
        return ioSlot;
    }

    /**
     * @brief 等待 GM tile 到达 UB 并转换为 fp32。
     */
    __aicore__ inline void FinishLoadTensor(AscendC::LocalTensor<float32_t> dst, uint32_t ioSlot, uint32_t count)
    {
        auto input = IoInTensor<T>(ioSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(InputEventId(ioSlot));
        if constexpr (std::is_same<T, float32_t>::value) {
            AscendC::DataCopy(dst, input, count);
        } else {
            AscendC::Cast(dst, input, AscendC::RoundMode::CAST_NONE, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReleaseInputSlot(ioSlot);
    }

    /**
     * @brief 将 fp32 tile 转回 T 并异步写回 GM。
     */
    __aicore__ inline void StoreTensor(AscendC::GlobalTensor<T> &dst, uint64_t elemOffset,
                                       AscendC::LocalTensor<float32_t> src, uint32_t count)
    {
        uint32_t ioSlot = NextOutputSlot();
        auto output = IoOutTensor<T>(ioSlot);
        WaitOutputSlot(ioSlot);
        if constexpr (std::is_same<T, float32_t>::value) {
            AscendC::DataCopy(output, src, count);
        } else {
            AscendC::Cast(output, src, AscendC::RoundMode::CAST_RINT, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SyncVToMte3(ioSlot);
        AscendC::DataCopy(dst[elemOffset], output, count);
        ReleaseOutputSlot(ioSlot);
    }

    /**
     * @brief 将 fp32 gate 向量转回 GateT 并写回 GM。
     */
    __aicore__ inline void StoreGateVector(AscendC::GlobalTensor<GateT> &dst, uint64_t elemOffset,
                                           AscendC::LocalTensor<float32_t> src, uint32_t count)
    {
        uint32_t ioSlot = NextOutputSlot();
        auto output = IoOutTensor<GateT>(ioSlot);
        WaitOutputSlot(ioSlot);
        if constexpr (std::is_same<GateT, float32_t>::value) {
            AscendC::Adds(output, src, 0.0f, count);
        } else {
            AscendC::Cast(output, src, AscendC::RoundMode::CAST_RINT, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SyncVToMte3(ioSlot);
        AscendC::DataCopyPad(dst[elemOffset], output, {1, count * static_cast<uint32_t>(sizeof(GateT)), 0, 0, 0});
        ReleaseOutputSlot(ioSlot);
    }

    /**
     * @brief 将每行一个 gate 标量广播成 BRCB 布局。
     */
    __aicore__ inline void BroadcastGate(AscendC::LocalTensor<float32_t> dst,
                                         AscendC::LocalTensor<float32_t> src, uint32_t rows)
    {
        AscendC::Brcb(dst, src, static_cast<uint8_t>(CeilDiv(rows, BWD_A5_FP32_PER_BLOCK)), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 按行使用 BRCB gate 缩放一个二维 tile。
     */
    __aicore__ inline void ScaleRows(AscendC::LocalTensor<float32_t> dst, AscendC::LocalTensor<float32_t> src,
                                     AscendC::LocalTensor<float32_t> gateBrcb, uint32_t rows, uint32_t width)
    {
        const uint8_t repeatStride = static_cast<uint8_t>(width * sizeof(float32_t) / BWD_A5_ONE_BLOCK_BYTES);
        for (uint32_t col = 0; col < width; col += BWD_A5_FP32_PER_REPEAT) {
            AscendC::Mul(dst[col], src[col], gateBrcb, BWD_A5_FP32_PER_REPEAT, rows,
                         {1, 1, 0, repeatStride, repeatStride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * @brief 返回可执行 owner unit 总数，tiling 旧缓存为 0 时从 chunkNum/HK 推导。
     */
    __aicore__ inline uint64_t OwnerUnitNum() const
    {
        uint64_t ownerUnitNum = static_cast<uint64_t>(tiling_->ownerUnitNum);
        if (ownerUnitNum == 0) {
            ownerUnitNum = static_cast<uint64_t>(tiling_->chunkNum) * static_cast<uint64_t>(tiling_->HK);
        }
        return ownerUnitNum;
    }

    /**
     * @brief 返回 A5 vector 行 tile，tiling 旧缓存为 0 时退回通用 rowTileInput。
     */
    __aicore__ inline uint32_t VectorRowTile() const
    {
        uint64_t rowTile = static_cast<uint64_t>(tiling_->a5VectorRowTile);
        if (rowTile == 0) {
            rowTile = static_cast<uint64_t>(tiling_->rowTileInput);
        }
        if (rowTile == 0) {
            rowTile = 8;
        }
        return static_cast<uint32_t>(rowTile);
    }

    /**
     * @brief 等待所有输入和输出 buffer 的异步搬运完成。
     */
    __aicore__ inline void DrainIoBuffers() const
    {
        for (uint32_t slot = 0; slot < BWD_A5_SLOT_COUNT; ++slot) {
            WaitInputSlot(slot);
            WaitOutputSlot(slot);
        }
    }

    /**
     * @brief 获取任意 UB byte offset 对应的 LocalTensor。
     */
    template <typename D>
    __aicore__ inline AscendC::LocalTensor<D> UbTensor(uint64_t byteOffset)
    {
        return ubBuf_.template GetBufferByByte<D>(static_cast<uint32_t>(byteOffset));
    }

    /**
     * @brief 获取指定输入 buffer slot 的 LocalTensor。
     */
    template <typename D>
    __aicore__ inline AscendC::LocalTensor<D> IoInTensor(uint32_t slot)
    {
        return UbTensor<D>(ioInOffset_ + static_cast<uint64_t>(slot) * ioInStride_);
    }

    /**
     * @brief 获取指定输出 buffer slot 的 LocalTensor。
     */
    template <typename D>
    __aicore__ inline AscendC::LocalTensor<D> IoOutTensor(uint32_t slot)
    {
        return UbTensor<D>(ioOutOffset_ + static_cast<uint64_t>(slot) * ioOutStride_);
    }

    /**
     * @brief 获取 fp32 临时计算区。
     */
    __aicore__ inline AscendC::LocalTensor<float32_t> TempTensor()
    {
        return UbTensor<float32_t>(tempOffset_);
    }

    /**
     * @brief 获取一个 task slot 的 beta cache。
     */
    __aicore__ inline AscendC::LocalTensor<float32_t> SlotBetaCache(uint32_t slot)
    {
        return UbTensor<float32_t>(gateCacheOffset_ + static_cast<uint64_t>(slot) * 2U * GateCacheBytes());
    }

    /**
     * @brief 获取一个 task slot 的 g cache。
     */
    __aicore__ inline AscendC::LocalTensor<float32_t> SlotGCache(uint32_t slot)
    {
        return SlotBetaCache(slot)[GateCacheElems()];
    }

    /**
     * @brief 获取一个 task slot 的 exp(g) cache。
     */
    __aicore__ inline AscendC::LocalTensor<float32_t> SlotGExpCache(uint32_t slot)
    {
        return UbTensor<float32_t>(persistentOffset_ + static_cast<uint64_t>(slot) * GateCacheBytes());
    }

    /**
     * @brief 返回一个 gate cache 向量的元素数。
     */
    __aicore__ inline uint64_t GateCacheElems() const
    {
        return CeilDiv(static_cast<uint64_t>(tiling_->chunkSize) * sizeof(float32_t),
                       static_cast<uint64_t>(BWD_A5_ONE_BLOCK_BYTES)) *
               BWD_A5_ONE_BLOCK_BYTES / sizeof(float32_t);
    }

    /**
     * @brief 返回一个 gate cache 向量的 byte 数。
     */
    __aicore__ inline uint64_t GateCacheBytes() const
    {
        return GateCacheElems() * sizeof(float32_t);
    }

    /**
     * @brief 将 fp32 向量计算长度向上对齐到一个 32B block，避免尾块使用过小 mask。
     */
    __aicore__ inline uint32_t AlignedFp32Count(uint32_t count) const
    {
        return static_cast<uint32_t>(CeilDiv(static_cast<uint64_t>(count), BWD_A5_FP32_PER_BLOCK) *
                                     BWD_A5_FP32_PER_BLOCK);
    }

    /**
     * @brief 获取下一个输入 buffer slot。
     */
    __aicore__ inline uint32_t NextInputSlot()
    {
        uint32_t slot = ioInSlot_;
        ioInSlot_ ^= 1U;
        return slot;
    }

    /**
     * @brief 获取下一个输出 buffer slot。
     */
    __aicore__ inline uint32_t NextOutputSlot()
    {
        uint32_t slot = ioOutSlot_;
        ioOutSlot_ ^= 1U;
        return slot;
    }

    /**
     * @brief 输入 buffer slot 使用的 event id。
     */
    __aicore__ inline int32_t InputEventId(uint32_t slot) const
    {
        return static_cast<int32_t>(slot);
    }

    /**
     * @brief 输出 buffer slot 使用的 event id。
     */
    __aicore__ inline int32_t OutputEventId(uint32_t slot) const
    {
        return static_cast<int32_t>(slot);
    }

    /**
     * @brief 等待一个输入 buffer slot 可复用。
     */
    __aicore__ inline void WaitInputSlot(uint32_t slot) const
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(InputEventId(slot));
    }

    /**
     * @brief 标记一个输入 buffer slot 可被 MTE2 复用。
     */
    __aicore__ inline void ReleaseInputSlot(uint32_t slot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(InputEventId(slot));
    }

    /**
     * @brief 等待一个输出 buffer slot 可复用。
     */
    __aicore__ inline void WaitOutputSlot(uint32_t slot) const
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OutputEventId(slot));
    }

    /**
     * @brief 标记一个输出 buffer slot 可被 VEC 复用。
     */
    __aicore__ inline void ReleaseOutputSlot(uint32_t slot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OutputEventId(slot));
    }

    /**
     * @brief 建立 MTE2 到 VEC 的单 slot 同步。
     */
    __aicore__ inline void SyncMte2ToV(uint32_t slot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(InputEventId(slot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(InputEventId(slot));
    }

    /**
     * @brief 建立 VEC 到 MTE3 的单 slot 同步。
     */
    __aicore__ inline void SyncVToMte3(uint32_t slot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OutputEventId(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OutputEventId(slot));
    }

    /**
     * @brief 计算 gate 张量的 GM offset。
     */
    __aicore__ inline uint64_t GateOffset(uint64_t batch, uint64_t head, uint64_t token) const
    {
        return (batch * static_cast<uint64_t>(tiling_->HV) + head) * static_cast<uint64_t>(tiling_->T) + token;
    }

    /**
     * @brief 计算 key/dk 张量的 GM offset。
     */
    __aicore__ inline uint64_t KeyOffset(uint64_t batch, uint64_t head, uint64_t token, uint64_t col) const
    {
        return (((batch * static_cast<uint64_t>(tiling_->HK) + head) * static_cast<uint64_t>(tiling_->T) + token) *
                static_cast<uint64_t>(tiling_->K)) + col;
    }

    /**
     * @brief 计算 value/dv 张量的 GM offset。
     */
    __aicore__ inline uint64_t ValueOffset(uint64_t batch, uint64_t head, uint64_t token, uint64_t col) const
    {
        return (((batch * static_cast<uint64_t>(tiling_->HV) + head) * static_cast<uint64_t>(tiling_->T) + token) *
                static_cast<uint64_t>(tiling_->V)) + col;
    }

    /**
     * @brief 返回两个整数的较小值。
     */
    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    /**
     * @brief 正整数向上整除。
     */
    __aicore__ inline uint64_t CeilDiv(uint64_t lhs, uint64_t rhs) const
    {
        return rhs == 0 ? 0 : (lhs + rhs - 1) / rhs;
    }

    const PrepareWyReprBwdTilingData *tiling_ = nullptr;
    uint64_t coreIdx_ = 0;
    uint64_t coreNum_ = 1;
    uint64_t subBlockIdx_ = 0;
    uint64_t subBlockNum_ = 1;

    AscendC::GlobalTensor<T> kGm_;
    AscendC::GlobalTensor<T> vGm_;
    AscendC::GlobalTensor<GateT> betaGm_;
    AscendC::GlobalTensor<GateT> gGm_;
    AscendC::GlobalTensor<T> dkGm_;
    AscendC::GlobalTensor<T> dvGm_;
    AscendC::GlobalTensor<GateT> dbetaGm_;
    AscendC::GlobalTensor<GateT> dgGm_;
    AscendC::GlobalTensor<uint64_t> cuSeqlensGm_;
    AscendC::GlobalTensor<uint64_t> chunkIndicesGm_;

    GDN::A5Pipeline::RawLocalBuffer<AscendC::TPosition::VECCALC, 248U * 1024U> ubBuf_;
    uint64_t ioInOffset_ = 0;
    uint64_t ioOutOffset_ = 0;
    uint64_t persistentOffset_ = 0;
    uint64_t gateCacheOffset_ = 0;
    uint64_t tempOffset_ = 0;
    uint64_t ioInStride_ = 0;
    uint64_t ioOutStride_ = 0;
    uint64_t ubBytes_ = 0;
    uint32_t ioInSlot_ = 0;
    uint32_t ioOutSlot_ = 0;
};

} // namespace GDN

#endif // PREPARE_WY_REPR_BWD_ARCH35_VECTOR_H

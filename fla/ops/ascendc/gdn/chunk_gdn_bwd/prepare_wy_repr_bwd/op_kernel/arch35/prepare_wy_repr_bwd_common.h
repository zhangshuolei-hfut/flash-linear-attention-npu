/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_ARCH35_COMMON_H
#define PREPARE_WY_REPR_BWD_ARCH35_COMMON_H

#include "kernel_operator.h"

namespace GDN::A5Pipeline {

static constexpr uint32_t HEAD_SLOT_COUNT = 2;

struct ChunkTask {
    uint64_t linear;
    uint64_t batch;
    uint64_t chunkBegin;
    uint32_t chunkLen;
};

struct HeadTask {
    ChunkTask chunk;
    uint64_t headBase;
};

template <typename T, int V, int CHUNK_SIZE>
struct BufferLayout {
    static constexpr uint32_t kMaxWidth = V > 128 ? V : 128;
    static constexpr uint32_t kSplitResultBytes = (CHUNK_SIZE / 2U) * kMaxWidth * sizeof(T);
    static constexpr uint32_t kBroadcastResultBytes = CHUNK_SIZE * CHUNK_SIZE * sizeof(T);
    static constexpr uint32_t kResultSlotBytes =
        kSplitResultBytes > kBroadcastResultBytes ? kSplitResultBytes : kBroadcastResultBytes;
    static constexpr uint32_t kResultBytes = 2U * kResultSlotBytes;

    static constexpr uint32_t kL1SlotBytes = CHUNK_SIZE * kMaxWidth * sizeof(T);
    static constexpr uint32_t kL1Bytes = 2U * kL1SlotBytes;
    static constexpr uint32_t kL1TileKBytes = CHUNK_SIZE * 128U * sizeof(T);
    static constexpr uint32_t kL1TileVBytes = CHUNK_SIZE * V * sizeof(T);
    static constexpr uint32_t kL1TileABytes = CHUNK_SIZE * CHUNK_SIZE * sizeof(T);
    static constexpr uint32_t kL1ScratchKBytes = 3U * kL1TileKBytes;
    static constexpr uint32_t kL1ScratchVBytes = 2U * kL1TileVBytes;
    static constexpr uint32_t kL1ScratchBytes =
        kL1ScratchKBytes > kL1ScratchVBytes ? kL1ScratchKBytes : kL1ScratchVBytes;
    static constexpr uint32_t kResidentDaBaseOffset = kL1ScratchBytes;
    static constexpr uint32_t kResidentDwBaseOffset = kResidentDaBaseOffset + 2U * kL1TileABytes;
    static constexpr uint32_t kResidentDuBaseOffset = kResidentDwBaseOffset + 2U * kL1TileKBytes;
    static constexpr uint32_t kResidentATBaseOffset = kResidentDuBaseOffset + 2U * kL1TileVBytes;
    static constexpr uint32_t kResidentKBaseOffset = kResidentATBaseOffset + 2U * kL1TileABytes;
    static_assert(kResidentKBaseOffset + 2U * kL1TileKBytes <= 512U * 1024U,
                  "A5 cube L1 layout exceeds 512 KiB");

    __aicore__ static inline uint32_t ResultOffset(uint32_t slot)
    {
        return slot * kResultSlotBytes;
    }

    __aicore__ static inline uint32_t L1Offset(uint32_t slot)
    {
        return slot * kL1SlotBytes;
    }

    __aicore__ static inline uint32_t ResidentDaOffset(uint32_t slot)
    {
        return kResidentDaBaseOffset + slot * kL1TileABytes;
    }
};

__aicore__ inline uint32_t LocalRowBegin(uint32_t subBlockIdx, uint32_t chunkSize)
{
    return subBlockIdx * (chunkSize / 2U);
}

__aicore__ inline uint32_t LocalRowCount(uint32_t subBlockIdx, uint32_t chunkLen, uint32_t chunkSize)
{
    uint32_t begin = LocalRowBegin(subBlockIdx, chunkSize);
    if (begin >= chunkLen) {
        return 0;
    }
    uint32_t rows = chunkSize / 2U;
    return begin + rows > chunkLen ? chunkLen - begin : rows;
}

} // namespace GDN::A5Pipeline

#endif // PREPARE_WY_REPR_BWD_ARCH35_COMMON_H

/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_CUBE_H
#define PREPARE_WY_REPR_BWD_CUBE_H

#define CATLASS_ARCH 2201
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "prepare_wy_repr_bwd_struct.h"

namespace GDN {

using PrepareWyReprBwdInt64Tensor = AscendC::GlobalTensor<uint64_t>;
template <typename T, bool IS_VARLEN, int V, int CHUNK_SIZE>
class PrepareWyReprBwdCube {
    using ArchTag = Catlass::Arch::AtlasA2;
    using L0TileShapeDa1 = tla::Shape<tla::Int<CHUNK_SIZE>, tla::Int<CHUNK_SIZE>, tla::Int<128>>;
    using L0TileShapeDa2 = tla::Shape<tla::Int<CHUNK_SIZE>, tla::Int<CHUNK_SIZE>, tla::Int<V>>;
    using L0TileShapeDa5 =
        tla::Shape<tla::Int<CHUNK_SIZE>, tla::Int<CHUNK_SIZE>, tla::Int<CHUNK_SIZE>>;
    using L0TileShapeDa6 = L0TileShapeDa5;
    using L0TileShapeFullK = tla::Shape<tla::Int<CHUNK_SIZE>, tla::Int<128>, tla::Int<CHUNK_SIZE>>;
    static constexpr int DVB_TILE_M = V == 128 ? 128 : CHUNK_SIZE;
    static constexpr int DVB_TILE_N = V == 128 ? 256 : V;
    static constexpr int DVB_TILE_K = V == 128 ? 64 : CHUNK_SIZE;
    using L0TileShapeDvb = tla::Shape<tla::Int<DVB_TILE_M>, tla::Int<DVB_TILE_N>, tla::Int<DVB_TILE_K>>;
    using L0TileShapeKkt = L0TileShapeDa1;
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;

    using TileCopyDa =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, T, RowMajor, T, ColumnMajor, T, RowMajor>;
    using TileCopyDa6 =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, T, ColumnMajor, T, RowMajor, T, RowMajor>;
    using TileCopyDK =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, T, RowMajor, T, RowMajor, T, RowMajor>;
    using TileCopyDkb =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, T, ColumnMajor, T, RowMajor, T, RowMajor>;
    using TileCopyDkbg =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, T, ColumnMajor, T, RowMajor, T, RowMajor>;
    using TileCopyDvb =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, T, ColumnMajor, T, RowMajor, T, RowMajor>;
    using TileCopyKkt =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, T, RowMajor, T, ColumnMajor, T, RowMajor>;

    using Da1TileCopy = TileCopyDa;
    using Da1ElementA = typename Da1TileCopy::ElementA;
    using Da1ElementB = typename Da1TileCopy::ElementB;
    using Da1ElementAccumulator = typename Da1TileCopy::ElementAccumulator;
    using Da1LayoutTagL1A = typename Da1TileCopy::LayoutTagL1A;
    using Da1LayoutTagL1B = typename Da1TileCopy::LayoutTagL1B;
    using Da1LayoutTagL0A = typename Da1TileCopy::LayoutTagL0A;
    using Da1LayoutTagL0B = typename Da1TileCopy::LayoutTagL0B;
    using Da1CopyL1ToL0A = typename Da1TileCopy::CopyL1ToL0A;
    using Da1CopyL1ToL0B = typename Da1TileCopy::CopyL1ToL0B;
    using Da1TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Da1ElementA, Da1LayoutTagL1A>;

    using Da2TileCopy = TileCopyDa;
    using Da2ElementA = typename Da2TileCopy::ElementA;
    using Da2ElementB = typename Da2TileCopy::ElementB;
    using Da2ElementAccumulator = typename Da2TileCopy::ElementAccumulator;
    using Da2LayoutTagL1A = typename Da2TileCopy::LayoutTagL1A;
    using Da2LayoutTagL1B = typename Da2TileCopy::LayoutTagL1B;
    using Da2LayoutTagL0A = typename Da2TileCopy::LayoutTagL0A;
    using Da2LayoutTagL0B = typename Da2TileCopy::LayoutTagL0B;
    using Da2CopyL1ToL0B = typename Da2TileCopy::CopyL1ToL0B;
    using Da2TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Da2ElementA, Da2LayoutTagL1A>;
    using Da2ResidentLayoutTagL1A = typename TileCopyDa::LayoutTagL1A;

    using Da5TileCopy = TileCopyDa;
    using Da5ElementA = typename Da5TileCopy::ElementA;
    using Da5ElementB = typename Da5TileCopy::ElementB;
    using Da5ElementAccumulator = typename Da5TileCopy::ElementAccumulator;
    using Da5LayoutTagL1A = typename Da5TileCopy::LayoutTagL1A;
    using Da5LayoutTagL1B = typename Da5TileCopy::LayoutTagL1B;
    using Da5LayoutTagL0A = typename Da5TileCopy::LayoutTagL0A;
    using Da5LayoutTagL0B = typename Da5TileCopy::LayoutTagL0B;
    using Da5CopyL1ToL0A = typename Da5TileCopy::CopyL1ToL0A;
    using Da5TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Da5ElementA, Da5LayoutTagL1A>;
    using Da5ResidentLayoutTagL1B = typename TileCopyDa::LayoutTagL1B;

    using Da6TileCopy = TileCopyDa6;
    using Da6ElementA = typename Da6TileCopy::ElementA;
    using Da6ElementB = typename Da6TileCopy::ElementB;
    using Da6ElementAccumulator = typename Da6TileCopy::ElementAccumulator;
    using Da6LayoutTagL1A = typename Da6TileCopy::LayoutTagL1A;
    using Da6LayoutTagL1B = typename Da6TileCopy::LayoutTagL1B;
    using Da6LayoutTagL0A = typename Da6TileCopy::LayoutTagL0A;
    using Da6LayoutTagL0B = typename Da6TileCopy::LayoutTagL0B;
    using Da6CopyL1ToL0A = typename Da6TileCopy::CopyL1ToL0A;
    using Da6CopyL1ToL0B = typename Da6TileCopy::CopyL1ToL0B;
    using Da6TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Da6ElementA, Da6LayoutTagL1A>;

    using FullDkTileCopy = TileCopyDK;
    using FullDkElementA = typename FullDkTileCopy::ElementA;
    using FullDkElementB = typename FullDkTileCopy::ElementB;
    using FullDkElementAccumulator = typename FullDkTileCopy::ElementAccumulator;
    using FullDkLayoutTagL1A = typename FullDkTileCopy::LayoutTagL1A;
    using FullDkLayoutTagL1B = typename FullDkTileCopy::LayoutTagL1B;
    using FullDkLayoutTagL0A = typename FullDkTileCopy::LayoutTagL0A;
    using FullDkLayoutTagL0B = typename FullDkTileCopy::LayoutTagL0B;
    using FullDkCopyL1ToL0A = typename FullDkTileCopy::CopyL1ToL0A;
    using FullDkCopyL1ToL0B = typename FullDkTileCopy::CopyL1ToL0B;
    using FullDkTileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, FullDkElementA, FullDkLayoutTagL1A>;

    using FullDkbTileCopy = TileCopyDkb;
    using FullDkbElementA = typename FullDkbTileCopy::ElementA;
    using FullDkbElementB = typename FullDkbTileCopy::ElementB;
    using FullDkbElementAccumulator = typename FullDkbTileCopy::ElementAccumulator;
    using FullDkbLayoutTagL1A = typename FullDkbTileCopy::LayoutTagL1A;
    using FullDkbLayoutTagL1B = typename FullDkbTileCopy::LayoutTagL1B;
    using FullDkbLayoutTagL0A = typename FullDkbTileCopy::LayoutTagL0A;
    using FullDkbLayoutTagL0B = typename FullDkbTileCopy::LayoutTagL0B;
    using FullDkbTileMmad =
        Catlass::Gemm::Tile::TileMmadTla<ArchTag, FullDkbElementA, FullDkbLayoutTagL1A>;
    using FullDkbResidentLayoutTagL1A = typename TileCopyDkb::LayoutTagL1A;
    using FullDkbResidentLayoutTagL1B = typename TileCopyDkb::LayoutTagL1B;

    using FullDkbgTileCopy = TileCopyDkbg;
    using FullDkbgElementA = typename FullDkbgTileCopy::ElementA;
    using FullDkbgElementB = typename FullDkbgTileCopy::ElementB;
    using FullDkbgElementAccumulator = typename FullDkbgTileCopy::ElementAccumulator;
    using FullDkbgLayoutTagL1A = typename FullDkbgTileCopy::LayoutTagL1A;
    using FullDkbgLayoutTagL1B = typename FullDkbgTileCopy::LayoutTagL1B;
    using FullDkbgLayoutTagL0A = typename FullDkbgTileCopy::LayoutTagL0A;
    using FullDkbgLayoutTagL0B = typename FullDkbgTileCopy::LayoutTagL0B;
    using FullDkbgCopyL1ToL0A = typename FullDkbgTileCopy::CopyL1ToL0A;
    using FullDkbgCopyL1ToL0B = typename FullDkbgTileCopy::CopyL1ToL0B;
    using FullDkbgTileMmad =
        Catlass::Gemm::Tile::TileMmadTla<ArchTag, FullDkbgElementA, FullDkbgLayoutTagL1A>;

    using FullDvbTileCopy = TileCopyDvb;
    using FullDvbElementA = typename FullDvbTileCopy::ElementA;
    using FullDvbElementB = typename FullDvbTileCopy::ElementB;
    using FullDvbElementAccumulator = typename FullDvbTileCopy::ElementAccumulator;
    using FullDvbLayoutTagL1A = typename FullDvbTileCopy::LayoutTagL1A;
    using FullDvbLayoutTagL1B = typename FullDvbTileCopy::LayoutTagL1B;
    using FullDvbLayoutTagL0A = typename FullDvbTileCopy::LayoutTagL0A;
    using FullDvbLayoutTagL0B = typename FullDvbTileCopy::LayoutTagL0B;
    using FullDvbTileMmad =
        Catlass::Gemm::Tile::TileMmadTla<ArchTag, FullDvbElementA, FullDvbLayoutTagL1A>;
    using FullDvbResidentLayoutTagL1A = typename TileCopyDa::LayoutTagL1B;
    using FullDvbResidentLayoutTagL1B = typename TileCopyDa::LayoutTagL1A;

    using FullKktTileCopy = TileCopyKkt;
    using FullKktElementA = typename FullKktTileCopy::ElementA;
    using FullKktElementB = typename FullKktTileCopy::ElementB;
    using FullKktElementAccumulator = typename FullKktTileCopy::ElementAccumulator;
    using FullKktLayoutTagL1A = typename FullKktTileCopy::LayoutTagL1A;
    using FullKktLayoutTagL1B = typename FullKktTileCopy::LayoutTagL1B;
    using FullKktLayoutTagL0A = typename FullKktTileCopy::LayoutTagL0A;
    using FullKktLayoutTagL0B = typename FullKktTileCopy::LayoutTagL0B;
    using FullKktCopyL1ToL0A = typename FullKktTileCopy::CopyL1ToL0A;
    using FullKktCopyL1ToL0B = typename FullKktTileCopy::CopyL1ToL0B;
    using FullKktTileMmad =
        Catlass::Gemm::Tile::TileMmadTla<ArchTag, FullKktElementA, FullKktLayoutTagL1A>;

    static constexpr uint32_t da1TileM = tla::get<0>(L0TileShapeDa1{});
    static constexpr uint32_t da1TileN = tla::get<1>(L0TileShapeDa1{});
    static constexpr uint32_t da1TileK = tla::get<2>(L0TileShapeDa1{});
    static constexpr uint32_t da1L1ATileBytes = da1TileM * da1TileK * sizeof(Da1ElementA);
    static constexpr uint32_t da1L0ATileBytes = da1TileM * da1TileK * sizeof(Da1ElementA);
    static constexpr uint32_t da1L0BTileBytes = da1TileK * da1TileN * sizeof(Da1ElementB);

    static constexpr uint32_t da2TileM = tla::get<0>(L0TileShapeDa2{});
    static constexpr uint32_t da2TileN = tla::get<1>(L0TileShapeDa2{});
    static constexpr uint32_t da2TileK = tla::get<2>(L0TileShapeDa2{});
    static constexpr uint32_t da2L1ATileBytes = da2TileM * da2TileK * sizeof(Da2ElementA);
    static constexpr uint32_t da2L0ATileBytes = da2TileM * da2TileK * sizeof(Da2ElementA);
    static constexpr uint32_t da2L0BTileBytes = da2TileK * da2TileN * sizeof(Da2ElementB);

    static constexpr uint32_t da5TileM = tla::get<0>(L0TileShapeDa5{});
    static constexpr uint32_t da5TileN = tla::get<1>(L0TileShapeDa5{});
    static constexpr uint32_t da5TileK = tla::get<2>(L0TileShapeDa5{});
    static constexpr uint32_t da5L1ATileBytes = da5TileM * da5TileK * sizeof(Da5ElementA);
    static constexpr uint32_t da5L0ATileBytes = da5TileM * da5TileK * sizeof(Da5ElementA);
    static constexpr uint32_t da5L0BTileBytes = da5TileK * da5TileN * sizeof(Da5ElementB);

    static constexpr uint32_t da6TileM = tla::get<0>(L0TileShapeDa6{});
    static constexpr uint32_t da6TileN = tla::get<1>(L0TileShapeDa6{});
    static constexpr uint32_t da6TileK = tla::get<2>(L0TileShapeDa6{});
    static constexpr uint32_t da6L1ATileBytes = da6TileM * da6TileK * sizeof(Da6ElementA);
    static constexpr uint32_t da6L0ATileBytes = da6TileM * da6TileK * sizeof(Da6ElementA);
    static constexpr uint32_t da6L0BTileBytes = da6TileK * da6TileN * sizeof(Da6ElementB);

    static constexpr uint32_t fullDkTileM = tla::get<0>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkTileN = tla::get<1>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkTileK = tla::get<2>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkL1ATileBytes = fullDkTileM * fullDkTileK * sizeof(FullDkElementA);
    static constexpr uint32_t fullDkL0ATileBytes = fullDkTileM * fullDkTileK * sizeof(FullDkElementA);
    static constexpr uint32_t fullDkL0BTileBytes = fullDkTileK * fullDkTileN * sizeof(FullDkElementB);

    static constexpr uint32_t fullDkbTileM = tla::get<0>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkbTileN = tla::get<1>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkbTileK = tla::get<2>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkbL1ATileBytes = fullDkbTileM * fullDkbTileK * sizeof(FullDkbElementA);
    static constexpr uint32_t fullDkbL0ATileBytes = fullDkbTileM * fullDkbTileK * sizeof(FullDkbElementA);
    static constexpr uint32_t fullDkbL0BTileBytes = fullDkbTileK * fullDkbTileN * sizeof(FullDkbElementB);

    static constexpr uint32_t fullDkbgTileM = tla::get<0>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkbgTileN = tla::get<1>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkbgTileK = tla::get<2>(L0TileShapeFullK{});
    static constexpr uint32_t fullDkbgL1ATileBytes =
        fullDkbgTileM * fullDkbgTileK * sizeof(FullDkbgElementA);
    static constexpr uint32_t fullDkbgL0ATileBytes =
        fullDkbgTileM * fullDkbgTileK * sizeof(FullDkbgElementA);
    static constexpr uint32_t fullDkbgL0BTileBytes =
        fullDkbgTileK * fullDkbgTileN * sizeof(FullDkbgElementB);

    static constexpr uint32_t fullDvbTileM = tla::get<0>(L0TileShapeDvb{});
    static constexpr uint32_t fullDvbTileN = tla::get<1>(L0TileShapeDvb{});
    static constexpr uint32_t fullDvbTileK = tla::get<2>(L0TileShapeDvb{});
    static constexpr uint32_t fullDvbL1ATileBytes =
        fullDvbTileM * fullDvbTileK * sizeof(FullDvbElementA);
    static constexpr uint32_t fullDvbL0ATileBytes =
        fullDvbTileM * fullDvbTileK * sizeof(FullDvbElementA);
    static constexpr uint32_t fullDvbL0BTileBytes =
        fullDvbTileK * static_cast<uint32_t>(V) * sizeof(FullDvbElementB);

    static constexpr uint32_t fullKktTileM = tla::get<0>(L0TileShapeKkt{});
    static constexpr uint32_t fullKktTileN = tla::get<1>(L0TileShapeKkt{});
    static constexpr uint32_t fullKktTileK = tla::get<2>(L0TileShapeKkt{});
    static constexpr uint32_t fullKktL1ATileBytes = fullKktTileM * fullKktTileK * sizeof(FullKktElementA);
    static constexpr uint32_t fullKktL0ATileBytes = fullKktTileM * fullKktTileK * sizeof(FullKktElementA);
    static constexpr uint32_t fullKktL0BTileBytes = fullKktTileK * fullKktTileN * sizeof(FullKktElementB);

    static constexpr uint32_t L1_TILE_K_BYTES = static_cast<uint32_t>(CHUNK_SIZE) * 128U * sizeof(T);
    static constexpr uint32_t L1_TILE_V_BYTES =
        static_cast<uint32_t>(CHUNK_SIZE) * static_cast<uint32_t>(V) * sizeof(T);
    static constexpr uint32_t L1_TILE_A_BYTES =
        static_cast<uint32_t>(CHUNK_SIZE) * static_cast<uint32_t>(CHUNK_SIZE) * sizeof(T);
    static constexpr uint32_t L1_SCRATCH_K_BYTES = 3U * L1_TILE_K_BYTES;
    static constexpr uint32_t L1_SCRATCH_V_BYTES = 2U * L1_TILE_V_BYTES;
    static constexpr uint32_t L1_SCRATCH_BYTES =
        L1_SCRATCH_K_BYTES > L1_SCRATCH_V_BYTES ? L1_SCRATCH_K_BYTES : L1_SCRATCH_V_BYTES;
    static constexpr uint32_t RESIDENT_DW_BASE_OFFSET = L1_SCRATCH_BYTES;
    static constexpr uint32_t RESIDENT_DU_BASE_OFFSET = RESIDENT_DW_BASE_OFFSET + 2U * L1_TILE_K_BYTES;
    static constexpr uint32_t RESIDENT_AT_BASE_OFFSET = RESIDENT_DU_BASE_OFFSET + 2U * L1_TILE_V_BYTES;
    static constexpr uint32_t RESIDENT_DA_BASE_OFFSET = RESIDENT_AT_BASE_OFFSET + 2U * L1_TILE_A_BYTES;
    static constexpr uint32_t RESIDENT_K_BASE_OFFSET = RESIDENT_DA_BASE_OFFSET + 2U * L1_TILE_A_BYTES;
    static constexpr int32_t PREFETCH_DU_EVENT_BASE = 2;
    static constexpr int32_t PREFETCH_AT_EVENT_BASE = 4;
    static constexpr uint32_t L0_STAGE_NUM = 2;
    static constexpr int32_t L0A_EVENT_BASE = 0;
    static constexpr int32_t L0B_EVENT_BASE = 2;
    static constexpr int32_t L0_READY_EVENT_BASE = 0;

    __aicore__ inline int32_t L0AEvent(uint32_t stage) const
    {
        return L0A_EVENT_BASE + static_cast<int32_t>(stage);
    }

    __aicore__ inline int32_t L0BEvent(uint32_t stage) const
    {
        return L0B_EVENT_BASE + static_cast<int32_t>(stage);
    }

    __aicore__ inline int32_t L0ReadyEvent(uint32_t stage) const
    {
        return L0_READY_EVENT_BASE + static_cast<int32_t>(stage);
    }

    __aicore__ inline void MarkL0BuffersFree() const
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(0));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(1));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(0));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(1));
    }

    __aicore__ inline void WaitL0BuffersFree() const
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(0));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(1));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(0));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(1));
    }

    __aicore__ inline void InitGemmEvents(int32_t eventL1A, int32_t eventL1B, int32_t eventL0C) const
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
        MarkL0BuffersFree();
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
    }

    __aicore__ inline void DrainGemmEvents(int32_t eventL1A, int32_t eventL1B, int32_t eventL0C) const
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
        WaitL0BuffersFree();
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);
    }

    __aicore__ inline void SignalAicFinishStore()
    {
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
    }

    __aicore__ inline void SignalAicFinishFullStore()
    {
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
    }

    __aicore__ inline void WaitAivFinishStore()
    {
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAivFinishStore);
    }

    __aicore__ inline int32_t PrefetchDuEvent(uint64_t slot) const
    {
        return PREFETCH_DU_EVENT_BASE + static_cast<int32_t>(slot);
    }

    __aicore__ inline int32_t PrefetchATEvent(uint64_t slot) const
    {
        return PREFETCH_AT_EVENT_BASE + static_cast<int32_t>(slot);
    }

    __aicore__ inline uint64_t ResidentKSlot(uint64_t headBase, uint64_t slot, uint64_t valueHead) const
    {
        uint64_t keyHead = valueHead / static_cast<uint64_t>(tiling_->headGroup);
        uint64_t firstKeyHead = headBase / static_cast<uint64_t>(tiling_->headGroup);
        return (slot != 0U && keyHead == firstKeyHead) ? 0U : slot;
    }

    __aicore__ inline uint32_t ResidentKOffset(uint64_t headBase, uint64_t slot, uint64_t valueHead) const
    {
        return RESIDENT_K_BASE_OFFSET +
               static_cast<uint32_t>(ResidentKSlot(headBase, slot, valueHead)) * L1_TILE_K_BYTES;
    }

    __aicore__ inline bool IsResidentKOwner(uint64_t headBase, uint64_t slot, uint64_t valueHead) const
    {
        return ResidentKSlot(headBase, slot, valueHead) == slot;
    }

    __aicore__ inline void PrefetchDuResident(uint64_t batch, uint64_t valueHead, uint64_t chunkBegin,
                                              uint32_t chunkLen, uint32_t residentDuOffset, int32_t eventId)
    {
        using TileCopy = TileCopyDa;
        using ElementA = typename TileCopy::ElementA;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;

        static constexpr uint32_t tileM = tla::get<0>(L0TileShapeDa2{});

        AscendC::GlobalTensor<T> gmDu;
        gmDu.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(du_) + ValueOffset(batch, valueHead, chunkBegin, 0));
        auto layoutDu = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->V);
        auto tensorA = tla::MakeTensor(gmDu, layoutDu, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0),
                              tla::MakeShape(chunkLen, static_cast<uint32_t>(tiling_->V)));

        auto l1ABuf = resource_.l1Buf.template GetBufferByByte<ElementA>(residentDuOffset);
        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<tileM>{}, tla::Int<V>{});
        auto tensorL1A = tla::MakeTensor(l1ABuf, layoutL1A, Catlass::Arch::PositionL1{});

        typename TileCopy::template CopyGmToL1A<decltype(blockA)> copyGmToL1A;
        copyGmToL1A(tensorL1A, blockA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventId);
    }

    __aicore__ inline void PrefetchATResident(uint64_t batch, uint64_t valueHead, uint64_t chunkBegin,
                                              uint32_t chunkLen, uint32_t residentATOffset, int32_t eventId)
    {
        using TileCopy = TileCopyDa;
        using ElementB = typename TileCopy::ElementB;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        static constexpr uint32_t tileN = tla::get<1>(L0TileShapeDa5{});
        static constexpr uint32_t tileK = tla::get<2>(L0TileShapeDa5{});

        AscendC::GlobalTensor<T> gmAT;
        gmAT.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a_) + MatrixOffset(batch, valueHead, chunkBegin, 0));
        auto layoutAT = tla::MakeLayout<T, ColumnMajor>(tiling_->chunkSize, tiling_->chunkSize);
        auto tensorB = tla::MakeTensor(gmAT, layoutAT, Catlass::Arch::PositionGM{});
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));

        auto l1BBuf = resource_.l1Buf.template GetBufferByByte<ElementB>(residentATOffset);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<tileK>{}, tla::Int<tileN>{});
        auto tensorL1B = tla::MakeTensor(l1BBuf, layoutL1B, Catlass::Arch::PositionL1{});

        typename TileCopy::template CopyGmToL1B<decltype(blockB)> copyGmToL1B;
        copyGmToL1B(tensorL1B, blockB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventId);
    }

    __aicore__ inline void PrefetchKResident(uint64_t batch, uint64_t keyHead, uint64_t chunkBegin,
                                             uint32_t chunkLen, uint32_t residentKOffset, int32_t eventId)
    {
        using TileCopy = TileCopyDkb;
        using ElementB = typename TileCopy::ElementB;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        static constexpr uint32_t tileN = tla::get<1>(L0TileShapeFullK{});
        static constexpr uint32_t tileK = tla::get<2>(L0TileShapeFullK{});

        AscendC::GlobalTensor<T> gmK;
        gmK.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k_) + KeyOffset(batch, keyHead, chunkBegin, 0));
        auto layoutK = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->K);
        auto tensorB = tla::MakeTensor(gmK, layoutK, Catlass::Arch::PositionGM{});
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0),
                              tla::MakeShape(chunkLen, static_cast<uint32_t>(tiling_->K)));

        auto l1BBuf = resource_.l1Buf.template GetBufferByByte<ElementB>(residentKOffset);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<tileK>{}, tla::Int<tileN>{});
        auto tensorL1B = tla::MakeTensor(l1BBuf, layoutL1B, Catlass::Arch::PositionL1{});

        typename TileCopy::template CopyGmToL1B<decltype(blockB)> copyGmToL1B;
        copyGmToL1B(tensorL1B, blockB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventId);
    }

public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR v, GM_ADDR A, GM_ADDR dw, GM_ADDR du,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
                                const PrepareWyReprBwdTilingData *tiling)
    {
        k_ = k;
        v_ = v;
        a_ = A;
        dw_ = dw;
        du_ = du;
        workspace_ = workspace;
        tiling_ = tiling;
        coreIdx_ = static_cast<uint64_t>(AscendC::GetBlockIdx());
        if (cuSeqlens != nullptr) {
            cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(cuSeqlens));
        }
        if (chunkIndices != nullptr) {
            chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(chunkIndices));
        }
    }

    __aicore__ inline void ProcessPipeline()
    {
        constexpr int32_t eventL1A = 0;
        constexpr int32_t eventL1B = 1;
        constexpr int32_t eventL0C = 0;

        auto layoutDw = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->K);
        auto layoutKbgT = tla::MakeLayout<T, ColumnMajor>(tiling_->K, tiling_->chunkSize);
        auto layoutDu = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->V);
        auto layoutVbT = tla::MakeLayout<T, ColumnMajor>(tiling_->V, tiling_->chunkSize);
        auto layoutDa = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->chunkSize);
        auto layoutDaT = tla::MakeLayout<T, ColumnMajor>(tiling_->chunkSize, tiling_->chunkSize);
        auto layoutA = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->chunkSize);
        auto layoutAT = tla::MakeLayout<T, ColumnMajor>(tiling_->chunkSize, tiling_->chunkSize);
        auto layoutK = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->K);
        auto layoutKT = tla::MakeLayout<T, ColumnMajor>(tiling_->K, tiling_->chunkSize);
        auto layoutKOut = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->K);
        auto layoutV = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->V);
        auto layoutVOut = tla::MakeLayout<T, RowMajor>(tiling_->chunkSize, tiling_->V);
        constexpr uint64_t slotCount = 2;

        const uint64_t coreLoops = static_cast<uint64_t>(tiling_->chunkNum);
        for (uint64_t loopIdx = coreIdx_; loopIdx < coreLoops;
             loopIdx += static_cast<uint64_t>(AscendC::GetBlockNum())) {
            uint64_t batch = 0;
            uint64_t chunkBegin = 0;
            uint64_t chunkEnd = 0;
            GetChunkOffset(loopIdx, batch, chunkBegin, chunkEnd);
            if (chunkEnd <= chunkBegin) {
                continue;
            }
            uint32_t chunkLen = static_cast<uint32_t>(chunkEnd - chunkBegin);
            for (uint64_t headBase = 0; headBase < static_cast<uint64_t>(tiling_->HV); headBase += slotCount) {
                // Advance both head slots one dependency stage at a time to overlap AIC(slot 1) with AIV(slot 0).
                for (uint32_t pipelineStage = 0; pipelineStage < 3; ++pipelineStage) {
                    if (pipelineStage == 2) {
                        // Return AIV credit for both final-dA slots before either long full stage can block on AIV.
                        for (uint64_t slot = 0; slot < slotCount; ++slot) {
                            uint64_t valueHead = 0;
                            if (ResolvePipelineHeadSlot(headBase, slot, valueHead)) {
                                WaitAivFinishStore();
                            }
                        }
                    }
                    for (uint64_t slot = 0; slot < slotCount; ++slot) {
                    uint64_t valueHead = 0;
                    if (!ResolvePipelineHeadSlot(headBase, slot, valueHead)) {
                        continue;
                    }
                    InitGemmEvents(eventL1A, eventL1B, eventL0C);
                    uint32_t residentDwOffset = RESIDENT_DW_BASE_OFFSET + static_cast<uint32_t>(slot) * L1_TILE_K_BYTES;
                    uint32_t residentDuOffset = RESIDENT_DU_BASE_OFFSET + static_cast<uint32_t>(slot) * L1_TILE_V_BYTES;
                    uint32_t residentATOffset = RESIDENT_AT_BASE_OFFSET + static_cast<uint32_t>(slot) * L1_TILE_A_BYTES;
                    if (pipelineStage == 0) {
                    uint32_t da1L1BOffset = da1L1ATileBytes;

                    AscendC::GlobalTensor<T> da1GmDw;
                    AscendC::GlobalTensor<T> da1GmKbgT;
                    AscendC::GlobalTensor<T> da1GmDa1;
                    da1GmDw.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dw_) + (ValueKeyLikeOffset(batch, valueHead, chunkBegin, 0)));
                    da1GmKbgT.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->kbgOffset, SlotValueKeyLikeOffset(valueHead, 0, 0))));
                    da1GmDa1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->da1Offset, SlotMatrixOffset(valueHead, 0, 0))));

                    auto da1TensorA = tla::MakeTensor(da1GmDw, layoutDw, Catlass::Arch::PositionGM{});
                    auto da1TensorB = tla::MakeTensor(da1GmKbgT, layoutKbgT, Catlass::Arch::PositionGM{});
                    auto da1TensorC = tla::MakeTensor(da1GmDa1, layoutDa, Catlass::Arch::PositionGM{});

                    auto da1L1ABuf = resource_.l1Buf.template GetBufferByByte<Da1ElementA>(residentDwOffset);
                    auto da1L1BBuf = resource_.l1Buf.template GetBufferByByte<Da1ElementB>(da1L1BOffset);

                    auto da1L0CBuf = resource_.l0CBuf.template GetBufferByByte<Da1ElementAccumulator>(0);
                    auto da1LayoutL1A = tla::MakeLayout<Da1ElementA, Da1LayoutTagL1A>(tla::Int<da1TileM>{}, tla::Int<da1TileK>{});
                    auto da1LayoutL1B = tla::MakeLayout<Da1ElementB, Da1LayoutTagL1B>(tla::Int<da1TileK>{}, tla::Int<da1TileN>{});
                    auto da1TensorL1A = tla::MakeTensor(da1L1ABuf, da1LayoutL1A, Catlass::Arch::PositionL1{});
                    auto da1TensorL1B = tla::MakeTensor(da1L1BBuf, da1LayoutL1B, Catlass::Arch::PositionL1{});

                    Da1CopyL1ToL0A da1CopyL1ToL0A;
                    Da1CopyL1ToL0B da1CopyL1ToL0B;
                    Da1TileMmad da1TileMmad;
                    uint32_t da1MActual = chunkLen;
                    if (da1MActual == 1) {
                        da1MActual = 16;
                    }

                    auto da1BlockC = GetTile(da1TensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
                    auto da1LayoutL0C = tla::MakeLayoutL0C(da1MActual, chunkLen);
                    auto da1TensorL0C = tla::MakeTensor(da1L0CBuf, da1LayoutL0C, Catlass::Arch::PositionL0C{});
                    auto da1TileL0C = GetTile(da1TensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(da1MActual, chunkLen));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    WaitAivFinishStore();

                    for (uint32_t da1KOffset = 0; da1KOffset < (static_cast<uint32_t>(tiling_->K)); da1KOffset += da1TileK) {
                        uint32_t da1CurK = da1KOffset + da1TileK > (static_cast<uint32_t>(tiling_->K)) ? (static_cast<uint32_t>(tiling_->K)) - da1KOffset : da1TileK;
                        auto da1BlockA = GetTile(da1TensorA, tla::MakeCoord(0, da1KOffset), tla::MakeShape(chunkLen, da1CurK));
                        auto da1BlockB = GetTile(da1TensorB, tla::MakeCoord(da1KOffset, 0), tla::MakeShape(da1CurK, chunkLen));
                        typename Da1TileCopy::template CopyGmToL1A<decltype(da1BlockA)> da1CopyGmToL1A;
                        typename Da1TileCopy::template CopyGmToL1B<decltype(da1BlockB)> da1CopyGmToL1B;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
                        da1CopyGmToL1A(da1TensorL1A, da1BlockA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
                        da1CopyGmToL1B(da1TensorL1B, da1BlockB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);
                        if (da1KOffset == 0) {
                            PrefetchDuResident(batch, valueHead, chunkBegin, chunkLen, residentDuOffset,
                                                   PrefetchDuEvent(slot));
                            PrefetchATResident(batch, valueHead, chunkBegin, chunkLen, residentATOffset,
                                                   PrefetchATEvent(slot));
                        }

                        uint32_t da1L0Stage = (da1KOffset / da1TileK) & 1U;
                        auto da1L0ABuf = resource_.l0ABuf.template GetBufferByByte<Da1ElementA>(da1L0Stage * da1L0ATileBytes);
                        auto da1L0BBuf = resource_.l0BBuf.template GetBufferByByte<Da1ElementB>(da1L0Stage * da1L0BTileBytes);

                        auto da1LayoutL0A = tla::MakeLayout<Da1ElementA, Da1LayoutTagL0A>(da1MActual, da1CurK);
                        auto da1LayoutL0B = tla::MakeLayout<Da1ElementB, Da1LayoutTagL0B>(da1CurK, chunkLen);
                        auto da1TensorL0A = tla::MakeTensor(da1L0ABuf, da1LayoutL0A, Catlass::Arch::PositionL0A{});
                        auto da1TensorL0B = tla::MakeTensor(da1L0BBuf, da1LayoutL0B, Catlass::Arch::PositionL0B{});
                        auto da1TileL1A = GetTile(da1TensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(da1MActual, da1CurK));
                        auto da1TileL1B = GetTile(da1TensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(da1CurK, chunkLen));

                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(da1L0Stage));
                        da1CopyL1ToL0A(da1TensorL0A, da1TileL1A);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(da1L0Stage));
                        da1CopyL1ToL0B(da1TensorL0B, da1TileL1B);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(da1L0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(da1L0Stage));
                        uint8_t da1UnitFlag = (da1KOffset + da1CurK >= (static_cast<uint32_t>(tiling_->K))) ? 0b11 : 0b10;
                        da1TileMmad(da1TileL0C, da1TensorL0A, da1TensorL0B, da1MActual, chunkLen, da1CurK, da1KOffset == 0, da1UnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(da1L0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(da1L0Stage));
                    }

                    typename Da1TileCopy::template CopyL0CToGm<decltype(da1BlockC)> da1CopyL0CToGm;
                    da1CopyL0CToGm(da1BlockC, da1TileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    uint32_t da2L1BOffset = da2L1ATileBytes;

                    AscendC::GlobalTensor<T> da2GmVbT;
                    AscendC::GlobalTensor<T> da2GmDa2;
                    da2GmVbT.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->vbOffset, SlotValueOffset(valueHead, 0, 0))));
                    da2GmDa2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->da2Offset, SlotMatrixOffset(valueHead, 0, 0))));

                    auto da2TensorB = tla::MakeTensor(da2GmVbT, layoutVbT, Catlass::Arch::PositionGM{});
                    auto da2TensorC = tla::MakeTensor(da2GmDa2, layoutDa, Catlass::Arch::PositionGM{});

                    auto da2L1ABuf = resource_.l1Buf.template GetBufferByByte<Da2ElementA>(residentDuOffset);
                    auto da2L1BBuf = resource_.l1Buf.template GetBufferByByte<Da2ElementB>(da2L1BOffset);

                    auto da2L0CBuf = resource_.l0CBuf.template GetBufferByByte<Da2ElementAccumulator>(0);
                    auto da2LayoutL1B = tla::MakeLayout<Da2ElementB, Da2LayoutTagL1B>(tla::Int<da2TileK>{}, tla::Int<da2TileN>{});
                    auto da2LayoutResidentL1A =
                        tla::MakeLayout<Da2ElementA, Da2ResidentLayoutTagL1A>(tla::Int<da2TileM>{}, tla::Int<V>{});
                    auto da2TensorL1B = tla::MakeTensor(da2L1BBuf, da2LayoutL1B, Catlass::Arch::PositionL1{});
                    auto da2TensorResidentL1A = tla::MakeTensor(da2L1ABuf, da2LayoutResidentL1A, Catlass::Arch::PositionL1{});

                    Da2CopyL1ToL0B da2CopyL1ToL0B;
                    Da2TileMmad da2TileMmad;
                    uint32_t da2MActual = chunkLen;
                    if (da2MActual == 1) {
                        da2MActual = 16;
                    }

                    auto da2BlockC = GetTile(da2TensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
                    auto da2LayoutL0C = tla::MakeLayoutL0C(da2MActual, chunkLen);
                    auto da2TensorL0C = tla::MakeTensor(da2L0CBuf, da2LayoutL0C, Catlass::Arch::PositionL0C{});
                    auto da2TileL0C = GetTile(da2TensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(da2MActual, chunkLen));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(PrefetchDuEvent(slot));

                    for (uint32_t da2KOffset = 0; da2KOffset < (static_cast<uint32_t>(tiling_->V)); da2KOffset += da2TileK) {
                        uint32_t da2CurK = da2KOffset + da2TileK > (static_cast<uint32_t>(tiling_->V)) ? (static_cast<uint32_t>(tiling_->V)) - da2KOffset : da2TileK;
                        auto da2BlockB = GetTile(da2TensorB, tla::MakeCoord(da2KOffset, 0), tla::MakeShape(da2CurK, chunkLen));
                        typename Da2TileCopy::template CopyGmToL1B<decltype(da2BlockB)> da2CopyGmToL1B;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
                        da2CopyGmToL1B(da2TensorL1B, da2BlockB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);

                        uint32_t da2L0Stage = (da2KOffset / da2TileK) & 1U;
                        auto da2L0ABuf = resource_.l0ABuf.template GetBufferByByte<Da2ElementA>(da2L0Stage * da2L0ATileBytes);
                        auto da2L0BBuf = resource_.l0BBuf.template GetBufferByByte<Da2ElementB>(da2L0Stage * da2L0BTileBytes);

                        auto da2LayoutL0A = tla::MakeLayout<Da2ElementA, Da2LayoutTagL0A>(da2MActual, da2CurK);
                        auto da2LayoutL0B = tla::MakeLayout<Da2ElementB, Da2LayoutTagL0B>(da2CurK, chunkLen);
                        auto da2TensorL0A = tla::MakeTensor(da2L0ABuf, da2LayoutL0A, Catlass::Arch::PositionL0A{});
                        auto da2TensorL0B = tla::MakeTensor(da2L0BBuf, da2LayoutL0B, Catlass::Arch::PositionL0B{});
                        auto da2TileL1B = GetTile(da2TensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(da2CurK, chunkLen));
                        auto da2ResidentShapeA = tla::MakeShape(da2MActual, da2CurK);
                        auto da2TileResidentL1A =
                            GetTile(da2TensorResidentL1A, tla::MakeCoord(0, da2KOffset), da2ResidentShapeA);
                        Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(da2TileResidentL1A),
                                                         decltype(da2TensorL0A)> da2CopyResidentL1ToL0A;

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(da2L0Stage));
                        da2CopyResidentL1ToL0A(da2TensorL0A, da2TileResidentL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(da2L0Stage));
                        da2CopyL1ToL0B(da2TensorL0B, da2TileL1B);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(da2L0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(da2L0Stage));
                        uint8_t da2UnitFlag = (da2KOffset + da2CurK >= (static_cast<uint32_t>(tiling_->V))) ? 0b11 : 0b10;
                        da2TileMmad(da2TileL0C, da2TensorL0A, da2TensorL0B, da2MActual, chunkLen, da2CurK, da2KOffset == 0, da2UnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(da2L0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(da2L0Stage));
                    }

                    typename Da2TileCopy::template CopyL0CToGm<decltype(da2BlockC)> da2CopyL0CToGm;
                    da2CopyL0CToGm(da2BlockC, da2TileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    SignalAicFinishStore();
                    } else if (pipelineStage == 1) {
                    uint32_t da5L1BOffset = residentATOffset;

                    AscendC::GlobalTensor<T> da5GmDa4;
                    AscendC::GlobalTensor<T> da5GmDa5;
                    da5GmDa4.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->da4Offset, SlotMatrixOffset(valueHead, 0, 0))));
                    da5GmDa5.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->da5Offset, SlotMatrixOffset(valueHead, 0, 0))));

                    auto da5TensorA = tla::MakeTensor(da5GmDa4, layoutDa, Catlass::Arch::PositionGM{});
                    auto da5TensorC = tla::MakeTensor(da5GmDa5, layoutDa, Catlass::Arch::PositionGM{});

                    auto da5L1ABuf = resource_.l1Buf.template GetBufferByByte<Da5ElementA>(0U);
                    auto da5L1BBuf = resource_.l1Buf.template GetBufferByByte<Da5ElementB>(da5L1BOffset);

                    auto da5L0CBuf = resource_.l0CBuf.template GetBufferByByte<Da5ElementAccumulator>(0);
                    auto da5LayoutL1A = tla::MakeLayout<Da5ElementA, Da5LayoutTagL1A>(tla::Int<da5TileM>{}, tla::Int<da5TileK>{});
                    auto da5LayoutResidentL1B =
                        tla::MakeLayout<Da5ElementB, Da5ResidentLayoutTagL1B>(tla::Int<da5TileK>{}, tla::Int<da5TileN>{});
                    auto da5TensorL1A = tla::MakeTensor(da5L1ABuf, da5LayoutL1A, Catlass::Arch::PositionL1{});
                    auto da5TensorResidentL1B = tla::MakeTensor(da5L1BBuf, da5LayoutResidentL1B, Catlass::Arch::PositionL1{});

                    Da5CopyL1ToL0A da5CopyL1ToL0A;
                    Da5TileMmad da5TileMmad;
                    uint32_t da5MActual = chunkLen;
                    if (da5MActual == 1) {
                        da5MActual = 16;
                    }

                    auto da5BlockC = GetTile(da5TensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
                    auto da5LayoutL0C = tla::MakeLayoutL0C(da5MActual, chunkLen);
                    auto da5TensorL0C = tla::MakeTensor(da5L0CBuf, da5LayoutL0C, Catlass::Arch::PositionL0C{});
                    auto da5TileL0C = GetTile(da5TensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(da5MActual, chunkLen));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(PrefetchATEvent(slot));
                    WaitAivFinishStore();

                    for (uint32_t da5KOffset = 0; da5KOffset < (chunkLen); da5KOffset += da5TileK) {
                        uint32_t da5CurK = da5KOffset + da5TileK > (chunkLen) ? (chunkLen) - da5KOffset : da5TileK;
                        auto da5BlockA = GetTile(da5TensorA, tla::MakeCoord(0, da5KOffset), tla::MakeShape(chunkLen, da5CurK));
                        typename Da5TileCopy::template CopyGmToL1A<decltype(da5BlockA)> da5CopyGmToL1A;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
                        da5CopyGmToL1A(da5TensorL1A, da5BlockA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);

                        uint32_t da5L0Stage = (da5KOffset / da5TileK) & 1U;
                        auto da5L0ABuf = resource_.l0ABuf.template GetBufferByByte<Da5ElementA>(da5L0Stage * da5L0ATileBytes);
                        auto da5L0BBuf = resource_.l0BBuf.template GetBufferByByte<Da5ElementB>(da5L0Stage * da5L0BTileBytes);

                        auto da5LayoutL0A = tla::MakeLayout<Da5ElementA, Da5LayoutTagL0A>(da5MActual, da5CurK);
                        auto da5LayoutL0B = tla::MakeLayout<Da5ElementB, Da5LayoutTagL0B>(da5CurK, chunkLen);
                        auto da5TensorL0A = tla::MakeTensor(da5L0ABuf, da5LayoutL0A, Catlass::Arch::PositionL0A{});
                        auto da5TensorL0B = tla::MakeTensor(da5L0BBuf, da5LayoutL0B, Catlass::Arch::PositionL0B{});
                        auto da5TileL1A = GetTile(da5TensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(da5MActual, da5CurK));
                        auto da5ResidentShapeB = tla::MakeShape(da5CurK, chunkLen);
                        auto da5TileResidentL1B = GetTile(da5TensorResidentL1B, tla::MakeCoord(0, 0), da5ResidentShapeB);
                        Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(da5TileResidentL1B),
                                                         decltype(da5TensorL0B)> da5CopyResidentL1ToL0B;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(da5L0Stage));
                        da5CopyL1ToL0A(da5TensorL0A, da5TileL1A);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(da5L0Stage));
                        da5CopyResidentL1ToL0B(da5TensorL0B, da5TileResidentL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(da5L0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(da5L0Stage));
                        uint8_t da5UnitFlag = (da5KOffset + da5CurK >= (chunkLen)) ? 0b11 : 0b10;
                        da5TileMmad(da5TileL0C, da5TensorL0A, da5TensorL0B, da5MActual, chunkLen, da5CurK, da5KOffset == 0, da5UnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(da5L0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(da5L0Stage));
                    }

                    typename Da5TileCopy::template CopyL0CToGm<decltype(da5BlockC)> da5CopyL0CToGm;
                    da5CopyL0CToGm(da5BlockC, da5TileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventL0C);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventL0C);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    uint32_t da6L1BOffset = da6L1ATileBytes;

                    AscendC::GlobalTensor<T> da6GmDa5;
                    AscendC::GlobalTensor<T> da6GmA;
                    AscendC::GlobalTensor<T> da6GmDa;
                    da6GmDa5.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->da5Offset, SlotMatrixOffset(valueHead, 0, 0))));
                    da6GmA.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a_) +
                                           MatrixOffset(batch, valueHead, chunkBegin, 0));
                    da6GmDa.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->daOffset, SlotMatrixOffset(valueHead, 0, 0))));

                    auto da6TensorA = tla::MakeTensor(da6GmDa5, layoutDaT, Catlass::Arch::PositionGM{});
                    auto da6TensorB = tla::MakeTensor(da6GmA, layoutA, Catlass::Arch::PositionGM{});
                    auto da6TensorC = tla::MakeTensor(da6GmDa, layoutDa, Catlass::Arch::PositionGM{});

                    auto da6L1ABuf = resource_.l1Buf.template GetBufferByByte<Da6ElementA>(0U);
                    auto da6L1BBuf = resource_.l1Buf.template GetBufferByByte<Da6ElementB>(da6L1BOffset);

                    auto da6L0CBuf = resource_.l0CBuf.template GetBufferByByte<Da6ElementAccumulator>(0);
                    auto da6LayoutL1A = tla::MakeLayout<Da6ElementA, Da6LayoutTagL1A>(tla::Int<da6TileM>{}, tla::Int<da6TileK>{});
                    auto da6LayoutL1B = tla::MakeLayout<Da6ElementB, Da6LayoutTagL1B>(
                        tla::Int<da6TileK>{}, tla::Int<da6TileN>{});
                    auto da6TensorL1A = tla::MakeTensor(da6L1ABuf, da6LayoutL1A, Catlass::Arch::PositionL1{});
                    auto da6TensorL1B = tla::MakeTensor(da6L1BBuf, da6LayoutL1B, Catlass::Arch::PositionL1{});

                    Da6CopyL1ToL0A da6CopyL1ToL0A;
                    Da6CopyL1ToL0B da6CopyL1ToL0B;
                    Da6TileMmad da6TileMmad;
                    uint32_t da6MActual = chunkLen;
                    if (da6MActual == 1) {
                        da6MActual = 16;
                    }

                    auto da6BlockC = GetTile(da6TensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
                    auto da6LayoutL0C = tla::MakeLayoutL0C(da6MActual, chunkLen);
                    auto da6TensorL0C = tla::MakeTensor(da6L0CBuf, da6LayoutL0C, Catlass::Arch::PositionL0C{});
                    auto da6TileL0C = GetTile(da6TensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(da6MActual, chunkLen));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);

                    for (uint32_t da6KOffset = 0; da6KOffset < (chunkLen); da6KOffset += da6TileK) {
                        uint32_t da6CurK = da6KOffset + da6TileK > (chunkLen) ? (chunkLen) - da6KOffset : da6TileK;
                        auto da6BlockA = GetTile(da6TensorA, tla::MakeCoord(0, da6KOffset), tla::MakeShape(chunkLen, da6CurK));
                        auto da6BlockB = GetTile(da6TensorB, tla::MakeCoord(da6KOffset, 0),
                                                tla::MakeShape(da6CurK, chunkLen));
                        typename Da6TileCopy::template CopyGmToL1A<decltype(da6BlockA)> da6CopyGmToL1A;
                        typename Da6TileCopy::template CopyGmToL1B<decltype(da6BlockB)> da6CopyGmToL1B;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
                        da6CopyGmToL1A(da6TensorL1A, da6BlockA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
                        da6CopyGmToL1B(da6TensorL1B, da6BlockB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);

                        uint32_t da6L0Stage = (da6KOffset / da6TileK) & 1U;
                        auto da6L0ABuf = resource_.l0ABuf.template GetBufferByByte<Da6ElementA>(da6L0Stage * da6L0ATileBytes);
                        auto da6L0BBuf = resource_.l0BBuf.template GetBufferByByte<Da6ElementB>(da6L0Stage * da6L0BTileBytes);

                        auto da6LayoutL0A = tla::MakeLayout<Da6ElementA, Da6LayoutTagL0A>(da6MActual, da6CurK);
                        auto da6LayoutL0B = tla::MakeLayout<Da6ElementB, Da6LayoutTagL0B>(da6CurK, chunkLen);
                        auto da6TensorL0A = tla::MakeTensor(da6L0ABuf, da6LayoutL0A, Catlass::Arch::PositionL0A{});
                        auto da6TensorL0B = tla::MakeTensor(da6L0BBuf, da6LayoutL0B, Catlass::Arch::PositionL0B{});
                        auto da6TileL1A = GetTile(da6TensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(da6MActual, da6CurK));
                        auto da6TileL1B = GetTile(da6TensorL1B, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(da6CurK, chunkLen));

                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(da6L0Stage));
                        da6CopyL1ToL0A(da6TensorL0A, da6TileL1A);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(da6L0Stage));
                        da6CopyL1ToL0B(da6TensorL0B, da6TileL1B);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(da6L0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(da6L0Stage));
                        uint8_t da6UnitFlag = (da6KOffset + da6CurK >= (chunkLen)) ? 0b11 : 0b10;
                        da6TileMmad(da6TileL0C, da6TensorL0A, da6TensorL0B, da6MActual, chunkLen, da6CurK, da6KOffset == 0, da6UnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(da6L0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(da6L0Stage));
                    }

                    typename Da6TileCopy::template CopyL0CToGm<decltype(da6BlockC)> da6CopyL0CToGm;
                    da6CopyL0CToGm(da6BlockC, da6TileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    SignalAicFinishStore();
                    } else {
                    uint64_t keyHead = valueHead / static_cast<uint64_t>(tiling_->headGroup);
                    uint32_t residentKOffset = ResidentKOffset(headBase, slot, valueHead);
                    uint32_t residentDaOffset = RESIDENT_DA_BASE_OFFSET + static_cast<uint32_t>(slot) * L1_TILE_A_BYTES;
                    uint32_t fullDkL1BOffset = fullDkL1ATileBytes;

                    AscendC::GlobalTensor<T> fullDkGmDa;
                    AscendC::GlobalTensor<T> fullDkGmKBeta;
                    AscendC::GlobalTensor<T> fullDkGmFullDk;
                    fullDkGmDa.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->daOffset, SlotMatrixOffset(valueHead, 0, 0))));
                    fullDkGmKBeta.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->kbetaOffset, SlotValueKeyLikeOffset(valueHead, 0, 0))));
                    fullDkGmFullDk.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->fullDkOffset, SlotValueKeyLikeOffset(valueHead, 0, 0))));

                    auto fullDkTensorA = tla::MakeTensor(fullDkGmDa, layoutDa, Catlass::Arch::PositionGM{});
                    auto fullDkTensorB = tla::MakeTensor(fullDkGmKBeta, layoutK, Catlass::Arch::PositionGM{});
                    auto fullDkTensorC = tla::MakeTensor(fullDkGmFullDk, layoutKOut, Catlass::Arch::PositionGM{});

                    auto fullDkL1ABuf = resource_.l1Buf.template GetBufferByByte<FullDkElementA>(residentDaOffset);
                    auto fullDkL1BBuf = resource_.l1Buf.template GetBufferByByte<FullDkElementB>(fullDkL1BOffset);

                    auto fullDkL0CBuf = resource_.l0CBuf.template GetBufferByByte<FullDkElementAccumulator>(0);
                    auto fullDkLayoutL1A = tla::MakeLayout<FullDkElementA, FullDkLayoutTagL1A>(tla::Int<fullDkTileM>{}, tla::Int<fullDkTileK>{});
                    auto fullDkLayoutL1B = tla::MakeLayout<FullDkElementB, FullDkLayoutTagL1B>(tla::Int<fullDkTileK>{}, tla::Int<fullDkTileN>{});
                    auto fullDkTensorL1A = tla::MakeTensor(fullDkL1ABuf, fullDkLayoutL1A, Catlass::Arch::PositionL1{});
                    auto fullDkTensorL1B = tla::MakeTensor(fullDkL1BBuf, fullDkLayoutL1B, Catlass::Arch::PositionL1{});

                    FullDkCopyL1ToL0A fullDkCopyL1ToL0A;
                    FullDkCopyL1ToL0B fullDkCopyL1ToL0B;
                    FullDkTileMmad fullDkTileMmad;
                    uint32_t fullDkMActual = chunkLen;
                    if (fullDkMActual == 1) {
                        fullDkMActual = 16;
                    }

                    auto fullDkBlockC = GetTile(fullDkTensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, static_cast<uint32_t>(tiling_->K)));
                    auto fullDkLayoutL0C = tla::MakeLayoutL0C(fullDkMActual, static_cast<uint32_t>(tiling_->K));
                    auto fullDkTensorL0C = tla::MakeTensor(fullDkL0CBuf, fullDkLayoutL0C, Catlass::Arch::PositionL0C{});
                    auto fullDkTileL0C = GetTile(fullDkTensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(fullDkMActual, static_cast<uint32_t>(tiling_->K)));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);

                    for (uint32_t fullDkKOffset = 0; fullDkKOffset < (chunkLen); fullDkKOffset += fullDkTileK) {
                        uint32_t fullDkCurK = fullDkKOffset + fullDkTileK > (chunkLen) ? (chunkLen) - fullDkKOffset : fullDkTileK;
                        auto fullDkBlockA = GetTile(fullDkTensorA, tla::MakeCoord(0, fullDkKOffset), tla::MakeShape(chunkLen, fullDkCurK));
                        auto fullDkBlockB = GetTile(fullDkTensorB, tla::MakeCoord(fullDkKOffset, 0), tla::MakeShape(fullDkCurK, static_cast<uint32_t>(tiling_->K)));
                        typename FullDkTileCopy::template CopyGmToL1A<decltype(fullDkBlockA)> fullDkCopyGmToL1A;
                        typename FullDkTileCopy::template CopyGmToL1B<decltype(fullDkBlockB)> fullDkCopyGmToL1B;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
                        fullDkCopyGmToL1A(fullDkTensorL1A, fullDkBlockA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
                        fullDkCopyGmToL1B(fullDkTensorL1B, fullDkBlockB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);

                        uint32_t fullDkL0Stage = (fullDkKOffset / fullDkTileK) & 1U;
                        auto fullDkL0ABuf = resource_.l0ABuf.template GetBufferByByte<FullDkElementA>(fullDkL0Stage * fullDkL0ATileBytes);
                        auto fullDkL0BBuf = resource_.l0BBuf.template GetBufferByByte<FullDkElementB>(fullDkL0Stage * fullDkL0BTileBytes);

                        auto fullDkLayoutL0A = tla::MakeLayout<FullDkElementA, FullDkLayoutTagL0A>(fullDkMActual, fullDkCurK);
                        auto fullDkLayoutL0B = tla::MakeLayout<FullDkElementB, FullDkLayoutTagL0B>(fullDkCurK, static_cast<uint32_t>(tiling_->K));
                        auto fullDkTensorL0A = tla::MakeTensor(fullDkL0ABuf, fullDkLayoutL0A, Catlass::Arch::PositionL0A{});
                        auto fullDkTensorL0B = tla::MakeTensor(fullDkL0BBuf, fullDkLayoutL0B, Catlass::Arch::PositionL0B{});
                        auto fullDkTileL1A = GetTile(fullDkTensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(fullDkMActual, fullDkCurK));
                        auto fullDkTileL1B = GetTile(fullDkTensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(fullDkCurK, static_cast<uint32_t>(tiling_->K)));

                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullDkL0Stage));
                        fullDkCopyL1ToL0A(fullDkTensorL0A, fullDkTileL1A);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullDkL0Stage));
                        fullDkCopyL1ToL0B(fullDkTensorL0B, fullDkTileL1B);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullDkL0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullDkL0Stage));
                        uint8_t fullDkUnitFlag = (fullDkKOffset + fullDkCurK >= (chunkLen)) ? 0b11 : 0b10;
                        fullDkTileMmad(fullDkTileL0C, fullDkTensorL0A, fullDkTensorL0B, fullDkMActual, static_cast<uint32_t>(tiling_->K), fullDkCurK, fullDkKOffset == 0, fullDkUnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullDkL0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullDkL0Stage));
                    }

                    typename FullDkTileCopy::template CopyL0CToGm<decltype(fullDkBlockC)> fullDkCopyL0CToGm;
                    fullDkCopyL0CToGm(fullDkBlockC, fullDkTileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    SignalAicFinishFullStore();
                    uint32_t fullDkbL1BOffset = residentKOffset;

                    AscendC::GlobalTensor<T> fullDkbGmFullDkb;
                    fullDkbGmFullDkb.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->fullDkbOffset, SlotValueKeyLikeOffset(valueHead, 0, 0))));

                    auto fullDkbTensorC = tla::MakeTensor(fullDkbGmFullDkb, layoutKOut, Catlass::Arch::PositionGM{});

                    auto fullDkbL1ABuf = resource_.l1Buf.template GetBufferByByte<FullDkbElementA>(residentDaOffset);
                    auto fullDkbL1BBuf = resource_.l1Buf.template GetBufferByByte<FullDkbElementB>(fullDkbL1BOffset);

                    auto fullDkbL0CBuf = resource_.l0CBuf.template GetBufferByByte<FullDkbElementAccumulator>(0);
                    auto fullDkbLayoutResidentL1A =
                        tla::MakeLayout<FullDkbElementA, FullDkbResidentLayoutTagL1A>(tla::Int<fullDkbTileM>{}, tla::Int<fullDkbTileK>{});
                    auto fullDkbLayoutResidentL1B =
                        tla::MakeLayout<FullDkbElementB, FullDkbResidentLayoutTagL1B>(tla::Int<fullDkbTileK>{}, tla::Int<fullDkbTileN>{});
                    auto fullDkbTensorResidentL1A = tla::MakeTensor(fullDkbL1ABuf, fullDkbLayoutResidentL1A, Catlass::Arch::PositionL1{});
                    auto fullDkbTensorResidentL1B = tla::MakeTensor(fullDkbL1BBuf, fullDkbLayoutResidentL1B, Catlass::Arch::PositionL1{});

                    FullDkbTileMmad fullDkbTileMmad;
                    uint32_t fullDkbMActual = chunkLen;
                    if (fullDkbMActual == 1) {
                        fullDkbMActual = 16;
                    }

                    auto fullDkbBlockC = GetTile(fullDkbTensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, static_cast<uint32_t>(tiling_->K)));
                    auto fullDkbLayoutL0C = tla::MakeLayoutL0C(fullDkbMActual, static_cast<uint32_t>(tiling_->K));
                    auto fullDkbTensorL0C = tla::MakeTensor(fullDkbL0CBuf, fullDkbLayoutL0C, Catlass::Arch::PositionL0C{});
                    auto fullDkbTileL0C = GetTile(fullDkbTensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(fullDkbMActual, static_cast<uint32_t>(tiling_->K)));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    if (IsResidentKOwner(headBase, slot, valueHead)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
                        PrefetchKResident(batch, keyHead, chunkBegin, chunkLen,
                                          ResidentKOffset(headBase, slot, valueHead),
                                          eventL1B);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
                    }

                    for (uint32_t fullDkbKOffset = 0; fullDkbKOffset < (chunkLen); fullDkbKOffset += fullDkbTileK) {
                        uint32_t fullDkbCurK = fullDkbKOffset + fullDkbTileK > (chunkLen) ? (chunkLen) - fullDkbKOffset : fullDkbTileK;

                        uint32_t fullDkbL0Stage = (fullDkbKOffset / fullDkbTileK) & 1U;
                        auto fullDkbL0ABuf = resource_.l0ABuf.template GetBufferByByte<FullDkbElementA>(fullDkbL0Stage * fullDkbL0ATileBytes);
                        auto fullDkbL0BBuf = resource_.l0BBuf.template GetBufferByByte<FullDkbElementB>(fullDkbL0Stage * fullDkbL0BTileBytes);

                        auto fullDkbLayoutL0A = tla::MakeLayout<FullDkbElementA, FullDkbLayoutTagL0A>(fullDkbMActual, fullDkbCurK);
                        auto fullDkbLayoutL0B = tla::MakeLayout<FullDkbElementB, FullDkbLayoutTagL0B>(fullDkbCurK, static_cast<uint32_t>(tiling_->K));
                        auto fullDkbTensorL0A = tla::MakeTensor(fullDkbL0ABuf, fullDkbLayoutL0A, Catlass::Arch::PositionL0A{});
                        auto fullDkbTensorL0B = tla::MakeTensor(fullDkbL0BBuf, fullDkbLayoutL0B, Catlass::Arch::PositionL0B{});
                        auto fullDkbResidentShapeA = tla::MakeShape(fullDkbMActual, fullDkbCurK);
                        auto fullDkbResidentShapeB = tla::MakeShape(fullDkbCurK, static_cast<uint32_t>(tiling_->K));
                        auto fullDkbTileResidentL1A = GetTile(fullDkbTensorResidentL1A, tla::MakeCoord(0, 0), fullDkbResidentShapeA);
                        auto fullDkbTileResidentL1B = GetTile(fullDkbTensorResidentL1B, tla::MakeCoord(0, 0), fullDkbResidentShapeB);
                        Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(fullDkbTileResidentL1A),
                                                         decltype(fullDkbTensorL0A)> fullDkbCopyResidentL1ToL0A;
                        Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(fullDkbTileResidentL1B),
                                                         decltype(fullDkbTensorL0B)> fullDkbCopyResidentL1ToL0B;

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullDkbL0Stage));
                        fullDkbCopyResidentL1ToL0A(fullDkbTensorL0A, fullDkbTileResidentL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullDkbL0Stage));
                        fullDkbCopyResidentL1ToL0B(fullDkbTensorL0B, fullDkbTileResidentL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullDkbL0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullDkbL0Stage));
                        uint8_t fullDkbUnitFlag = (fullDkbKOffset + fullDkbCurK >= (chunkLen)) ? 0b11 : 0b10;
                        fullDkbTileMmad(fullDkbTileL0C, fullDkbTensorL0A, fullDkbTensorL0B, fullDkbMActual, static_cast<uint32_t>(tiling_->K), fullDkbCurK, fullDkbKOffset == 0, fullDkbUnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullDkbL0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullDkbL0Stage));
                    }

                    typename FullDkbTileCopy::template CopyL0CToGm<decltype(fullDkbBlockC)> fullDkbCopyL0CToGm;
                    fullDkbCopyL0CToGm(fullDkbBlockC, fullDkbTileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    SignalAicFinishFullStore();
                    uint32_t fullDkbgL1AOffset = residentATOffset;
                    uint32_t fullDkbgL1BOffset = residentDwOffset;

                    AscendC::GlobalTensor<T> fullDkbgGmFullDkbg;
                    fullDkbgGmFullDkbg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->fullDkbgOffset, SlotValueKeyLikeOffset(valueHead, 0, 0))));

                    auto fullDkbgTensorC = tla::MakeTensor(fullDkbgGmFullDkbg, layoutKOut, Catlass::Arch::PositionGM{});

                    auto fullDkbgL1ABuf = resource_.l1Buf.template GetBufferByByte<FullDkbgElementA>(fullDkbgL1AOffset);
                    auto fullDkbgL1BBuf = resource_.l1Buf.template GetBufferByByte<FullDkbgElementB>(fullDkbgL1BOffset);

                    auto fullDkbgL0CBuf = resource_.l0CBuf.template GetBufferByByte<FullDkbgElementAccumulator>(0);
                    auto fullDkbgLayoutL1A =
                        tla::MakeLayout<FullDkbgElementA, FullDkbgLayoutTagL1A>(tla::Int<fullDkbgTileM>{}, tla::Int<fullDkbgTileK>{});
                    auto fullDkbgLayoutL1B =
                        tla::MakeLayout<FullDkbgElementB, FullDkbgLayoutTagL1B>(tla::Int<fullDkbgTileK>{}, tla::Int<fullDkbgTileN>{});
                    auto fullDkbgTensorL1A = tla::MakeTensor(fullDkbgL1ABuf, fullDkbgLayoutL1A, Catlass::Arch::PositionL1{});
                    auto fullDkbgTensorL1B = tla::MakeTensor(fullDkbgL1BBuf, fullDkbgLayoutL1B, Catlass::Arch::PositionL1{});

                    FullDkbgCopyL1ToL0A fullDkbgCopyL1ToL0A;
                    FullDkbgCopyL1ToL0B fullDkbgCopyL1ToL0B;
                    FullDkbgTileMmad fullDkbgTileMmad;
                    uint32_t fullDkbgMActual = chunkLen;
                    if (fullDkbgMActual == 1) {
                        fullDkbgMActual = 16;
                    }

                    auto fullDkbgBlockC = GetTile(fullDkbgTensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, static_cast<uint32_t>(tiling_->K)));
                    auto fullDkbgLayoutL0C = tla::MakeLayoutL0C(fullDkbgMActual, static_cast<uint32_t>(tiling_->K));
                    auto fullDkbgTensorL0C = tla::MakeTensor(fullDkbgL0CBuf, fullDkbgLayoutL0C, Catlass::Arch::PositionL0C{});
                    auto fullDkbgTileL0C = GetTile(fullDkbgTensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(fullDkbgMActual, static_cast<uint32_t>(tiling_->K)));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);

                    for (uint32_t fullDkbgKOffset = 0; fullDkbgKOffset < (chunkLen); fullDkbgKOffset += fullDkbgTileK) {
                        uint32_t fullDkbgCurK = fullDkbgKOffset + fullDkbgTileK > (chunkLen) ? (chunkLen) - fullDkbgKOffset : fullDkbgTileK;

                        uint32_t fullDkbgL0Stage = (fullDkbgKOffset / fullDkbgTileK) & 1U;
                        auto fullDkbgL0ABuf = resource_.l0ABuf.template GetBufferByByte<FullDkbgElementA>(fullDkbgL0Stage * fullDkbgL0ATileBytes);
                        auto fullDkbgL0BBuf = resource_.l0BBuf.template GetBufferByByte<FullDkbgElementB>(fullDkbgL0Stage * fullDkbgL0BTileBytes);

                        auto fullDkbgLayoutL0A = tla::MakeLayout<FullDkbgElementA, FullDkbgLayoutTagL0A>(fullDkbgMActual, fullDkbgCurK);
                        auto fullDkbgLayoutL0B = tla::MakeLayout<FullDkbgElementB, FullDkbgLayoutTagL0B>(fullDkbgCurK, static_cast<uint32_t>(tiling_->K));
                        auto fullDkbgTensorL0A = tla::MakeTensor(fullDkbgL0ABuf, fullDkbgLayoutL0A, Catlass::Arch::PositionL0A{});
                        auto fullDkbgTensorL0B = tla::MakeTensor(fullDkbgL0BBuf, fullDkbgLayoutL0B, Catlass::Arch::PositionL0B{});
                        auto fullDkbgTileL1A = GetTile(fullDkbgTensorL1A, tla::MakeCoord(0, 0),
                                                      tla::MakeShape(fullDkbgMActual, fullDkbgCurK));
                        auto fullDkbgTileL1B = GetTile(fullDkbgTensorL1B, tla::MakeCoord(0, 0),
                                                      tla::MakeShape(fullDkbgCurK, static_cast<uint32_t>(tiling_->K)));

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullDkbgL0Stage));
                        fullDkbgCopyL1ToL0A(fullDkbgTensorL0A, fullDkbgTileL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullDkbgL0Stage));
                        fullDkbgCopyL1ToL0B(fullDkbgTensorL0B, fullDkbgTileL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullDkbgL0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullDkbgL0Stage));
                        uint8_t fullDkbgUnitFlag = (fullDkbgKOffset + fullDkbgCurK >= (chunkLen)) ? 0b11 : 0b10;
                        fullDkbgTileMmad(fullDkbgTileL0C, fullDkbgTensorL0A, fullDkbgTensorL0B, fullDkbgMActual, static_cast<uint32_t>(tiling_->K), fullDkbgCurK, fullDkbgKOffset == 0, fullDkbgUnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullDkbgL0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullDkbgL0Stage));
                    }

                    typename FullDkbgTileCopy::template CopyL0CToGm<decltype(fullDkbgBlockC)> fullDkbgCopyL0CToGm;
                    fullDkbgCopyL0CToGm(fullDkbgBlockC, fullDkbgTileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    SignalAicFinishFullStore();
                    uint32_t fullDvbL1BOffset = residentDuOffset;

                    AscendC::GlobalTensor<T> fullDvbGmFullDvb;
                    fullDvbGmFullDvb.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->fullDvbOffset, SlotValueOffset(valueHead, 0, 0))));

                    auto fullDvbTensorC = tla::MakeTensor(fullDvbGmFullDvb, layoutVOut, Catlass::Arch::PositionGM{});

                    auto fullDvbL1ABuf = resource_.l1Buf.template GetBufferByByte<FullDvbElementA>(residentATOffset);
                    auto fullDvbL1BBuf = resource_.l1Buf.template GetBufferByByte<FullDvbElementB>(fullDvbL1BOffset);

                    auto fullDvbL0CBuf = resource_.l0CBuf.template GetBufferByByte<FullDvbElementAccumulator>(0);
                    auto fullDvbLayoutResidentL1A =
                        tla::MakeLayout<FullDvbElementA, FullDvbResidentLayoutTagL1A>(
                            tla::Int<da5TileK>{}, tla::Int<da5TileN>{});
                    auto fullDvbLayoutResidentL1B =
                        tla::MakeLayout<FullDvbElementB, FullDvbResidentLayoutTagL1B>(
                            tla::Int<fullDvbTileM>{}, tla::Int<V>{});
                    auto fullDvbTensorResidentL1A = tla::MakeTensor(fullDvbL1ABuf, fullDvbLayoutResidentL1A, Catlass::Arch::PositionL1{});
                    auto fullDvbTensorResidentL1B = tla::MakeTensor(fullDvbL1BBuf, fullDvbLayoutResidentL1B, Catlass::Arch::PositionL1{});

                    FullDvbTileMmad fullDvbTileMmad;
                    uint32_t fullDvbMActual = chunkLen;
                    if (fullDvbMActual == 1) {
                        fullDvbMActual = 16;
                    }

                    auto fullDvbBlockC = GetTile(fullDvbTensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, static_cast<uint32_t>(tiling_->V)));
                    auto fullDvbLayoutL0C = tla::MakeLayoutL0C(fullDvbMActual, static_cast<uint32_t>(tiling_->V));
                    auto fullDvbTensorL0C = tla::MakeTensor(fullDvbL0CBuf, fullDvbLayoutL0C, Catlass::Arch::PositionL0C{});
                    auto fullDvbTileL0C = GetTile(fullDvbTensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(fullDvbMActual, static_cast<uint32_t>(tiling_->V)));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);

                    for (uint32_t fullDvbKOffset = 0; fullDvbKOffset < (chunkLen); fullDvbKOffset += fullDvbTileK) {
                        uint32_t fullDvbCurK = fullDvbKOffset + fullDvbTileK > (chunkLen) ? (chunkLen) - fullDvbKOffset : fullDvbTileK;
                        // The resident TileCopyTla path aliases its L0 destination; consume K slices serially.
                        uint32_t fullDvbL0Stage = 0U;
                        auto fullDvbL0ABuf = resource_.l0ABuf.template GetBufferByByte<FullDvbElementA>(fullDvbL0Stage * fullDvbL0ATileBytes);
                        auto fullDvbL0BBuf = resource_.l0BBuf.template GetBufferByByte<FullDvbElementB>(fullDvbL0Stage * fullDvbL0BTileBytes);

                        auto fullDvbLayoutL0A = tla::MakeLayout<FullDvbElementA, FullDvbLayoutTagL0A>(fullDvbMActual, fullDvbCurK);
                        auto fullDvbLayoutL0B = tla::MakeLayout<FullDvbElementB, FullDvbLayoutTagL0B>(fullDvbCurK, static_cast<uint32_t>(tiling_->V));
                        auto fullDvbTensorL0A = tla::MakeTensor(fullDvbL0ABuf, fullDvbLayoutL0A, Catlass::Arch::PositionL0A{});
                        auto fullDvbTensorL0B = tla::MakeTensor(fullDvbL0BBuf, fullDvbLayoutL0B, Catlass::Arch::PositionL0B{});
                        auto fullDvbResidentShapeA = tla::MakeShape(fullDvbMActual, fullDvbCurK);
                        auto fullDvbResidentShapeB = tla::MakeShape(fullDvbCurK, static_cast<uint32_t>(tiling_->V));
                        auto fullDvbTileResidentL1A =
                            GetTile(fullDvbTensorResidentL1A, tla::MakeCoord(0, fullDvbKOffset), fullDvbResidentShapeA);
                        auto fullDvbTileResidentL1B =
                            GetTile(fullDvbTensorResidentL1B, tla::MakeCoord(fullDvbKOffset, 0), fullDvbResidentShapeB);
                        Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(fullDvbTileResidentL1A),
                                                         decltype(fullDvbTensorL0A)> fullDvbCopyResidentL1ToL0A;
                        Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(fullDvbTileResidentL1B),
                                                         decltype(fullDvbTensorL0B)> fullDvbCopyResidentL1ToL0B;

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullDvbL0Stage));
                        fullDvbCopyResidentL1ToL0A(fullDvbTensorL0A, fullDvbTileResidentL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullDvbL0Stage));
                        fullDvbCopyResidentL1ToL0B(fullDvbTensorL0B, fullDvbTileResidentL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullDvbL0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullDvbL0Stage));
                        uint8_t fullDvbUnitFlag = (fullDvbKOffset + fullDvbCurK >= (chunkLen)) ? 0b11 : 0b10;
                        fullDvbTileMmad(fullDvbTileL0C, fullDvbTensorL0A, fullDvbTensorL0B, fullDvbMActual, static_cast<uint32_t>(tiling_->V), fullDvbCurK, fullDvbKOffset == 0, fullDvbUnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullDvbL0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullDvbL0Stage));
                    }

                    typename FullDvbTileCopy::template CopyL0CToGm<decltype(fullDvbBlockC)> fullDvbCopyL0CToGm;
                    fullDvbCopyL0CToGm(fullDvbBlockC, fullDvbTileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    SignalAicFinishFullStore();
                    uint32_t fullKktL1AOffset = residentKOffset;
                    uint32_t fullKktL1BOffset = 0;

                    AscendC::GlobalTensor<T> fullKktGmK;
                    AscendC::GlobalTensor<T> fullKktGmFullKkt;
                    fullKktGmK.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k_) +
                                               KeyOffset(batch, keyHead, chunkBegin, 0));
                    fullKktGmFullKkt.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace_) + (SlotWorkspaceElem(slot, tiling_->fullKktOffset, SlotMatrixOffset(valueHead, 0, 0))));

                    auto fullKktTensorB = tla::MakeTensor(fullKktGmK, layoutKT, Catlass::Arch::PositionGM{});
                    auto fullKktTensorC = tla::MakeTensor(fullKktGmFullKkt, layoutDa, Catlass::Arch::PositionGM{});

                    auto fullKktL1ABuf = resource_.l1Buf.template GetBufferByByte<FullKktElementA>(fullKktL1AOffset);
                    auto fullKktL1BBuf = resource_.l1Buf.template GetBufferByByte<FullKktElementB>(fullKktL1BOffset);

                    auto fullKktL0CBuf = resource_.l0CBuf.template GetBufferByByte<FullKktElementAccumulator>(0);
                    auto fullKktLayoutL1A =
                        tla::MakeLayout<FullKktElementA, FullKktLayoutTagL1A>(tla::Int<fullKktTileM>{}, tla::Int<fullKktTileK>{});
                    auto fullKktLayoutL1B =
                        tla::MakeLayout<FullKktElementB, FullKktLayoutTagL1B>(tla::Int<fullKktTileK>{}, tla::Int<fullKktTileN>{});
                    auto fullKktTensorL1A = tla::MakeTensor(fullKktL1ABuf, fullKktLayoutL1A, Catlass::Arch::PositionL1{});
                    auto fullKktTensorL1B = tla::MakeTensor(fullKktL1BBuf, fullKktLayoutL1B, Catlass::Arch::PositionL1{});

                    auto fullKktBlockB = GetTile(fullKktTensorB, tla::MakeCoord(0, 0),
                                                 tla::MakeShape(static_cast<uint32_t>(tiling_->K), chunkLen));
                    typename FullKktTileCopy::template CopyGmToL1B<decltype(fullKktBlockB)> fullKktCopyGmToL1B;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);
                    fullKktCopyGmToL1B(fullKktTensorL1B, fullKktBlockB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);

                    FullKktCopyL1ToL0A fullKktCopyL1ToL0A;
                    FullKktCopyL1ToL0B fullKktCopyL1ToL0B;
                    FullKktTileMmad fullKktTileMmad;
                    uint32_t fullKktMActual = chunkLen;
                    if (fullKktMActual == 1) {
                        fullKktMActual = 16;
                    }

                    auto fullKktBlockC = GetTile(fullKktTensorC, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
                    auto fullKktLayoutL0C = tla::MakeLayoutL0C(fullKktMActual, chunkLen);
                    auto fullKktTensorL0C = tla::MakeTensor(fullKktL0CBuf, fullKktLayoutL0C, Catlass::Arch::PositionL0C{});
                    auto fullKktTileL0C = GetTile(fullKktTensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(fullKktMActual, chunkLen));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventL0C);

                    for (uint32_t fullKktKOffset = 0; fullKktKOffset < (static_cast<uint32_t>(tiling_->K)); fullKktKOffset += fullKktTileK) {
                        uint32_t fullKktCurK = fullKktKOffset + fullKktTileK > (static_cast<uint32_t>(tiling_->K)) ? (static_cast<uint32_t>(tiling_->K)) - fullKktKOffset : fullKktTileK;

                        uint32_t fullKktL0Stage = (fullKktKOffset / fullKktTileK) & 1U;
                        auto fullKktL0ABuf = resource_.l0ABuf.template GetBufferByByte<FullKktElementA>(fullKktL0Stage * fullKktL0ATileBytes);
                        auto fullKktL0BBuf = resource_.l0BBuf.template GetBufferByByte<FullKktElementB>(fullKktL0Stage * fullKktL0BTileBytes);

                        auto fullKktLayoutL0A = tla::MakeLayout<FullKktElementA, FullKktLayoutTagL0A>(fullKktMActual, fullKktCurK);
                        auto fullKktLayoutL0B = tla::MakeLayout<FullKktElementB, FullKktLayoutTagL0B>(fullKktCurK, chunkLen);
                        auto fullKktTensorL0A = tla::MakeTensor(fullKktL0ABuf, fullKktLayoutL0A, Catlass::Arch::PositionL0A{});
                        auto fullKktTensorL0B = tla::MakeTensor(fullKktL0BBuf, fullKktLayoutL0B, Catlass::Arch::PositionL0B{});
                        auto fullKktTileL1A = GetTile(fullKktTensorL1A, tla::MakeCoord(0, 0),
                                                     tla::MakeShape(fullKktMActual, fullKktCurK));
                        auto fullKktTileL1B = GetTile(fullKktTensorL1B, tla::MakeCoord(0, 0),
                                                     tla::MakeShape(fullKktCurK, chunkLen));

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullKktL0Stage));
                        fullKktCopyL1ToL0A(fullKktTensorL0A, fullKktTileL1A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventL1B);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullKktL0Stage));
                        fullKktCopyL1ToL0B(fullKktTensorL0B, fullKktTileL1B);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventL1B);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullKktL0Stage));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ReadyEvent(fullKktL0Stage));
                        uint8_t fullKktUnitFlag = (fullKktKOffset + fullKktCurK >= (static_cast<uint32_t>(tiling_->K))) ? 0b11 : 0b10;
                        fullKktTileMmad(fullKktTileL0C, fullKktTensorL0A, fullKktTensorL0B, fullKktMActual, chunkLen, fullKktCurK, fullKktKOffset == 0, fullKktUnitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AEvent(fullKktL0Stage));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BEvent(fullKktL0Stage));
                    }

                    typename FullKktTileCopy::template CopyL0CToGm<decltype(fullKktBlockC)> fullKktCopyL0CToGm;
                    fullKktCopyL0CToGm(fullKktBlockC, fullKktTileL0C, 0b11);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventL0C);
                    SignalAicFinishFullStore();
                    }
                    DrainGemmEvents(eventL1A, eventL1B, eventL0C);
                    }
                }
                AscendC::PipeBarrier<PIPE_FIX>();
            }
        }
    }

private:
    __aicore__ inline bool ResolvePipelineHeadSlot(uint64_t headBase, uint64_t slot, uint64_t &valueHead) const
    {
        valueHead = headBase + slot;
        return valueHead < static_cast<uint64_t>(tiling_->HV);
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
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

    __aicore__ inline void GetChunkOffset(uint64_t chunkLinear, uint64_t &batch, uint64_t &chunkBegin,
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

private:
    const PrepareWyReprBwdTilingData *tiling_ = nullptr;
    Catlass::Arch::Resource<ArchTag> resource_;
    Catlass::Arch::CrossCoreFlagWithReverse<14> flagAicFinishStore{SYNC_FLAG_2, SYNC_FLAG_3};
    Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_FLAG_4, SYNC_FLAG_5};
    uint64_t coreIdx_ = 0;
    GM_ADDR k_ = nullptr;
    GM_ADDR v_ = nullptr;
    GM_ADDR a_ = nullptr;
    GM_ADDR dw_ = nullptr;
    GM_ADDR du_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    PrepareWyReprBwdInt64Tensor cuSeqlensGm_;
    PrepareWyReprBwdInt64Tensor chunkIndicesGm_;
};

} // namespace GDN

#endif // PREPARE_WY_REPR_BWD_CUBE_H

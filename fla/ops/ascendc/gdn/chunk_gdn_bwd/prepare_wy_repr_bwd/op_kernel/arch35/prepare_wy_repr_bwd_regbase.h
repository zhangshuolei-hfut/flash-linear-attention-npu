/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_ARCH35_REGBASE_H
#define PREPARE_WY_REPR_BWD_ARCH35_REGBASE_H

#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename kType, typename betaType>
class PrepareWyReprBwdRegBase {
public:
    constexpr static CastTrait ctHalf2Fp32Zero = {
        RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE,
    };
    constexpr static CastTrait ctHalf2Fp32One = {
        RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE,
    };
    constexpr static CastTrait ctFp322HalfZero = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, AscendC::RoundMode::CAST_ROUND,
    };
    constexpr static CastTrait ctFp322HalfOne = {
        RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND,
    };

    __simd_vf__ inline void ProcessVBetaComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessVBetaComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, uint16_t, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessVBetaComputerVFTwoCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessMDuDwComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t, uint32_t);
    __simd_vf__ inline void ProcessMDuDwComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t, uint32_t, uint16_t);
    __simd_vf__ inline void ProcessKBetaGComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessKBetaGComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *,
        uint16_t, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessKBetaGComputerVFTwoCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessKBetaAndKBetaGComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *,
        uint16_t, uint16_t);
    __simd_vf__ inline void ProcessKBetaAndKBetaGComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *,
        uint16_t, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessGComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *,
        uint16_t, uint16_t, uint32_t, uint32_t);
    __simd_vf__ inline void ProcessGComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *,
        uint16_t, uint16_t, uint32_t, uint16_t, uint32_t);
    __simd_vf__ inline void PackFinalDaNz(
        __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t, uint16_t);

    __simd_vf__ inline void ProcessKBetaComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, uint16_t, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessKBetaComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessKBetaComputerVFTwoCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkbComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ betaType *,
        __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkbComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ betaType *,
        __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkbComputerVFTwoCol(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ betaType *,
        __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkbgComputerGatherDkVF(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *, __ubuf__ kType *,
        __ubuf__ betaType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ kType *,
        __ubuf__ betaType *, __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkbgComputerVF(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *, __ubuf__ kType *,
        __ubuf__ betaType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ betaType *,
        __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkbDkbgComputerVF(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ betaType *, __ubuf__ kType *,
        __ubuf__ betaType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ kType *,
        __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDvbComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ kType *,
        __ubuf__ betaType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDvbComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ kType *,
        __ubuf__ betaType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDvbComputerVFTwoCol(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ kType *,
        __ubuf__ betaType *, __ubuf__ betaType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkGatherComputerVFOneLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkGatherComputerVFMutiLineOneCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessDkGatherComputerVFTwoCol(
        __ubuf__ kType *, __ubuf__ kType *, __ubuf__ kType *, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessKKTComputerVF(
        __ubuf__ kType *, __ubuf__ betaType *, __ubuf__ kType *, __ubuf__ float *,
        __ubuf__ float *, uint32_t, uint16_t, uint16_t);
    __simd_vf__ inline void ProcessKKTComputerVFSub(
        __ubuf__ betaType *, __ubuf__ betaType *, __ubuf__ float *, __ubuf__ float *, uint16_t);
};

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKBetaGComputerVFOneLineOneCol(
    __ubuf__ kType* kBetaGOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg;
    RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg;
    RegTensor<float> betaFP32Reg, gFP32Reg, betaGFP32Reg;
    RegTensor<kType> kBetaGOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<kType, false>(kInReg, kIn);

    HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
    HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
    Exp(gFP32Reg, gFP32Reg, maskFull32);
    Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

    CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
    MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
    CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
    StoreAlign(kBetaGOut, kBetaGOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKBetaGComputerVFMutiLineOneCol(
    __ubuf__ kType* kBetaGOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<betaType> gInReg, gInReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1;
    RegTensor<float> betaFP32Reg, betaFP32Reg1;
    RegTensor<float> gFP32Reg, gFP32Reg1;
    RegTensor<float> betaGFP32Reg, betaGFP32Reg1;
    RegTensor<kType> kBetaGOutReg, kBetaGOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(gInReg1, gIn + 1);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaFP32Reg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg1, gInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(gInReg1, gIn + (mIdx + 1) * PRELOAD_NUM + 1);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Exp(gFP32Reg1, gFP32Reg1, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg1, betaFP32Reg1, gFP32Reg1, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaGFP32Reg1, betaGFP32Reg1, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), kBetaGOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaFP32Reg1, betaInReg1, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg1, gInReg1, maskFull16, maskFull32);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Exp(gFP32Reg1, gFP32Reg1, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg1, betaFP32Reg1, gFP32Reg1, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);

        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaGFP32Reg1, betaGFP32Reg1, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), kBetaGOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<betaType, true>(gInReg, gIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
            HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
            Exp(gFP32Reg, gFP32Reg, maskFull32);
            Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
            CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKBetaGComputerVFTwoCol(
    __ubuf__ kType* kBetaGOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1;
    RegTensor<float> betaFP32Reg, gFP32Reg, betaGFP32Reg;
    RegTensor<kType> kBetaGOutReg, kBetaGOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1));
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * 2));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * 2 + 1));

        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaGFP32Reg, betaGFP32Reg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * 2), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * 2 + 1), kBetaGOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaGFP32Reg, betaGFP32Reg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * 2), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * 2 + 1), kBetaGOutReg1, maskFull16);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKBetaAndKBetaGComputerVFOneLineOneCol(
    __ubuf__ kType* kBetaGOut, __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg;
    RegTensor<float> kBetaFP32ZeroReg, kBetaFP32OneReg;
    RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg;
    RegTensor<float> betaFP32Reg, gFP32Reg, betaGFP32Reg;
    RegTensor<kType> kBetaOutReg, kBetaGOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<kType, false>(kInReg, kIn);

    HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
    HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
    Exp(gFP32Reg, gFP32Reg, maskFull32);
    Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

    CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
    MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                   betaFP32Reg, betaFP32Reg, maskFull32);
    MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                   betaGFP32Reg, betaGFP32Reg, maskFull32);
    CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
    CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
    StoreAlign(kBetaOut, kBetaOutReg, maskFull16);
    StoreAlign(kBetaGOut, kBetaGOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKBetaAndKBetaGComputerVFMutiLineOneCol(
    __ubuf__ kType* kBetaGOut, __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize,
    uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<betaType> gInReg, gInReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> kBetaFP32ZeroReg, kBetaFP32OneReg, kBetaFP32ZeroReg1, kBetaFP32OneReg1;
    RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1;
    RegTensor<float> betaFP32Reg, betaFP32Reg1;
    RegTensor<float> gFP32Reg, gFP32Reg1;
    RegTensor<float> betaGFP32Reg, betaGFP32Reg1;
    RegTensor<kType> kBetaOutReg, kBetaOutReg1;
    RegTensor<kType> kBetaGOutReg, kBetaGOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(gInReg1, gIn + 1);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaFP32Reg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg1, gInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(gInReg1, gIn + (mIdx + 1) * PRELOAD_NUM + 1);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Exp(gFP32Reg1, gFP32Reg1, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg1, betaFP32Reg1, gFP32Reg1, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                       betaFP32Reg, betaFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1,
                       betaFP32Reg1, betaFP32Reg1, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                       betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1,
                       betaGFP32Reg1, betaGFP32Reg1, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), kBetaOutReg1, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), kBetaGOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaFP32Reg1, betaInReg1, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg1, gInReg1, maskFull16, maskFull32);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Exp(gFP32Reg1, gFP32Reg1, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg1, betaFP32Reg1, gFP32Reg1, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);

        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                       betaFP32Reg, betaFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1,
                       betaFP32Reg1, betaFP32Reg1, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                       betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1,
                       betaGFP32Reg1, betaGFP32Reg1, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), kBetaOutReg1, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), kBetaGOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<betaType, true>(gInReg, gIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
            HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
            Exp(gFP32Reg, gFP32Reg, maskFull32);
            Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                           betaFP32Reg, betaFP32Reg, maskFull32);
            MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                           betaGFP32Reg, betaGFP32Reg, maskFull32);
            CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
            CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaOutReg, maskFull16);
            StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessVBetaComputerVFOneLineOneCol(
    __ubuf__ kType* vBetaOut, __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    RegTensor<kType> vInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg;
    RegTensor<float> vBetaFP32ZeroReg, vBetaFP32OneReg;
    RegTensor<float> betaBrcbFP32Reg;
    RegTensor<kType> vBetaOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(vInReg, vIn);

    HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
    CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
    MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
    CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
    StoreAlign(vBetaOut, vBetaOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessVBetaComputerVFMutiLineOneCol(
    __ubuf__ kType* vBetaOut, __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> vInReg, vInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg1, vFP32OneReg1;
    RegTensor<float> vBetaFP32ZeroReg, vBetaFP32OneReg, vBetaFP32ZeroReg1, vBetaFP32OneReg1;
    RegTensor<float> betaBrcbFP32Reg, betaBrcbFP32Reg1;
    RegTensor<kType> vBetaOutReg, vBetaOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(vInReg1, vIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        LoadIn<kType, false>(vInReg, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(vInReg1, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, betaBrcbFP32Reg1, betaBrcbFP32Reg1, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), vBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), vBetaOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg1, betaInReg1, maskFull16, maskFull32);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, betaBrcbFP32Reg1, betaBrcbFP32Reg1, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), vBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), vBetaOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(vInReg, vIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
            CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
            MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
            CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), vBetaOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessVBetaComputerVFTwoCol(
    __ubuf__ kType* vBetaOut, __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<kType> vInReg, vInReg1;
    RegTensor<betaType> betaInReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg1, vFP32OneReg1;
    RegTensor<float> vBetaFP32ZeroReg, vBetaFP32OneReg, vBetaFP32ZeroReg1, vBetaFP32OneReg1;
    RegTensor<float> betaBrcbFP32Reg;
    RegTensor<kType> vBetaOutReg, vBetaOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(vInReg1, vIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        LoadIn<kType, false>(vInReg, vIn + oneEleNum * ((mIdx + 1) * 2));
        LoadIn<kType, false>(vInReg1, vIn + oneEleNum * ((mIdx + 1) * 2 + 1));
        MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * 2), vBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * 2 + 1), vBetaOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * 2), vBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * 2 + 1), vBetaOutReg1, maskFull16);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessMDuDwComputerVFOneLineOneCol(
    __ubuf__ kType* mduwOut, __ubuf__ kType* mduIn, __ubuf__ kType* mdwIn,
    uint16_t mSize, uint16_t nSize, uint32_t startRow)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    RegTensor<kType> mduInReg, mdwInReg;
    RegTensor<float> mduFP32ZeroReg, mduFP32OneReg;
    RegTensor<float> mdwFP32ZeroReg, mdwFP32OneReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg, zeroReg, rowIdxReg;
    RegTensor<kType> mduwOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskStore;
    Duplicate(zeroReg, static_cast<float>(0), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    LoadIn<kType, false>(mduInReg, mduIn);
    LoadIn<kType, false>(mdwInReg, mdwIn);
    Duplicate(rowIdxReg, static_cast<float>(startRow));
    CastHalf2Float<kType>(mduFP32ZeroReg, mduFP32OneReg, mduInReg, maskFull16);
    CastHalf2Float<kType>(mdwFP32ZeroReg, mdwFP32OneReg, mdwInReg, maskFull16);
    AddFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, mduFP32ZeroReg, mduFP32OneReg, mdwFP32ZeroReg, mdwFP32OneReg, maskFull32);

    CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
    SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);

    CastFloat2Half<kType>(mduwOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
    uint32_t storeLen = oneEleNum;
    maskStore = UpdateMask<kType>(storeLen);
    StoreAlign(mduwOut, mduwOutReg, maskStore);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessMDuDwComputerVFMutiLineOneCol(
    __ubuf__ kType* mduwOut, __ubuf__ kType* mduIn, __ubuf__ kType* mdwIn,
    uint16_t mSize, uint16_t nSize, uint32_t startRow, uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> mduInReg, mduInReg1;
    RegTensor<kType> mdwInReg, mdwInReg1;
    RegTensor<float> mduFP32ZeroReg, mduFP32OneReg, mduFP32ZeroReg1, mduFP32OneReg1;
    RegTensor<float> mdwFP32ZeroReg, mdwFP32OneReg, mdwFP32ZeroReg1, mdwFP32OneReg1;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg, resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, rowIdxReg;
    RegTensor<kType> mduwOutReg, mduwOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskZeroSelect1, maskOneSelect1;
    MaskReg maskStore;
    Duplicate(zeroReg, static_cast<float>(0), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    LoadIn<kType, false>(mduInReg, mduIn);
    LoadIn<kType, false>(mduInReg1, mduIn + oneEleNum);
    LoadIn<kType, false>(mdwInReg, mdwIn);
    LoadIn<kType, false>(mdwInReg1, mdwIn + oneEleNum);

    Duplicate(rowIdxReg, static_cast<float>(startRow));
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        CastHalf2Float<kType>(mduFP32ZeroReg, mduFP32OneReg, mduInReg, maskFull16);
        CastHalf2Float<kType>(mduFP32ZeroReg1, mduFP32OneReg1, mduInReg1, maskFull16);
        LoadIn<kType, false>(mduInReg, mduIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(mduInReg1, mduIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(mdwFP32ZeroReg, mdwFP32OneReg, mdwInReg, maskFull16);
        CastHalf2Float<kType>(mdwFP32ZeroReg1, mdwFP32OneReg1, mdwInReg1, maskFull16);
        LoadIn<kType, false>(mdwInReg, mdwIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(mdwInReg1, mdwIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        AddFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, mduFP32ZeroReg, mduFP32OneReg, mdwFP32ZeroReg, mdwFP32OneReg, maskFull32);
        AddFloatTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, mduFP32ZeroReg1, mduFP32OneReg1, mdwFP32ZeroReg1, mdwFP32OneReg1, maskFull32);

        CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect1, maskOneSelect1, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, zeroReg, resultFP32ZeroReg1, resultFP32OneReg1, maskZeroSelect1, maskOneSelect1);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CastFloat2Half<kType>(mduwOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        CastFloat2Half<kType>(mduwOutReg1, resultFP32ZeroReg1, resultFP32OneReg1, maskFull32);
        uint32_t storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM), mduwOutReg, maskStore);
        storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), mduwOutReg1, maskStore);
    }
    uint16_t mIdx = mLoopCnt;
    {
        CastHalf2Float<kType>(mduFP32ZeroReg, mduFP32OneReg, mduInReg, maskFull16);
        CastHalf2Float<kType>(mduFP32ZeroReg1, mduFP32OneReg1, mduInReg1, maskFull16);
        CastHalf2Float<kType>(mdwFP32ZeroReg, mdwFP32OneReg, mdwInReg, maskFull16);
        CastHalf2Float<kType>(mdwFP32ZeroReg1, mdwFP32OneReg1, mdwInReg1, maskFull16);

        AddFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, mduFP32ZeroReg, mduFP32OneReg, mdwFP32ZeroReg, mdwFP32OneReg, maskFull32);
        AddFloatTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, mduFP32ZeroReg1, mduFP32OneReg1, mdwFP32ZeroReg1, mdwFP32OneReg1, maskFull32);

        CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect1, maskOneSelect1, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, zeroReg, resultFP32ZeroReg1, resultFP32OneReg1, maskZeroSelect1, maskOneSelect1);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CastFloat2Half<kType>(mduwOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        CastFloat2Half<kType>(mduwOutReg1, resultFP32ZeroReg1, resultFP32OneReg1, maskFull32);
        uint32_t storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM), mduwOutReg, maskStore);
        storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), mduwOutReg1, maskStore);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<kType, false>(mduInReg, mduIn + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(mdwInReg, mdwIn + oneEleNum * (mIdx * PRELOAD_NUM));
            CastHalf2Float<kType>(mduFP32ZeroReg, mduFP32OneReg, mduInReg, maskFull16);
            CastHalf2Float<kType>(mdwFP32ZeroReg, mdwFP32OneReg, mdwInReg, maskFull16);
            AddFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, mduFP32ZeroReg, mduFP32OneReg, mdwFP32ZeroReg, mdwFP32OneReg, maskFull32);

            CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
            SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);

            CastFloat2Half<kType>(mduwOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
            uint32_t storeLen = oneEleNum;
            maskStore = UpdateMask<kType>(storeLen);
            StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM), mduwOutReg, maskStore);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessGComputerVFOneLineOneCol(
    __ubuf__ kType* dAOut, __ubuf__ kType* dA6In, __ubuf__ betaType* gIn, __ubuf__ betaType* gAllIn,
    uint16_t mSize, uint16_t nSize, uint32_t startRow, uint32_t calcColSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> dA6InReg;
    RegTensor<betaType> gInReg, gAllInReg;
    RegTensor<float> dA6FP32ZeroReg, dA6FP32OneReg;
    RegTensor<float> gFP32ZeroReg;
    RegTensor<float> gFactorZeroReg, gFactorOneReg;
    RegTensor<float> gAllFP32ZeroReg, gAllFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;
    RegTensor<float> zeroReg, negOneReg, rowIdxReg;
    RegTensor<kType> dAOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskStore;
    MaskReg castDA6MaskCount16;
    uint32_t castDA6Count = calcColSize;
    castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);

    Duplicate(zeroReg, static_cast<float>(0), maskFull32);
    Duplicate(negOneReg, static_cast<float>(-1), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    if constexpr (!std::is_same<betaType, float>()) {
        LoadAlign(gAllInReg, gAllIn);
        Cast<float, betaType, ctHalf2Fp32Zero>(gAllFP32ZeroReg, gAllInReg, maskFull16);
        Cast<float, betaType, ctHalf2Fp32One>(gAllFP32OneReg, gAllInReg, maskFull16);
    } else {
        LoadAlign<betaType, LoadDist::DIST_DINTLV_B32>(gAllFP32ZeroReg, gAllFP32OneReg, gAllIn);
    }
    LoadIn<kType, false>(dA6InReg, dA6In);
    LoadIn<betaType, true>(gInReg, gIn);

    Duplicate(rowIdxReg, static_cast<float>(startRow));
    CastHalf2Float<kType>(dA6FP32ZeroReg, dA6FP32OneReg, dA6InReg, castDA6MaskCount16);
    HalfOrFloat2Float<betaType>(gFP32ZeroReg, gInReg, maskFull16, maskFull32);

    SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg, gFP32ZeroReg, maskFull32);
    MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, float(0.0), maskFull32);
    ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);

    // result = -dA6 * gAll
    MulFloatTwoReg(dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, negOneReg, negOneReg, maskFull32);
    MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, gFactorZeroReg, gFactorOneReg, maskFull32);

    CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
    SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
    Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

    CastFloat2Half<kType>(dAOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
    uint32_t storeLen = oneEleNum;
    maskStore = UpdateMask<kType>(storeLen);
    StoreAlign((__ubuf__ kType*&)dAOut, dAOutReg, maskStore);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessGComputerVFMutiLineOneCol(
    __ubuf__ kType* dAOut, __ubuf__ kType* dA6In, __ubuf__ betaType* gIn, __ubuf__ betaType* gAllIn,
    uint16_t mSize, uint16_t nSize, uint32_t startRow, uint16_t lastLoopCnt, uint32_t calcColSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> dA6InReg, dA6InReg1;
    RegTensor<betaType> gInReg, gInReg1, gAllInReg;
    RegTensor<float> dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg1, dA6FP32OneReg1;
    RegTensor<float> gFP32ZeroReg, gFP32ZeroReg1;
    RegTensor<float> gFactorZeroReg, gFactorOneReg, gFactorZeroReg1, gFactorOneReg1;
    RegTensor<float> gAllFP32ZeroReg, gAllFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg, resultFP32ZeroReg1, resultFP32OneReg1;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;
    RegTensor<float> zeroReg, negOneReg, rowIdxReg;
    RegTensor<kType> dAOutReg, dAOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskZeroSelect1, maskOneSelect1;
    MaskReg maskStore;
    MaskReg castDA6MaskCount16;

    Duplicate(zeroReg, static_cast<float>(0), maskFull32);
    Duplicate(negOneReg, static_cast<float>(-1), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    if constexpr (!std::is_same<betaType, float>()) {
        LoadAlign(gAllInReg, gAllIn);
        Cast<float, betaType, ctHalf2Fp32Zero>(gAllFP32ZeroReg, gAllInReg, maskFull16);
        Cast<float, betaType, ctHalf2Fp32One>(gAllFP32OneReg, gAllInReg, maskFull16);
    } else {
        LoadAlign<betaType, LoadDist::DIST_DINTLV_B32>(gAllFP32ZeroReg, gAllFP32OneReg, gAllIn);
    }
    LoadIn<kType, false>(dA6InReg, dA6In);
    LoadIn<kType, false>(dA6InReg1, dA6In + oneEleNum);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(gInReg1, gIn + 1);

    Duplicate(rowIdxReg, static_cast<float>(startRow));
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        uint32_t castDA6Count = calcColSize;
        castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
        CastHalf2Float<kType>(dA6FP32ZeroReg, dA6FP32OneReg, dA6InReg, castDA6MaskCount16);
        castDA6Count = calcColSize;
        castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
        CastHalf2Float<kType>(dA6FP32ZeroReg1, dA6FP32OneReg1, dA6InReg1, castDA6MaskCount16);
        LoadIn<kType, false>(dA6InReg, dA6In + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dA6InReg1, dA6In + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        HalfOrFloat2Float<betaType>(gFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32ZeroReg1, gInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(gInReg1, gIn + (mIdx + 1) * PRELOAD_NUM + 1);

        SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg, gFP32ZeroReg, maskFull32);
        SubFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg1, gFP32ZeroReg1, maskFull32);
        MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, float(0.0), maskFull32);
        MinsFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gFactorZeroReg1, gFactorOneReg1, float(0.0), maskFull32);
        ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
        ExpFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gFactorZeroReg1, gFactorOneReg1, maskFull32);

        // result = -dA6 * gAll
        MulFloatTwoReg(dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, negOneReg, negOneReg, maskFull32);
        MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
        MulFloatTwoReg(dA6FP32ZeroReg1, dA6FP32OneReg1, dA6FP32ZeroReg1, dA6FP32OneReg1, negOneReg, negOneReg, maskFull32);
        MulFloatTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, dA6FP32ZeroReg1, dA6FP32OneReg1, gFactorZeroReg1, gFactorOneReg1, maskFull32);

        CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect1, maskOneSelect1, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, zeroReg, resultFP32ZeroReg1, resultFP32OneReg1, maskZeroSelect1, maskOneSelect1);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CastFloat2Half<kType>(dAOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dAOutReg1, resultFP32ZeroReg1, resultFP32OneReg1, maskFull32);

        uint32_t storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM), dAOutReg, maskStore);
        storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dAOutReg1, maskStore);
    }
    uint16_t mIdx = mLoopCnt;
    {
        uint32_t castDA6Count = calcColSize;
        castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
        CastHalf2Float<kType>(dA6FP32ZeroReg, dA6FP32OneReg, dA6InReg, castDA6MaskCount16);
        castDA6Count = calcColSize;
        castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
        CastHalf2Float<kType>(dA6FP32ZeroReg1, dA6FP32OneReg1, dA6InReg1, castDA6MaskCount16);
        HalfOrFloat2Float<betaType>(gFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32ZeroReg1, gInReg1, maskFull16, maskFull32);

        SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg, gFP32ZeroReg, maskFull32);
        SubFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg1, gFP32ZeroReg1, maskFull32);
        MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, float(0.0), maskFull32);
        MinsFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gFactorZeroReg1, gFactorOneReg1, float(0.0), maskFull32);
        ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
        ExpFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gFactorZeroReg1, gFactorOneReg1, maskFull32);

        // result = -dA6 * gAll
        MulFloatTwoReg(dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, negOneReg, negOneReg, maskFull32);
        MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
        MulFloatTwoReg(dA6FP32ZeroReg1, dA6FP32OneReg1, dA6FP32ZeroReg1, dA6FP32OneReg1, negOneReg, negOneReg, maskFull32);
        MulFloatTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, dA6FP32ZeroReg1, dA6FP32OneReg1, gFactorZeroReg1, gFactorOneReg1, maskFull32);

        CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect1, maskOneSelect1, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, zeroReg, resultFP32ZeroReg1, resultFP32OneReg1, maskZeroSelect1, maskOneSelect1);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CastFloat2Half<kType>(dAOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dAOutReg1, resultFP32ZeroReg1, resultFP32OneReg1, maskFull32);
        uint32_t storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM), dAOutReg, maskStore);
        storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dAOutReg1, maskStore);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<kType, false>(dA6InReg, dA6In + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<betaType, true>(gInReg, gIn + mIdx * PRELOAD_NUM);

            uint32_t castDA6Count = calcColSize;
            castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
            CastHalf2Float<kType>(dA6FP32ZeroReg, dA6FP32OneReg, dA6InReg, castDA6MaskCount16);
            HalfOrFloat2Float<betaType>(gFP32ZeroReg, gInReg, maskFull16, maskFull32);

            SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg, gFP32ZeroReg, maskFull32);
            MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, float(0.0), maskFull32);
            ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);

            // result = -dA6 * gAll
            MulFloatTwoReg(dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, negOneReg, negOneReg, maskFull32);
            MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, gFactorZeroReg, gFactorOneReg, maskFull32);

            CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
            SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
            Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

            CastFloat2Half<kType>(dAOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
            uint32_t storeLen = oneEleNum;
            maskStore = UpdateMask<kType>(storeLen);
            StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM), dAOutReg, maskStore);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::PackFinalDaNz(
    __ubuf__ kType* dst, __ubuf__ kType* src, uint16_t rows, uint16_t cols, uint16_t srcStride)
{
    RegTensor<kType> dataReg;
    uint32_t dataCount = cols;
    MaskReg dataMask = UpdateMask<kType>(dataCount);
    uint16_t alignedRows = (rows + 15U) & ~15U;
    __ubuf__ kType* dstPtr = dst;

    for (uint16_t row = 0; row < rows; ++row) {
        LoadAlign(dataReg, src + row * srcStride);
        StoreAlign<kType, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            dstPtr, dataReg, alignedRows, 1U, dataMask);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKKTComputerVF(
    __ubuf__ kType* kktIn, __ubuf__ betaType* betaIn, __ubuf__ kType* daIn,
    __ubuf__ float* reducesum1, __ubuf__ float* reducesum0,
    uint32_t calcColSize, uint16_t mSize, uint16_t nSize)
{
    RegTensor<betaType> betaInReg;
    RegTensor<kType> kktInReg;
    RegTensor<kType> daInReg;
    RegTensor<float> betaFP32ZeroReg, betaFP32OneReg, kktFP32ZeroReg, kktFP32OneReg, daFP32ZeroReg, daFP32OneReg;
    RegTensor<float> daaFP32ZeroReg, daaFP32OneReg, daaFP32AddReg, reduceSumReg, Sum1DaaLineFP32Reg;
    RegTensor<float> dgFP32AddZeroReg, dgFP32AddOneReg;
    RegTensor<float> reducesum0ZeroReg, reducesum0OneReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    uint32_t calcNNum = calcColSize;
    UnalignRegForStore uStoreReducesum;

    if constexpr (!std::is_same<betaType, float>()) {
        LoadAlign(betaInReg, betaIn);
        Cast<float, betaType, ctHalf2Fp32Zero>(betaFP32ZeroReg, betaInReg, maskFull16);
        Cast<float, betaType, ctHalf2Fp32One>(betaFP32OneReg, betaInReg, maskFull16);
    } else {
        LoadAlign<betaType, LoadDist::DIST_DINTLV_B32>(betaFP32ZeroReg, betaFP32OneReg, betaIn);
    }
    Duplicate(dgFP32AddZeroReg, static_cast<float>(0), maskFull32);
    Duplicate(dgFP32AddOneReg, static_cast<float>(0), maskFull32);
    uint32_t nextEleOffset = 0;
    // 先copy第一块数据
    LoadAlign(kktInReg, kktIn);
    LoadAlign(daInReg, daIn);
    // 做前mSize-1行的exe并copy下一个loop需要的数据
    for (uint16_t mIdx = 0; mIdx < mSize - 1; mIdx++) {
        Duplicate(reduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t castKktCountZero = calcNNum;
        uint32_t castKktCountOne = calcNNum;
        MaskReg castKktMaskCountZero16, castKktMaskCountOne16, castDaMaskCountZero16, castDaMaskCountOne16;
        castKktMaskCountZero16 = UpdateMask<half>(castKktCountZero);
        castKktMaskCountOne16 = UpdateMask<half>(castKktCountOne);
        uint32_t castDaCountZero = calcNNum;
        uint32_t castDaCountOne = calcNNum;
        castDaMaskCountZero16 = UpdateMask<half>(castDaCountZero);
        castDaMaskCountOne16 = UpdateMask<half>(castDaCountOne);
        Cast<float, kType, ctHalf2Fp32Zero>(kktFP32ZeroReg, kktInReg, castKktMaskCountZero16);
        Cast<float, kType, ctHalf2Fp32One>(kktFP32OneReg, kktInReg, castKktMaskCountOne16);
        Cast<float, kType, ctHalf2Fp32Zero>(daFP32ZeroReg, daInReg, castDaMaskCountZero16);
        Cast<float, kType, ctHalf2Fp32One>(daFP32OneReg, daInReg, castDaMaskCountOne16);
        uint32_t nextEleOffset = (mIdx + 1) * nSize;
        LoadAlign(kktInReg, kktIn + nextEleOffset);
        LoadAlign(daInReg, daIn + nextEleOffset);
        Mul(kktFP32ZeroReg, kktFP32ZeroReg, betaFP32ZeroReg, maskFull32);
        Mul(kktFP32OneReg, kktFP32OneReg, betaFP32OneReg, maskFull32);
        Mul(daaFP32ZeroReg, daFP32ZeroReg, kktFP32ZeroReg, maskFull32);
        Mul(daaFP32OneReg, daFP32OneReg, kktFP32OneReg, maskFull32);
        // 每行相加
        Add(daaFP32AddReg, daaFP32ZeroReg, daaFP32OneReg, maskFull32);
        Add(reduceSumReg, reduceSumReg, daaFP32AddReg, maskFull32);
        // 每列相加
        Add(dgFP32AddZeroReg, dgFP32AddZeroReg, daaFP32ZeroReg, maskFull32);
        Add(dgFP32AddOneReg, dgFP32AddOneReg, daaFP32OneReg, maskFull32);
        ReduceSum(Sum1DaaLineFP32Reg, reduceSumReg, maskFull32);
        auto actReducesum1 = reducesum1 + mIdx;
        StoreUnAlign(actReducesum1, Sum1DaaLineFP32Reg, uStoreReducesum, 1);
        StoreUnAlignPost(actReducesum1, uStoreReducesum, 0);
    }
    // 最后一行数据exe
    uint16_t mIdx = mSize - 1;
    {
        Duplicate(reduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t castKktCountZero = calcNNum;
        uint32_t castKktCountOne = calcNNum;
        MaskReg castKktMaskCountZero16, castKktMaskCountOne16, castDaMaskCountZero16, castDaMaskCountOne16;
        castKktMaskCountZero16 = UpdateMask<half>(castKktCountZero);
        castKktMaskCountOne16 = UpdateMask<half>(castKktCountOne);
        uint32_t castDaCountZero = calcNNum;
        uint32_t castDaCountOne = calcNNum;
        castDaMaskCountZero16 = UpdateMask<half>(castDaCountZero);
        castDaMaskCountOne16 = UpdateMask<half>(castDaCountOne);
        Cast<float, kType, ctHalf2Fp32Zero>(kktFP32ZeroReg, kktInReg, castKktMaskCountZero16);
        Cast<float, kType, ctHalf2Fp32One>(kktFP32OneReg, kktInReg, castKktMaskCountOne16);
        Cast<float, kType, ctHalf2Fp32Zero>(daFP32ZeroReg, daInReg, castDaMaskCountZero16);
        Cast<float, kType, ctHalf2Fp32One>(daFP32OneReg, daInReg, castDaMaskCountOne16);
        Mul(kktFP32ZeroReg, kktFP32ZeroReg, betaFP32ZeroReg, maskFull32);
        Mul(kktFP32OneReg, kktFP32OneReg, betaFP32OneReg, maskFull32);
        Mul(daaFP32ZeroReg, daFP32ZeroReg, kktFP32ZeroReg, maskFull32);
        Mul(daaFP32OneReg, daFP32OneReg, kktFP32OneReg, maskFull32);
        // 每行相加
        Add(daaFP32AddReg, daaFP32ZeroReg, daaFP32OneReg, maskFull32);
        Add(reduceSumReg, reduceSumReg, daaFP32AddReg, maskFull32);
        // 每列相加
        Add(dgFP32AddZeroReg, dgFP32AddZeroReg, daaFP32ZeroReg, maskFull32);
        Add(dgFP32AddOneReg, dgFP32AddOneReg, daaFP32OneReg, maskFull32);
        ReduceSum(Sum1DaaLineFP32Reg, reduceSumReg, maskFull32);
        auto actReducesum1 = reducesum1 + mIdx;
        StoreUnAlign(actReducesum1, Sum1DaaLineFP32Reg, uStoreReducesum, 1);
        StoreUnAlignPost(actReducesum1, uStoreReducesum, 0);
    }
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(reducesum0ZeroReg, reducesum0OneReg, reducesum0);
    Add(reducesum0ZeroReg, reducesum0ZeroReg, dgFP32AddZeroReg, maskFull32);
    Add(reducesum0OneReg, reducesum0OneReg, dgFP32AddOneReg, maskFull32);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(reducesum0, reducesum0ZeroReg, reducesum0OneReg, maskFull32);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKKTComputerVFSub(
    __ubuf__ betaType* dgOut, __ubuf__ betaType* dgIn, __ubuf__ float* reducesum1, __ubuf__ float* reducesum0,
    uint16_t calcSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(betaType);
    uint16_t nLoopCnt = (calcSize + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<betaType> dgInReg;
    RegTensor<float> sum1DaaLineZeroFP32Reg, sum1DaaLineOneFP32Reg, dgFP32AddZeroReg, dgFP32AddOneReg;
    RegTensor<float> dgFP32ZeroReg, dgFP32OneReg;
    RegTensor<betaType> dgOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
        uint32_t eleOffset = vfBlockIdx * eleNumPerVf;
        if constexpr (!std::is_same<betaType, float>()) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(sum1DaaLineZeroFP32Reg, sum1DaaLineOneFP32Reg, reducesum1 + eleOffset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dgFP32AddZeroReg, dgFP32AddOneReg, reducesum0 + eleOffset);
            LoadAlign(dgInReg, dgIn + eleOffset);
            Sub(dgFP32AddZeroReg, dgFP32AddZeroReg, sum1DaaLineZeroFP32Reg, maskFull32);
            Sub(dgFP32AddOneReg, dgFP32AddOneReg, sum1DaaLineOneFP32Reg, maskFull32);

            Cast<float, betaType, ctHalf2Fp32Zero>(dgFP32ZeroReg, dgInReg, maskFull16);
            Cast<float, betaType, ctHalf2Fp32One>(dgFP32OneReg, dgInReg, maskFull16);
            Add(dgFP32AddZeroReg, dgFP32AddZeroReg, dgFP32ZeroReg, maskFull32);
            Add(dgFP32AddOneReg, dgFP32AddOneReg, dgFP32OneReg, maskFull32);

            Cast<betaType, float, ctFp322HalfOne>(dgOutReg, dgFP32AddOneReg, maskFull32);
            Cast<betaType, float, ctFp322HalfZero>(dgOutReg, dgFP32AddZeroReg, maskFull32);
            UnalignRegForStore uStoreDg;
            auto actDgOut = dgOut + eleOffset;
            uint32_t copyNum = min(eleNumPerVf, static_cast<uint32_t>(calcSize) - eleOffset);
            StoreUnAlign(actDgOut, dgOutReg, uStoreDg, copyNum);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        } else {
            LoadAlign(sum1DaaLineZeroFP32Reg, reducesum1 + eleOffset);
            LoadAlign(dgFP32AddZeroReg, reducesum0 + eleOffset);
            LoadAlign(dgFP32ZeroReg, dgIn + eleOffset);
            Sub(dgFP32AddZeroReg, dgFP32AddZeroReg, sum1DaaLineZeroFP32Reg, maskFull32);
            Add(dgFP32AddZeroReg, dgFP32AddZeroReg, dgFP32ZeroReg, maskFull32);

            UnalignRegForStore uStoreDg;
            auto actDgOut = dgOut + eleOffset;
            uint32_t copyNum = min(eleNumPerVf, static_cast<uint32_t>(calcSize) - eleOffset);
            StoreUnAlign(actDgOut, dgFP32AddZeroReg, uStoreDg, copyNum);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDvbComputerVFMutiLineOneCol(
    __ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<kType> vInReg, vInReg1;
    RegTensor<kType> dvbInReg, dvbInReg1;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg1;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg1, vFP32OneReg1;
    RegTensor<float> dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg1, dvbFP32OneReg1;
    RegTensor<betaType> dbetaInReg;
    RegTensor<float> dbetaFP32Reg;
    RegTensor<float> dBetaFP32Reg;
    RegTensor<kType> dvOutReg, dvOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(vInReg1, vIn + oneEleNum);
    LoadIn<kType, false>(dvbInReg, dvbIn);
    LoadIn<kType, false>(dvbInReg1, dvbIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        LoadIn<kType, false>(vInReg, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(vInReg1, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbInReg1, maskFull16);
        LoadIn<kType, false>(dvbInReg, dvbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dvbInReg1, dvbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

        MulFloatTwoReg(vFP32ZeroReg1, vFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);

        CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dvOutReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dvOutReg1, maskFull16);

        // Row 0: dBeta = reduceSum(v * dvb) + dbeta
        Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);

        // Row 1
        Add(vFP32ZeroReg1, vFP32ZeroReg1, vFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg1, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM + 1);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM + 1, dBetaFP32Reg, maskFull32, uStore, 1);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbInReg1, maskFull16);

        MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(vFP32ZeroReg1, vFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);

        CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dvOutReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dvOutReg1, maskFull16);

        Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);

        Add(vFP32ZeroReg1, vFP32ZeroReg1, vFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg1, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM + 1);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM + 1, dBetaFP32Reg, maskFull32, uStore, 1);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(vInReg, vIn + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dvbInReg, dvbIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
            CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
            CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
            MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
            MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);

            Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
            ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
            LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM);
            HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
            Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
            StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDvbComputerVFOneLineOneCol(
    __ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn, uint16_t mSize, uint16_t nSize)
{
    RegTensor<betaType> betaInReg;
    RegTensor<kType> vInReg;
    RegTensor<kType> dvbInReg;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg;
    RegTensor<float> dvbFP32ZeroReg, dvbFP32OneReg;
    RegTensor<betaType> dbetaInReg;
    RegTensor<float> dbetaFP32Reg;
    RegTensor<float> dBetaFP32Reg;
    RegTensor<kType> dvOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(dvbInReg, dvbIn);
    HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
    CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
    CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
    MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
    MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
    CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
    StoreAlign((__ubuf__ kType*&)dvOut, dvOutReg, maskFull16);

    Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
    ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
    LoadIn<betaType, true>(dbetaInReg, dbetaIn);
    HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
    Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
    StoreUnAlignOut<betaType>(dBetaOut, dBetaFP32Reg, maskFull32, uStore, 1);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDvbComputerVFTwoCol(
    __ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<betaType> betaInReg;
    RegTensor<kType> vInReg, vInReg1;
    RegTensor<kType> dvbInReg, dvbInReg1;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg1, vFP32OneReg1;
    RegTensor<float> dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg1, dvbFP32OneReg1;
    RegTensor<betaType> dbetaInReg;
    RegTensor<float> dbetaFP32Reg;
    RegTensor<float> dBetaFP32Reg, dBetaFP32Reg1;
    RegTensor<kType> dvOutReg, dvOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(vInReg1, vIn + oneEleNum);
    LoadIn<kType, false>(dvbInReg, dvbIn);
    LoadIn<kType, false>(dvbInReg1, dvbIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + mIdx + 1);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        LoadIn<kType, false>(vInReg, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(vInReg1, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbInReg1, maskFull16);
        LoadIn<kType, false>(dvbInReg, dvbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dvbInReg1, dvbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(vFP32ZeroReg1, vFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

        CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dvOutReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dvOutReg1, maskFull16);

        Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);

        Add(vFP32ZeroReg1, vFP32ZeroReg1, vFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg1, vFP32ZeroReg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dBetaFP32Reg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaFP32Reg, maskFull32, uStore, 1);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbInReg1, maskFull16);

        MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(vFP32ZeroReg1, vFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

        CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dvOutReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dvOutReg1, maskFull16);

        Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);

        Add(vFP32ZeroReg1, vFP32ZeroReg1, vFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg1, vFP32ZeroReg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dBetaFP32Reg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaFP32Reg, maskFull32, uStore, 1);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkbgComputerGatherDkVF(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ betaType* dgOut,
    __ubuf__ kType* kIn, __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkGatherIn, __ubuf__ betaType* dBetaIn, __ubuf__ kType* dkbgIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<kType> dkInReg, dkGatherInReg;
    RegTensor<betaType> dbetaInReg;
    RegTensor<kType> dkbgInReg;
    RegTensor<float> betaBrcbFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, dkGatherFP32ZeroReg, dkGatherFP32OneReg, dgFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, dBetaAddZeroReg, dkbgFP32ZeroReg, dkbgFP32OneReg;
    RegTensor<float> gBrcbFP32ZeroReg, dbetaFP32ZeroReg, dbetaFP32OneReg;
    RegTensor<float> kReduceSumReg, dkReduceSumReg;
    RegTensor<kType> dkOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStoreDBeta;
    UnalignRegForStore uStoreDg;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(dbetaInReg, dBetaIn);
    LoadIn<kType>(kInReg, kIn);
    LoadIn<kType>(dkbgInReg, dkbgIn);
    LoadIn<kType>(dkInReg, dkIn);
    LoadIn<kType>(dkGatherInReg, dkGatherIn);
    // 前mSize - 1行
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(dbetaFP32ZeroReg, dbetaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1));
        LoadIn<betaType, true>(dbetaInReg, dBetaIn + (mIdx + 1));

        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);
            LoadIn<kType>(kInReg, kIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            LoadIn<kType>(dkbgInReg, dkbgIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            LoadIn<kType>(dkInReg, dkIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            CastHalf2Float<kType>(dkGatherFP32ZeroReg, dkGatherFP32OneReg, dkGatherInReg, maskFull16);
            LoadIn<kType>(dkGatherInReg, dkGatherIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkGatherFP32ZeroReg, dkGatherFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaAddZeroReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgFP32ZeroReg, maskFull32, uStoreDg, 1);
    }
    // 最后一行
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(dbetaFP32ZeroReg, dbetaInReg, maskFull16, maskFull32);

        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        // 最后一行的前nLoopCnt - 1
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt - 1; vfBlockIdx++) {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);
            LoadIn<kType>(kInReg, kIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            LoadIn<kType>(dkbgInReg, dkbgIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            LoadIn<kType>(dkInReg, dkIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            CastHalf2Float<kType>(dkGatherFP32ZeroReg, dkGatherFP32OneReg, dkGatherInReg, maskFull16);
            LoadIn<kType>(dkGatherInReg, dkGatherIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkGatherFP32ZeroReg, dkGatherFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        // 最后一行最后一个loop
        uint16_t vfBlockIdx = nLoopCnt - 1;
        {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            CastHalf2Float<kType>(dkGatherFP32ZeroReg, dkGatherFP32OneReg, dkGatherInReg, maskFull16);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkGatherFP32ZeroReg, dkGatherFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaAddZeroReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgFP32ZeroReg, maskFull32, uStoreDg, 1);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkbgComputerVF(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ betaType* dgOut,
    __ubuf__ kType* kIn, __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
    __ubuf__ kType* dkIn, __ubuf__ betaType* dBetaIn, __ubuf__ kType* dkbgIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<kType> dkInReg;
    RegTensor<betaType> dbetaInReg;
    RegTensor<kType> dkbgInReg;
    RegTensor<float> betaBrcbFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, dgFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, dBetaAddZeroReg, dkbgFP32ZeroReg, dkbgFP32OneReg;
    RegTensor<float> gBrcbFP32ZeroReg, dbetaFP32ZeroReg, dbetaFP32OneReg;
    RegTensor<float> kReduceSumReg, dkReduceSumReg;
    RegTensor<kType> dkOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStoreDBeta;
    UnalignRegForStore uStoreDg;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(dbetaInReg, dBetaIn);
    LoadIn<kType>(kInReg, kIn);
    LoadIn<kType>(dkbgInReg, dkbgIn);
    LoadIn<kType>(dkInReg, dkIn);
    // 前mSize - 1行
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(dbetaFP32ZeroReg, dbetaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1));
        LoadIn<betaType, true>(dbetaInReg, dBetaIn + (mIdx + 1));

        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);
            LoadIn<kType>(kInReg, kIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            LoadIn<kType>(dkbgInReg, dkbgIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            LoadIn<kType>(dkInReg, dkIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaAddZeroReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgFP32ZeroReg, maskFull32, uStoreDg, 1);
    }
    // 最后一行
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(dbetaFP32ZeroReg, dbetaInReg, maskFull16, maskFull32);

        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        // 最后一行的前nLoopCnt - 1
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt - 1; vfBlockIdx++) {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);
            LoadIn<kType>(kInReg, kIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            LoadIn<kType>(dkbgInReg, dkbgIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            LoadIn<kType>(dkInReg, dkIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        // 最后一行最后一个loop
        uint16_t vfBlockIdx = nLoopCnt - 1;
        {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaAddZeroReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgFP32ZeroReg, maskFull32, uStoreDg, 1);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkbDkbgComputerVF(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ betaType* dgOut,
    __ubuf__ kType* kIn, __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn, __ubuf__ kType* dkbgIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<kType> kInReg;
    RegTensor<kType> dkInReg;
    RegTensor<kType> dkbInReg;
    RegTensor<kType> dkbgInReg;
    RegTensor<float> betaBrcbFP32ZeroReg, gBrcbFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg;
    RegTensor<float> dkFP32ZeroReg, dkFP32OneReg;
    RegTensor<float> dkbFP32ZeroReg, dkbFP32OneReg;
    RegTensor<float> dkbgFP32ZeroReg, dkbgFP32OneReg;
    RegTensor<float> mulFP32ZeroReg, mulFP32OneReg;
    RegTensor<float> dBetaReduceReg, dgReduceReg, dBetaOutReg, dgOutReg;
    RegTensor<kType> dkOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStoreDBeta;
    UnalignRegForStore uStoreDg;

    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        LoadIn<betaType, true>(betaInReg, betaIn + mIdx);
        LoadIn<betaType, true>(gInReg, gIn + mIdx);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(dBetaReduceReg, static_cast<float>(0), maskFull32);
        Duplicate(dgReduceReg, static_cast<float>(0), maskFull32);

        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            uint32_t offset = mIdx * nSize + vfBlockIdx * eleKNumPerVf;
            LoadIn<kType>(kInReg, kIn + offset);
            LoadIn<kType>(dkInReg, dkIn + offset);
            LoadIn<kType>(dkbInReg, dkbIn + offset);
            LoadIn<kType>(dkbgInReg, dkbgIn + offset);
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);

            MulFloatTwoReg(mulFP32ZeroReg, mulFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                           dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
            Add(mulFP32ZeroReg, mulFP32ZeroReg, mulFP32OneReg, maskFull32);
            Add(dBetaReduceReg, dBetaReduceReg, mulFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg,
                           betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg,
                           dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg,
                           gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(mulFP32ZeroReg, mulFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                           dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            Add(mulFP32ZeroReg, mulFP32ZeroReg, mulFP32OneReg, maskFull32);
            Add(dBetaReduceReg, dBetaReduceReg, mulFP32ZeroReg, maskFull32);
            Mul(mulFP32ZeroReg, mulFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(dgReduceReg, dgReduceReg, mulFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg,
                           betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg,
                           dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + offset, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaOutReg, dBetaReduceReg, maskFull32);
        ReduceSum(dgOutReg, dgReduceReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaOutReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgOutReg, maskFull32, uStoreDg, 1);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkGatherComputerVFOneLineOneCol(
    __ubuf__ kType* dkOut, __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
    uint16_t mSize, uint16_t nSize)
{
    RegTensor<kType> dk1Reg, dk2Reg;
    RegTensor<float> dk1FP32ZeroReg, dk1FP32OneReg;
    RegTensor<float> dk2FP32ZeroReg, dk2FP32OneReg;
    RegTensor<kType> dkOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<kType, false>(dk1Reg, dkIn1);
    LoadIn<kType, false>(dk2Reg, dkIn2);
    CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
    CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
    AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
    CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
    StoreAlign((__ubuf__ kType*&)dkOut, dkOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkGatherComputerVFMutiLineOneCol(
    __ubuf__ kType* dkOut, __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<kType> dk1Reg, dk1Reg1;
    RegTensor<kType> dk2Reg, dk2Reg1;
    RegTensor<float> dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg1, dk1FP32OneReg1;
    RegTensor<float> dk2FP32ZeroReg, dk2FP32OneReg, dk2FP32ZeroReg1, dk2FP32OneReg1;
    RegTensor<kType> dkOutReg, dkOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<kType, false>(dk1Reg, dkIn1);
    LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum);
    LoadIn<kType, false>(dk2Reg, dkIn2);
    LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
        CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
        LoadIn<kType, false>(dk1Reg, dkIn1 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);
        LoadIn<kType, false>(dk2Reg, dkIn2 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
        AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
        CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);

        AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
        AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<kType, false>(dk1Reg, dkIn1 + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dk2Reg, dkIn2 + oneEleNum * (mIdx * PRELOAD_NUM));
            CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
            CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
            AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkGatherComputerVFTwoCol(
    __ubuf__ kType* dkOut, __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<kType> dk1Reg, dk1Reg1;
    RegTensor<kType> dk2Reg, dk2Reg1;
    RegTensor<float> dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg1, dk1FP32OneReg1;
    RegTensor<float> dk2FP32ZeroReg, dk2FP32OneReg, dk2FP32ZeroReg1, dk2FP32OneReg1;
    RegTensor<kType> dkOutReg, dkOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<kType, false>(dk1Reg, dkIn1);
    LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum);
    LoadIn<kType, false>(dk2Reg, dkIn2);
    LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
        CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
        LoadIn<kType, false>(dk1Reg, dkIn1 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);
        LoadIn<kType, false>(dk2Reg, dkIn2 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
        AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
        CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);

        AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
        AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<kType, false>(dk1Reg, dkIn1 + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum * (mIdx * PRELOAD_NUM + 1));
            LoadIn<kType, false>(dk2Reg, dkIn2 + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum * (mIdx * PRELOAD_NUM + 1));
            CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
            CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
            CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
            CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);
            AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
            AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
            StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkbComputerVFMutiLineOneCol(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<kType> dkInReg, dkInReg1;
    RegTensor<kType> dkbInReg, dkbInReg1;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg1, dkFP32OneReg1;
    RegTensor<float> dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg1, dkbFP32OneReg1;
    RegTensor<kType> dkOutReg, dkOutReg1;
    RegTensor<float> dBetaFP32Reg;
    RegTensor<betaType> dBetaOutZeroReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);
    LoadIn<kType, false>(dkInReg, dkIn);
    LoadIn<kType, false>(dkInReg1, dkIn + oneEleNum);
    LoadIn<kType, false>(dkbInReg, dkbIn);
    LoadIn<kType, false>(dkbInReg1, dkbIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg1, dkFP32OneReg1, dkInReg1, maskFull16);
        LoadIn<kType, false>(dkInReg, dkIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dkInReg1, dkIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbInReg1, maskFull16);
        LoadIn<kType, false>(dkbInReg, dkbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dkbInReg1, dkbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

        MulFloatTwoReg(kFP32ZeroReg1, kFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg1, dkFP32OneReg1, dkFP32ZeroReg1, dkFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dkFP32ZeroReg1, dkFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);

        Add(kFP32ZeroReg1, kFP32ZeroReg1, kFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg1, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM + 1, dBetaFP32Reg, maskFull32, uStore, 1);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg1, dkFP32OneReg1, dkInReg1, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbInReg1, maskFull16);

        MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

        MulFloatTwoReg(kFP32ZeroReg1, kFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg1, dkFP32OneReg1, dkFP32ZeroReg1, dkFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dkFP32ZeroReg1, dkFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);

        Add(kFP32ZeroReg1, kFP32ZeroReg1, kFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg1, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM + 1, dBetaFP32Reg, maskFull32, uStore, 1);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dkInReg, dkIn + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dkbInReg, dkbIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
            MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);

            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);
            StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkbComputerVFOneLineOneCol(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn, uint16_t mSize, uint16_t nSize)
{
    RegTensor<betaType> betaInReg;
    RegTensor<kType> kInReg;
    RegTensor<kType> dkInReg;
    RegTensor<kType> dkbInReg;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg;
    RegTensor<float> dkFP32ZeroReg, dkFP32OneReg;
    RegTensor<float> dkbFP32ZeroReg, dkbFP32OneReg;
    RegTensor<kType> dkOutReg;
    RegTensor<float> dBetaFP32Reg;
    RegTensor<betaType> dBetaOutZeroReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(dkInReg, dkIn);
    LoadIn<kType, false>(dkbInReg, dkbIn);
    HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
    CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
    CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
    CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
    MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
    MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
    AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
    CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
    StoreAlign((__ubuf__ kType*&)dkOut, dkOutReg, maskFull16);

    Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
    ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);
    StoreUnAlignOut<betaType>(dBetaOut, dBetaFP32Reg, maskFull32, uStore, 1);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessDkbComputerVFTwoCol(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<betaType> betaInReg;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<kType> dkInReg, dkInReg1;
    RegTensor<kType> dkbInReg, dkbInReg1;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg1, dkFP32OneReg1;
    RegTensor<float> dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg1, dkbFP32OneReg1;
    RegTensor<kType> dkOutReg, dkOutReg1;
    RegTensor<float> dBetaFP32Reg, dBetaFP32Reg1;
    RegTensor<betaType> dBetaOutZeroReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);
    LoadIn<kType, false>(dkInReg, dkIn);
    LoadIn<kType, false>(dkInReg1, dkIn + oneEleNum);
    LoadIn<kType, false>(dkbInReg, dkbIn);
    LoadIn<kType, false>(dkbInReg1, dkbIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + mIdx + 1);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg1, dkFP32OneReg1, dkInReg1, maskFull16);
        LoadIn<kType, false>(dkInReg, dkIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dkInReg1, dkIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbInReg1, maskFull16);
        LoadIn<kType, false>(dkbInReg, dkbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dkbInReg1, dkbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

        MulFloatTwoReg(kFP32ZeroReg1, kFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg1, dkFP32OneReg1, dkFP32ZeroReg1, dkFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dkFP32ZeroReg1, dkFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);

        Add(kFP32ZeroReg1, kFP32ZeroReg1, kFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg1, kFP32ZeroReg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dBetaFP32Reg1, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaFP32Reg, maskFull32, uStore, 1);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg1, dkFP32OneReg1, dkInReg1, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbInReg1, maskFull16);

        MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

        MulFloatTwoReg(kFP32ZeroReg1, kFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg1, dkFP32OneReg1, dkFP32ZeroReg1, dkFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dkFP32ZeroReg1, dkFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);

        Add(kFP32ZeroReg1, kFP32ZeroReg1, kFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg1, kFP32ZeroReg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dBetaFP32Reg1, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaFP32Reg, maskFull32, uStore, 1);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKBetaComputerVFMutiLineOneCol(
    __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    uint16_t mLoopCnt = mSize / 2 - 1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kBetaFP32ZeroReg, kBetaFP32OneReg;
    RegTensor<float> kFP32ZeroReg1, kFP32OneReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg1;
    RegTensor<kType> kBetaOutReg, kBetaOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);
    // 前mSize - 3行
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * 2);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * 2 + 1);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * 2));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * 2 + 1));
        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2 + 1), kBetaOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    // 最后三行
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);

        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2 + 1), kBetaOutReg1, maskFull16);

        mIdx += 1;
        // 最后一行
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * 2);
            LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx * 2));
            HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKBetaComputerVFOneLineOneCol(
    __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kBetaFP32ZeroReg, kBetaFP32OneReg;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<kType> kBetaOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(kInReg, kIn);

    HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
    CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
    MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
    CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
    StoreAlign(kBetaOut, kBetaOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdRegBase<kType, betaType>::ProcessKBetaComputerVFTwoCol(
    __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    uint32_t oneEleNum = min(eleKNumPerVf, static_cast<uint32_t>(nSize));
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kBetaFP32ZeroReg, kBetaFP32OneReg;
    RegTensor<float> kFP32ZeroReg1, kFP32OneReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg1;
    RegTensor<kType> kBetaOutReg, kBetaOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);
    for (uint16_t mIdx = 0; mIdx < mSize - 1; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx + 1) * 2);
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * (mIdx + 1) * 2 + 1);
        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2 + 1), kBetaOutReg1, maskFull16);
    }
    uint16_t mIdx = mSize - 1;
    // 最后一行
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2 + 1), kBetaOutReg1, maskFull16);
    }
}

#endif // PREPARE_WY_REPR_BWD_ARCH35_REGBASE_H

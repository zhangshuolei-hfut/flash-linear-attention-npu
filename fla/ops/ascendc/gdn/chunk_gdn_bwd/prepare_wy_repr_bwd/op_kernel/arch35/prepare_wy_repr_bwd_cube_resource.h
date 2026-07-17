/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PREPARE_WY_REPR_BWD_A5_CUBE_RESOURCE_H
#define PREPARE_WY_REPR_BWD_A5_CUBE_RESOURCE_H

#include "kernel_operator.h"

namespace GDN::A5Pipeline {

template <AscendC::TPosition POSITION, uint32_t SIZE>
class RawLocalBuffer {
public:
    __aicore__ inline RawLocalBuffer()
    {
        tensor_ = AscendC::LocalTensor<uint8_t>(POSITION, 0, SIZE);
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset) const
    {
        return tensor_[offset].template ReinterpretCast<T>();
    }

private:
    AscendC::LocalTensor<uint8_t> tensor_;
};

struct CubeResource {
    RawLocalBuffer<AscendC::TPosition::A1, 512U * 1024U> l1Buf;
    RawLocalBuffer<AscendC::TPosition::A2, 64U * 1024U> l0ABuf;
    RawLocalBuffer<AscendC::TPosition::B2, 64U * 1024U> l0BBuf;
    RawLocalBuffer<AscendC::TPosition::CO1, 256U * 1024U> l0CBuf;
    RawLocalBuffer<AscendC::TPosition::VECCALC, 248U * 1024U> ubBuf;
};

} // namespace GDN::A5Pipeline

#endif // PREPARE_WY_REPR_BWD_A5_CUBE_RESOURCE_H

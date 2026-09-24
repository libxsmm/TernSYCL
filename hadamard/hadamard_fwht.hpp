// SYCL port of TernOCL's hadamard_fwht.cl: fused sign flip + blockwise (1024)
// normalised Walsh-Hadamard transform
//   y = H_1024 (s * x) / 32   per 1024-wide block along K of a [rows, K] DT tensor.
// Input pre-transform of rotated-basis ternary checkpoints (Bonsai 2). One
// work-group of 128 items per block: the 10 radix-2 stages run as three radix-8
// passes through SLM and a final radix-2 pass that stores; fp32 butterflies,
// DT only at load/store. Same stage order as the OpenCL and xetla-plugin SYCL
// kernels, so results match them bit for bit.
//   launch: nd_range<1>(rows * K / 1024 * 128, 128)
#pragma once

#include "xe2.hpp"

namespace hadamard {

namespace syclex = sycl::ext::oneapi::experimental;

inline void wht8(float *v) {
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
}

template <bool BF16>
struct Fwht1024 {
    const unsigned short *x;
    const signed char *signs;  // [K], or nullptr
    unsigned short *y;
    int K;

    void operator()(sycl::nd_item<1> it) const {
        using D = xe2::dt<BF16>;
        float *slm = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(it.get_group());
        const int lid = (int)it.get_local_id(0);
        const size_t blk = it.get_group(0);
        const size_t bpr = K / 1024;
        const size_t off = (blk / bpr) * K + (blk % bpr) * 1024;
        const unsigned short *xb = x + off;
        unsigned short *yb = y + off;
        const signed char *sb = signs ? signs + (blk % bpr) * 1024 : nullptr;

        float v[8];
        int base = lid * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = D::tof(xb[base + j]);
            if (sb) v[j] *= (float)sb[base + j];
        }
        wht8(v);
#pragma unroll
        for (int j = 0; j < 8; ++j) slm[base + j] = v[j];
        xe2::barrier_local();

        base = (lid & 7) | ((lid >> 3) << 6);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = slm[base + (j << 3)];
        wht8(v);
#pragma unroll
        for (int j = 0; j < 8; ++j) slm[base + (j << 3)] = v[j];
        xe2::barrier_local();

        base = (lid & 63) | ((lid >> 6) << 9);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = slm[base + (j << 6)];
        wht8(v);
#pragma unroll
        for (int j = 0; j < 8; ++j) slm[base + (j << 6)] = v[j];
        xe2::barrier_local();

        const float scale = 1.0f / 32.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = lid + j * 128;
            const float a = slm[i], b = slm[i + 512];
            yb[i] = D::from((a + b) * scale);
            yb[i + 512] = D::from((a - b) * scale);
        }
    }

    auto get(syclex::properties_tag) const { return syclex::properties{syclex::work_group_size<128>}; }
};

}  // namespace hadamard

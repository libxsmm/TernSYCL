// SYCL port of TernOCL's int2_fp16_upcvt.cl, itself a port of the xetla int2 x
// fp16/bf16 "upcvt" GEMM/GEMV
// (xetla/include/experimental/group/gemm/impl/int2_fp16_upcvt_xmx_xe.hpp):
//
//   C[m,n] = DT( epi( sum_k A[m,k] * code(B[k,n]) * S[k/128, n] ) )
//
//   A : DT [M, K] row-major
//   B : uint32 [K/16, N], word (kp, n) holds the 2-bit codes of K rows
//       16*kp .. 16*kp+15 of column n (code of row 16*kp+j at bits 2j..2j+1)
//   S : DT [K/128, N]
//   C : DT (or fp32) [M, N]
//
// DT is fp16 or bf16 (template parameter BF16). Codes are {0, 1, 3} =
// {0, +1, -1}. As in xetla, the scale is folded into the upconvert with
// integer ops only -- (scale ^ sign) & magnitude -- so the B tile comes out in
// DT and goes straight into the DT DPAS, fp32 accumulate.
// Requires N % 16 == 0 and K % 128 == 0.
#pragma once

#include "epilogue_dev.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace upcvt {

using namespace xe2;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

constexpr int GS = 128;

// Two consecutive K codes (one nibble of the word, low bits of x) -> one VNNI
// dword holding the two DT weights, each 0, +scale or -scale (sign = bit 15).
inline int dq_pair(unsigned x, unsigned s2) {
    const unsigned mask = ((unsigned)(((int)(x << 31)) >> 31) & 0x0000FFFFu)
                        | ((unsigned)(((int)(x << 29)) >> 31) & 0xFFFF0000u);
    const unsigned sign = ((x & 2u) << 14) | ((x & 8u) << 28);
    return (int)((s2 ^ sign) & mask);
}

inline int8 dq_word(unsigned w, unsigned s2) {
    int8 b;
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i] = dq_pair(w >> (4 * i), s2);
    return b;
}

template <int SGM> struct rows;
template <> struct rows<1> { using a_t = short; using acc_t = float; };
template <> struct rows<2> { using a_t = short2; using acc_t = float2; };
template <> struct rows<4> { using a_t = short4; using acc_t = float4; };
template <> struct rows<8> { using a_t = short8; using acc_t = float8; };

template <int SGM, class V> inline float el(const V &v, int r) {
    if constexpr (SGM == 1) return v;
    else return v[r];
}

// ---------------------------------------------------------------------------
// GEMV / small M. A sub-group owns 16 columns (one per lane) and SGM rows and
// walks its K slice in steps of 128 (one scale group). Per step: 8 packed B
// rows (8 single-row 2D block reads, so the first DPAS waits for 64 B), one
// scale row, SGM A rows, 8 DPAS of K = 16.
//   SGM    rows per sub-group (DPAS repeat count: 1, 2, 4, 8)
//   NSG_N  sub-groups along N per work-group (WG covers 16*NSG_N columns)
//   LS     local k-slicing: sub-groups per column block splitting K, reduced
//          through SLM
//   U      k-steps whose loads are issued together before any compute
template <bool BF16, int SGM, int NSG_N, int LS, int U>
struct Gemv {
    const unsigned short *A;
    const unsigned *B;
    const unsigned short *S;
    void *C;
    Epi epi;
    int M, N, K;

    using a_t = typename rows<SGM>::a_t;
    using acc_t = typename rows<SGM>::acc_t;
    static constexpr int WG = 16 * NSG_N * LS;

    static acc_t step(acc_t acc, const unsigned *w, unsigned sc, const ushort8 *ar) {
        const unsigned s2 = sc | (sc << 16);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            a_t a;
            if constexpr (SGM == 1) a = (short)ar[0][c];
            else
#pragma unroll
                for (int r = 0; r < SGM; ++r) a[r] = (short)ar[r][c];
            acc = mad_k16<BF16>(a, dq_word(w[c], s2), acc);
        }
        return acc;
    }

    void load(const surf &sb, int m0, int n0, int s, unsigned *w, unsigned &sc, ushort8 *ar) const {
#pragma unroll
        for (int r = 0; r < 8; ++r) w[r] = rd_32b_1r16(sb, n0, s * 8 + r);
        sc = intel_sub_group_block_read_us(gptr(S + (size_t)s * N + n0));
#pragma unroll
        for (int r = 0; r < SGM; ++r)
            ar[r] = (SGM == 1 || m0 + r < M)
                    ? intel_sub_group_block_read_us8(gptr(A + (size_t)(m0 + r) * K + s * GS))
                    : ushort8{};
    }

    void operator()(sycl::nd_item<2> it) const {
        const auto sgp = it.get_sub_group();
        const int lane = sgp.get_local_linear_id();
        const int sg = sgp.get_group_linear_id();
        const int sgn = sg % NSG_N, sgk = sg / NSG_N;
        const int n0 = ((int)it.get_group(1) * NSG_N + sgn) * 16;
        const int m0 = (int)it.get_group(0) * SGM;

        const int nsteps = K / GS;
        const int per = (nsteps + LS - 1) / LS;
        const int s_begin = sgk * per;
        const int s_end = sycl::min(nsteps, s_begin + per);
        const surf sb(B, N * 4, K / 16, N * 4);

        acc_t acc = 0.0f;
        if (n0 < N) {
            int s = s_begin;
            for (; s + U <= s_end; s += U) {
                unsigned w[U][8], sc[U];
                ushort8 ar[U][SGM];
#pragma unroll
                for (int u = 0; u < U; ++u) load(sb, m0, n0, s + u, w[u], sc[u], ar[u]);
#pragma unroll
                for (int u = 0; u < U; ++u) acc = step(acc, w[u], sc[u], ar[u]);
            }
            for (; s < s_end; ++s) {
                unsigned w[8], sc;
                ushort8 ar[SGM];
                load(sb, m0, n0, s, w, sc, ar);
                acc = step(acc, w, sc, ar);
            }
        }

        if constexpr (LS > 1) {
            float *red = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[(LS - 1) * NSG_N * SGM * 16]>(
                    it.get_group());
            if (sgk > 0) {
                float *dst = red + (((sgk - 1) * NSG_N + sgn) * SGM) * 16;
#pragma unroll
                for (int r = 0; r < SGM; ++r) dst[r * 16 + lane] = el<SGM>(acc, r);
            }
            barrier_local();
            if (sgk > 0) return;
            for (int j = 0; j < LS - 1; ++j) {
                const float *src = red + ((j * NSG_N + sgn) * SGM) * 16;
                if constexpr (SGM == 1) acc += src[lane];
                else
#pragma unroll
                    for (int r = 0; r < SGM; ++r) acc[r] += src[r * 16 + lane];
            }
        }

        if (n0 >= N) return;
#pragma unroll
        for (int r = 0; r < SGM; ++r)
            if (m0 + r < M)
                epi_dev<BF16>::store(C, epi, (size_t)(m0 + r) * N + n0 + lane, n0 + lane, el<SGM>(acc, r));
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<16>, syclex::work_group_size<1, WG>};
    }
};

// ---------------------------------------------------------------------------
// Large-M GEMM. A sub-group computes an MT_M x MT_N tile (MT_M % 8 == 0,
// MT_N % 16 == 0): per K16 it dequantizes B once per 16-column block and
// reuses it for all MT_M/8 DPAS row blocks. A work-group is WG_M x WG_N
// sub-groups. A/B/C go through 2D block I/O, which zero-fills reads and clips
// writes outside the matrix, so any M works. 256 GRF.
template <bool BF16, int MT_M, int MT_N, int WG_M, int WG_N>
struct GemmMT {
    const unsigned short *A;
    const unsigned *B;
    const unsigned short *S;
    void *C;
    Epi epi;
    int M, N, K;

    static constexpr int MB = MT_M / 8, NB = MT_N / 16, WG = 16 * WG_M * WG_N;

    void operator()(sycl::nd_item<2> it) const {
        const int sg = it.get_sub_group().get_group_linear_id();
        const int m0 = ((int)it.get_group(0) * WG_M + sg / WG_N) * MT_M;
        const int n0 = ((int)it.get_group(1) * WG_N + sg % WG_N) * MT_N;
        const surf sa(A, K * 2, M, K * 2), sb(B, N * 4, K / 16, N * 4);

        float8 acc[MB][NB];
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
            for (int j = 0; j < NB; ++j) acc[i][j] = 0.0f;

        for (int s = 0; s < K / GS; ++s) {
            uint8 w[NB];
            unsigned s2[NB];
#pragma unroll
            for (int j = 0; j < NB; ++j) {
                w[j] = rd_32b_8r16(sb, n0 + 16 * j, s * 8);
                const unsigned sc = (n0 + 16 * j < N)
                        ? intel_sub_group_block_read_us(gptr(S + (size_t)s * N + n0 + 16 * j)) : 0u;
                s2[j] = sc | (sc << 16);
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                short8 a[MB];
#pragma unroll
                for (int i = 0; i < MB; ++i)
                    a[i] = __builtin_bit_cast(short8, rd_16b_8r16(sa, s * GS + 16 * c, m0 + 8 * i));
#pragma unroll
                for (int j = 0; j < NB; ++j) {
                    const int8 b = dq_word(w[j][c], s2[j]);
#pragma unroll
                    for (int i = 0; i < MB; ++i) acc[i][j] = mad_k16<BF16>(a[i], b, acc[i][j]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
            for (int j = 0; j < NB; ++j) {
                // barrier: otherwise IGC may fold the fp16 conversion into the K
                // loop's accumulator chain and spill (seen in OpenCL at MT_M*MT_N >= 2048)
                float8 v = acc[i][j];
#ifdef __SYCL_DEVICE_ONLY__
                __asm__ volatile("" : "+rw"(v));
#endif
                epi_dev<BF16>::store8x16(C, epi, M, N, m0 + 8 * i, n0 + 16 * j, v);
            }
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<16>, syclex::work_group_size<1, WG>,
                intelex::grf_size<256>};
    }
};

}  // namespace upcvt

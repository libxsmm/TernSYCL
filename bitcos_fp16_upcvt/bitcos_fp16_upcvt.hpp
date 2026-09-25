// SYCL port of TernOCL's bitcos_fp16_upcvt.cl, itself a port of the xetla
// BITCOS ternary GEMV (xetla/include/experimental/group/gemm/impl/
// bitcos_fp16_upcvt_xmx_xe.hpp as the vLLM plugin builds it: BITCOS_FP16_LUT,
// BITCOS_SIGN_GATHER4/3):
//
//   C[m,n] = DT( epi( sum_k A[m,k] * code(k,n) * S[k/128, n] ) ),  code in {-1, 0, +1}
//
//   A : DT [M, K] row-major, S : DT [K/128, N], C : DT (or fp32) [M, N]
//   B : uint32 buffer, three planes back to back (common/bitcos.hpp):
//       bitmap  [K/32][N]  bit c of word (kp, n) = weight(kp*32+c, n) != 0
//       offsets [N]        first sign word of column n
//       signs   bitstream  one bit per non-zero, column-major in k, 1 -> -1
//   SR: slice_ranks [LS-1][N], non-zeros of column n above slice boundary s*K/LS
//
// Unpack, per lane (= output column) and 64-k step, as xetla:
//   - one 2D block read gives the two 32-row bitmap words;
//   - the column's rank (its cursor into the sign stream) advances by one
//     popcount per 32-row block, and one 3-dword gather at word rank/32 covers
//     the sign windows of both blocks (a block consumes at most 32 bits);
//   - each block is decoded as 8 four-row groups: the presence nibble and the
//     next four sign bits index a 256-entry SLM table of four DT codes {0, +1,
//     -1} in VNNI order, the window shifts by popcount(nibble), and the codes
//     are scaled (Apply below);
//   - two DPAS of k = 16 per block, fp32 accumulate.
// Requires N % 16 == 0 and K % (64 * LS) == 0 (GEMV), K % 64 == 0 (M-tiled).
#pragma once

#include "epilogue_dev.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace bitcos {

using namespace xe2;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

constexpr int GS = 128, STEP = 64;

// Scale apply on the looked-up codes (two dwords = four VNNI DT codes per lane):
//   VISA : xetla's two SIMD32 hf multiplies in place, as inline vISA (fp16)
//   INT  : table of 0 / 0x7fff / 0xffff masks, one AND with (scale | 0x8000)
//          gives 0 / +s / -s exactly (bf16, which has no 16-bit multiply;
//          bit-identical to VISA for fp16)
//   SIMT : plain half4 * half (fp16), which IGC de- and re-interleaves
enum Apply { VISA = 0, INT = 1, SIMT = 2 };

typedef _Float16 hf2 __attribute__((ext_vector_type(2)));
typedef _Float16 hf4 __attribute__((ext_vector_type(4)));

template <int SGM> struct rows;
template <> struct rows<1> { using a_t = short; using acc_t = float; };
template <> struct rows<2> { using a_t = short2; using acc_t = float2; };
template <> struct rows<4> { using a_t = short4; using acc_t = float4; };
template <> struct rows<8> { using a_t = short8; using acc_t = float8; };

template <int SGM, class V> inline float el(const V &v, int r) {
    if constexpr (SGM == 1) return v;
    else return v[r];
}

// Entry (m, s), m = presence nibble of rows 4g..4g+3, s = next four compact
// sign bits: row i is 0 if absent, else the sign bit of rank sum(m_j, j < i).
template <int AP> inline void build_lut(uint2 *lut, int lid, int nthreads) {
    constexpr unsigned POS = AP == INT ? 0x7fffu : 0x3c00u, NEG = AP == INT ? 0xffffu : 0xbc00u;
    for (int idx = lid; idx < 256; idx += nthreads) {
        const unsigned m = (unsigned)idx >> 4, s = (unsigned)idx & 15u;
        unsigned c[4];
        unsigned r = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned p = (m >> i) & 1u;
            c[i] = p ? (((s >> r) & 1u) ? NEG : POS) : 0u;
            r += p;
        }
        lut[idx] = uint2{c[0] | (c[1] << 16), c[2] | (c[3] << 16)};
    }
    barrier_local();
}

// Sign window at bit `rank` from the gathered words q (q.x holds word w0 =
// rank0/32 of this step; rank/32 - w0 is 0 or 1).
inline unsigned sign_window(uint3 q, unsigned w0, unsigned rank) {
    const unsigned m = (rank >> 5) - w0;
    const unsigned lo = m ? q.y : q.x, hi = m ? q.z : q.y;
    const unsigned sh = rank & 31u;
    return (lo >> sh) | ((hi << ((32u - sh) & 31u)) & (sh ? 0xffffffffu : 0u));
}

inline uint3 load_signs(const unsigned *p) {
    return __builtin_IB_lsc_load_global_uint3((const XE2_GLOBAL uint3 *)p, 0, 4);
}

// scale operand for the apply, from the lane's 16-bit scale
template <int AP> inline unsigned scale_op(unsigned sc) {
    if constexpr (AP == INT) return (sc | 0x8000u) * 0x10001u;
    else if constexpr (AP == VISA) return sc * 0x10001u;
    else return sc;
}

inline uint2 hmul2(uint2 e, unsigned s2) {
    uint2 r;
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl EH v_type=G type=hf num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl RH v_type=G type=hf num_elts=64 align=GRF alias=<%0,0>\n"
            ".decl SH v_type=G type=hf num_elts=32 align=GRF alias=<%2,0>\n"
            "mul (M1_NM, 32) RH(0,0)<1> EH(0,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) RH(1,0)<1> EH(1,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "}\n"
            : "=rw"(r) : "rw"(e), "rw"(s2));
#endif
    return r;
}

// One 32-row block: bitmap word bmp, sign window w -> two k16 VNNI B operands.
template <int AP>
inline void unpack_block(const uint2 *lut, unsigned bmp, unsigned w, unsigned sx, int8 &b0, int8 &b1) {
    unsigned d[16];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const unsigned m4 = (bmp >> (4 * g)) & 15u;
        const uint2 e = lut[(m4 << 4) | (w & 15u)];
        w >>= sycl::popcount(m4);
        if constexpr (AP == INT) {
            d[2 * g] = e.x & sx;
            d[2 * g + 1] = e.y & sx;
        } else if constexpr (AP == VISA) {
            const uint2 v = hmul2(e, sx);
            d[2 * g] = v.x;
            d[2 * g + 1] = v.y;
        } else {
            const hf4 v = __builtin_bit_cast(hf4, e) * __builtin_bit_cast(hf2, sx).x;
            const uint2 u = __builtin_bit_cast(uint2, v);
            d[2 * g] = u.x;
            d[2 * g + 1] = u.y;
        }
    }
    b0 = __builtin_bit_cast(int8, uint8{d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]});
    b1 = __builtin_bit_cast(int8, uint8{d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]});
}

// ---------------------------------------------------------------------------
// GEMV / decode (M = 1..8 per sub-group row block).
//   SGM    rows per sub-group (DPAS repeat count: 1, 2, 4, 8)
//   NSG_N  sub-groups along N per work-group (WG covers 16*NSG_N columns)
//   LS     local K slicing: sub-groups splitting K, reduced through SLM; slice
//          s enters the sign stream at SR[s-1]
template <bool BF16, int AP, int SGM, int NSG_N, int LS>
struct Gemv {
    const unsigned short *A;
    const unsigned *B;
    const unsigned short *S;
    const unsigned *SR;
    void *C;
    Epi epi;
    int M, N, K;

    using a_t = typename rows<SGM>::a_t;
    using acc_t = typename rows<SGM>::acc_t;
    static constexpr int WG = 16 * NSG_N * LS;

    void operator()(sycl::nd_item<2> it) const {
        uint2 *lut = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint2[256]>(it.get_group());
        build_lut<AP>(lut, (int)it.get_local_linear_id(), WG);

        const auto sgp = it.get_sub_group();
        const int lane = sgp.get_local_linear_id();
        const int sg = sgp.get_group_linear_id();
        const int sgn = sg % NSG_N, sgk = sg / NSG_N;
        const int n0 = ((int)it.get_group(1) * NSG_N + sgn) * 16;
        const int m0 = (int)it.get_group(0) * SGM;
        const int kslice = K / LS;
        const int k_begin = sgk * kslice, k_end = k_begin + kslice;

        const size_t bw = (size_t)(K / 32) * N;
        const unsigned *signs = B + bw + N;
        const surf sbm(B, N * 4, K / 32, N * 4);
        const unsigned off = n0 < N ? B[bw + n0 + lane] : 0u;
        unsigned rank = (LS > 1 && sgk > 0 && n0 < N) ? SR[(size_t)(sgk - 1) * N + n0 + lane] : 0u;

        acc_t acc = 0.0f;
        if (n0 < N) {
            // loop-invariant bases; per-step offsets stay 32-bit
            const unsigned short *sp = S + n0;
            const unsigned *sg_col = signs + off;
            const unsigned short *ap[SGM];
#pragma unroll
            for (int r = 0; r < SGM; ++r) ap[r] = A + (size_t)sycl::min(m0 + r, M - 1) * K;
            for (unsigned k = k_begin; k < (unsigned)k_end; k += STEP) {
                const uint2 bm = rd_32b_2r16(sbm, n0, k / 32);
                const unsigned sc = intel_sub_group_block_read_us(gptr(sp + (k / GS) * (unsigned)N));
                ushort4 ar[SGM];
#pragma unroll
                for (int r = 0; r < SGM; ++r)
                    ar[r] = (SGM == 1 || m0 + r < M) ? intel_sub_group_block_read_us4(gptr(ap[r] + k)) : ushort4{};
                const unsigned r0 = rank, r1 = r0 + sycl::popcount(bm.x);
                rank = r1 + sycl::popcount(bm.y);
                const unsigned w0 = r0 >> 5;
                const uint3 q = load_signs(sg_col + w0);
                const unsigned sx = scale_op<AP>(sc);
#pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    int8 b0, b1;
                    unpack_block<AP>(lut, ii ? bm.y : bm.x, sign_window(q, w0, ii ? r1 : r0), sx, b0, b1);
                    a_t a0, a1;
                    if constexpr (SGM == 1) {
                        a0 = (short)ar[0][2 * ii];
                        a1 = (short)ar[0][2 * ii + 1];
                    } else {
#pragma unroll
                        for (int r = 0; r < SGM; ++r) {
                            a0[r] = (short)ar[r][2 * ii];
                            a1[r] = (short)ar[r][2 * ii + 1];
                        }
                    }
                    acc = mad_k16<BF16>(a0, b0, acc);
                    acc = mad_k16<BF16>(a1, b1, acc);
                }
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
// M-tiled GEMM for prefill (any M) on the same single BITCOS copy. A sub-group
// computes an MT_M x MT_N tile (MT_M % 8 == 0, MT_N % 16 == 0): per 64-k step
// and 16-column block it unpacks the two 32-row blocks once and reuses the four
// k16 B operands for all MT_M/8 DPAS row blocks. A work-group is WG_M x WG_N
// sub-groups; no K slicing, so SR is unused. A/C through 2D block I/O
// (zero-filled reads, clipped writes). 256 GRF.
template <bool BF16, int AP, int MT_M, int MT_N, int WG_M, int WG_N>
struct GemmMT {
    const unsigned short *A;
    const unsigned *B;
    const unsigned short *S;
    const unsigned *SR;
    void *C;
    Epi epi;
    int M, N, K;

    static constexpr int MB = MT_M / 8, NB = MT_N / 16, WG = 16 * WG_M * WG_N;

    void operator()(sycl::nd_item<2> it) const {
        uint2 *lut = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint2[256]>(it.get_group());
        build_lut<AP>(lut, (int)it.get_local_linear_id(), WG);

        const int lane = it.get_sub_group().get_local_linear_id();
        const int sg = it.get_sub_group().get_group_linear_id();
        const int m0 = ((int)it.get_group(0) * WG_M + sg / WG_N) * MT_M;
        const int n0 = ((int)it.get_group(1) * WG_N + sg % WG_N) * MT_N;
        const size_t bw = (size_t)(K / 32) * N;
        const unsigned *signs = B + bw + N;
        const surf sa(A, K * 2, M, K * 2), sbm(B, N * 4, K / 32, N * 4);

        const unsigned *sg_col[NB];
        const unsigned short *sp[NB];
        unsigned rank[NB];
#pragma unroll
        for (int j = 0; j < NB; ++j) {
            sg_col[j] = signs + (n0 + 16 * j < N ? B[bw + n0 + 16 * j + lane] : 0u);
            sp[j] = S + n0 + 16 * j;
            rank[j] = 0u;
        }
        float8 acc[MB][NB];
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
            for (int j = 0; j < NB; ++j) acc[i][j] = 0.0f;

        for (unsigned k = 0; k < (unsigned)K; k += STEP) {
            short8 a[MB][4];
#pragma unroll
            for (int i = 0; i < MB; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    a[i][c] = __builtin_bit_cast(short8, rd_16b_8r16(sa, k + 16 * c, m0 + 8 * i));
#pragma unroll
            for (int j = 0; j < NB; ++j) {
                if (n0 + 16 * j >= N) continue;  // uniform across the sub-group
                const uint2 bm = rd_32b_2r16(sbm, n0 + 16 * j, k / 32);
                const unsigned sc = intel_sub_group_block_read_us(gptr(sp[j] + (k / GS) * (unsigned)N));
                const unsigned r0 = rank[j], r1 = r0 + sycl::popcount(bm.x);
                rank[j] = r1 + sycl::popcount(bm.y);
                const unsigned w0 = r0 >> 5;
                const uint3 q = load_signs(sg_col[j] + w0);
                const unsigned sx = scale_op<AP>(sc);
#pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    int8 b0, b1;
                    unpack_block<AP>(lut, ii ? bm.y : bm.x, sign_window(q, w0, ii ? r1 : r0), sx, b0, b1);
#pragma unroll
                    for (int i = 0; i < MB; ++i) {
                        acc[i][j] = mad_k16<BF16>(a[i][2 * ii], b0, acc[i][j]);
                        acc[i][j] = mad_k16<BF16>(a[i][2 * ii + 1], b1, acc[i][j]);
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
            for (int j = 0; j < NB; ++j) {
                // barrier: keeps IGC from folding the store conversion into the K loop (spills)
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

}  // namespace bitcos

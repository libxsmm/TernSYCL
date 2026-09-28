// Xe2 sub-group primitives for SIMT SYCL device code, without OpenCL builtins
// or IGC-internal intrinsics:
//   * DPAS, 2D block I/O, cached LSC loads: inline vISA (the instructions IGC
//     emits for the OpenCL builtins; operands are the SIMD16 per-lane vectors,
//     row r of a block / DPAS operand = vector component r)
//   * sub-group block reads: sycl_ext_oneapi_group_load_store (group_load)
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <cstdint>

namespace xe2 {

#define XE2_VEC(T, n, name) typedef T name __attribute__((ext_vector_type(n)))
XE2_VEC(short, 2, short2);
XE2_VEC(short, 4, short4);
XE2_VEC(short, 8, short8);
XE2_VEC(unsigned short, 2, ushort2);
XE2_VEC(unsigned short, 4, ushort4);
XE2_VEC(unsigned short, 8, ushort8);
XE2_VEC(unsigned short, 16, ushort16);
XE2_VEC(int, 2, int2);
XE2_VEC(int, 4, int4);
XE2_VEC(int, 8, int8);
XE2_VEC(unsigned, 2, uint2);
XE2_VEC(unsigned, 3, uint3);
XE2_VEC(unsigned, 4, uint4);
XE2_VEC(unsigned, 8, uint8);
XE2_VEC(float, 2, float2);
XE2_VEC(float, 4, float4);
XE2_VEC(float, 8, float8);
#undef XE2_VEC

// 2D surface: width and pitch in bytes, height in rows
struct surf {
    long base;
    int w, h, p;
    surf(const void *b, int width_bytes, int height, int pitch_bytes)
        : base((long)b), w(width_bytes - 1), h(height - 1), p(pitch_bytes - 1) {}
};

#ifdef __SYCL_DEVICE_ONLY__
#define XE2_ASM(...) __asm__(__VA_ARGS__)
#define XE2_ASM_V(...) __asm__ volatile(__VA_ARGS__)
#else
#define XE2_ASM(...)
#define XE2_ASM_V(...)
#endif

// 2D block I/O, lsc_load_block2d / lsc_store_block2d on an explicit surface
// flat[base, width-1, height-1, pitch-1, x (elements), y (rows)]; shape
// d<bits>.[V x]C x R: V blocks of C columns x R rows, block-major.
#define XE2_2D_RD(name, T, shape)                                                  \
    inline T name(const surf &s, int x, int y) {                                   \
        T v;                                                                       \
        XE2_ASM("{\n"                                                              \
                ".decl SB v_type=G type=q num_elts=1 align=qword alias=<%1,0>\n"   \
                ".decl SW v_type=G type=d num_elts=1 align=dword alias=<%2,0>\n"   \
                ".decl SH v_type=G type=d num_elts=1 align=dword alias=<%3,0>\n"   \
                ".decl SP v_type=G type=d num_elts=1 align=dword alias=<%4,0>\n"   \
                ".decl SX v_type=G type=d num_elts=1 align=dword alias=<%5,0>\n"   \
                ".decl SY v_type=G type=d num_elts=1 align=dword alias=<%6,0>\n"   \
                "lsc_load_block2d.ugm (M1, 1) %0:" shape " flat[SB,SW,SH,SP,SX,SY]\n" \
                "}\n"                                                              \
                : "=rw"(v)                                                         \
                : "rw"(s.base), "rw"(s.w), "rw"(s.h), "rw"(s.p), "rw"(x), "rw"(y)); \
        return v;                                                                  \
    }
#define XE2_2D_WR(name, T, shape)                                                  \
    inline void name(const surf &s, int x, int y, T v) {                           \
        XE2_ASM_V("{\n"                                                            \
                ".decl SB v_type=G type=q num_elts=1 align=qword alias=<%0,0>\n"   \
                ".decl SW v_type=G type=d num_elts=1 align=dword alias=<%1,0>\n"   \
                ".decl SH v_type=G type=d num_elts=1 align=dword alias=<%2,0>\n"   \
                ".decl SP v_type=G type=d num_elts=1 align=dword alias=<%3,0>\n"   \
                ".decl SX v_type=G type=d num_elts=1 align=dword alias=<%4,0>\n"   \
                ".decl SY v_type=G type=d num_elts=1 align=dword alias=<%5,0>\n"   \
                "lsc_store_block2d.ugm (M1, 1) flat[SB,SW,SH,SP,SX,SY] %6:" shape "\n" \
                "}\n"                                                              \
                :: "rw"(s.base), "rw"(s.w), "rw"(s.h), "rw"(s.p), "rw"(x), "rw"(y), "rw"(v)); \
    }
XE2_2D_RD(rd_32b_1r16, unsigned, "d32.16x1nn")
XE2_2D_RD(rd_32b_2r16, uint2, "d32.16x2nn")
XE2_2D_RD(rd_32b_8r16, uint8, "d32.16x8nn")
// a 16x1 d16 block is half a GRF, but the load writes a whole (zero-padded) GRF
inline unsigned short rd_16b_1r16(const surf &s, int x, int y) {
    unsigned short v;
    XE2_ASM("{\n"
            ".decl SB v_type=G type=q num_elts=1 align=qword alias=<%1,0>\n"
            ".decl SW v_type=G type=d num_elts=1 align=dword alias=<%2,0>\n"
            ".decl SH v_type=G type=d num_elts=1 align=dword alias=<%3,0>\n"
            ".decl SP v_type=G type=d num_elts=1 align=dword alias=<%4,0>\n"
            ".decl SX v_type=G type=d num_elts=1 align=dword alias=<%5,0>\n"
            ".decl SY v_type=G type=d num_elts=1 align=dword alias=<%6,0>\n"
            ".decl TP v_type=G type=uw num_elts=32 align=GRF\n"
            "lsc_load_block2d.ugm (M1, 1) TP:d16.16x1nn flat[SB,SW,SH,SP,SX,SY]\n"
            "mov (M1, 16) %0(0,0)<1> TP(0,0)<1;1,0>\n"
            "}\n"
            : "=rw"(v)
            : "rw"(s.base), "rw"(s.w), "rw"(s.h), "rw"(s.p), "rw"(x), "rw"(y));
    return v;
}
// two adjacent 1x16 blocks = one 1x32 block: [0] = column x + lane, [1] = x + 16 + lane
XE2_2D_RD(rd_16b_1r16x2, ushort2, "d16.32x1nn")
XE2_2D_RD(rd_16b_8r16, ushort8, "d16.16x8nn")
// two adjacent 8x16 blocks, block-major: [0..7] = columns x..x+15, [8..15] = x+16..x+31
XE2_2D_RD(rd_16b_8r16x2, ushort16, "d16.2x16x8nn")
XE2_2D_WR(wr_16b_8r16, ushort8, "d16.16x8nn")
XE2_2D_WR(wr_32b_8r16, uint8, "d32.16x8nn")
#undef XE2_2D_RD
#undef XE2_2D_WR

// Sub-group block reads (group_load, striped: element i of lane l = p[l + 16 i]).
// p must be 4-byte aligned and point to global memory.
template <int N, typename T> inline void sg_load(const T *p, T (&o)[N]) {
    namespace syclex = sycl::ext::oneapi::experimental;
    auto gp = sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::yes>(p).get_decorated();
    syclex::group_load(sycl::ext::oneapi::this_work_item::get_sub_group(), gp, sycl::span<T, N>(o),
            syclex::properties{syclex::data_placement_striped, syclex::contiguous_memory, syclex::alignment<4>});
}
inline unsigned short sg_rd_us(const unsigned short *p) {
    unsigned short o[1];
    sg_load(p, o);
    return o[0];
}
inline ushort4 sg_rd_us4(const unsigned short *p) {
    unsigned short o[4];
    sg_load(p, o);
    return ushort4{o[0], o[1], o[2], o[3]};
}
inline ushort8 sg_rd_us8(const unsigned short *p) {
    unsigned short o[8];
    sg_load(p, o);
    return ushort8{o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]};
}
inline uint4 sg_rd_u4(const unsigned *p) {
    unsigned o[4];
    sg_load(p, o);
    return uint4{o[0], o[1], o[2], o[3]};
}

// per-lane 3-dword load, L1 and L3 cached
inline uint3 ld_u3_cached(const uint3 *p) {
    uint3 v;
    XE2_ASM("lsc_load.ugm.ca.ca (M1, 16) %0:d32x3 flat[%1]:a64" : "=rw"(v) : "rw"((long)p));
    return v;
}

// DPAS, SIMD16, systolic depth 8. Repeat count R = rows of A: A is one
// short per row per lane (K16 fp16/bf16, or K32 s8), B the VNNI weights;
// na / nb = A / B size in dwords.
#define XE2_DPAS(name, prec, A, B, C, R, na, nb)                                       \
    inline C name(A a, B b, C acc) {                                                    \
        XE2_ASM("{\n"                                                                   \
                ".decl DB v_type=G type=ud num_elts=" #nb " align=GRF alias=<%1,0>\n"   \
                ".decl DA v_type=G type=ud num_elts=" #na " align=GRF alias=<%2,0>\n"   \
                "dpas." prec ".8." #R " (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"              \
                "}\n"                                                                   \
                : "+rw"(acc) : "rw"(b), "rw"(a));                                       \
        return acc;                                                                     \
    }
#define XE2_DPAS_Z(name, prec, A, B, C, R, na, nb)                                      \
    inline C name(A a, B b) {                                                           \
        C d;                                                                            \
        XE2_ASM("{\n"                                                                   \
                ".decl DB v_type=G type=ud num_elts=" #nb " align=GRF alias=<%1,0>\n"   \
                ".decl DA v_type=G type=ud num_elts=" #na " align=GRF alias=<%2,0>\n"   \
                "dpas." prec ".8." #R " (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"          \
                "}\n"                                                                   \
                : "=rw"(d) : "rw"(b), "rw"(a));                                         \
        return d;                                                                       \
    }
XE2_DPAS(dpas_hf, "hf.hf", short, int8, float, 1, 8, 128)
XE2_DPAS(dpas_hf, "hf.hf", short2, int8, float2, 2, 16, 128)
XE2_DPAS(dpas_hf, "hf.hf", short4, int8, float4, 4, 32, 128)
XE2_DPAS(dpas_hf, "hf.hf", short8, int8, float8, 8, 64, 128)
XE2_DPAS(dpas_bf, "bf.bf", short, int8, float, 1, 8, 128)
XE2_DPAS(dpas_bf, "bf.bf", short2, int8, float2, 2, 16, 128)
XE2_DPAS(dpas_bf, "bf.bf", short4, int8, float4, 4, 32, 128)
XE2_DPAS(dpas_bf, "bf.bf", short8, int8, float8, 8, 64, 128)
// s8 A (K32) x s2 B: 4 weights per byte of B
XE2_DPAS(dpas_s2s8, "s2.s8", short, int2, int, 1, 8, 32)
XE2_DPAS(dpas_s2s8, "s2.s8", short2, int2, int2, 2, 16, 32)
XE2_DPAS(dpas_s2s8, "s2.s8", short4, int2, int4, 4, 32, 32)
XE2_DPAS(dpas_s2s8, "s2.s8", short8, int2, int8, 8, 64, 32)
// same with a zero accumulator (src0 = null)
XE2_DPAS_Z(dpas_s2s8_z, "s2.s8", short, int2, int, 1, 8, 32)
XE2_DPAS_Z(dpas_s2s8_z, "s2.s8", short2, int2, int2, 2, 16, 32)
XE2_DPAS_Z(dpas_s2s8_z, "s2.s8", short4, int2, int4, 4, 32, 32)
XE2_DPAS_Z(dpas_s2s8_z, "s2.s8", short8, int2, int8, 8, 64, 32)
#undef XE2_DPAS
#undef XE2_DPAS_Z

// work-group barrier with a local-memory fence only (OpenCL barrier(CLK_LOCAL_MEM_FENCE))
inline void barrier_local() {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
            __spv::MemorySemanticsMask::AcquireRelease | __spv::MemorySemanticsMask::WorkgroupMemory);
#endif
}

// 16-bit activation dtype: raw bits in, fp32 out, round-to-nearest-even back
template <bool BF16> struct dt;
template <> struct dt<false> {
    static float tof(unsigned short h) { return (float)sycl::bit_cast<sycl::half>(h); }
    static unsigned short from(float f) { return sycl::bit_cast<unsigned short>((sycl::half)f); }
};
template <> struct dt<true> {
    static float tof(unsigned short h) { return sycl::bit_cast<float>((unsigned)h << 16); }
    static unsigned short from(float f) {
        return sycl::bit_cast<unsigned short>(sycl::ext::oneapi::bfloat16(f));
    }
};

template <bool BF16, typename A, typename C> inline C mad_k16(A a, int8 b, C acc) {
    if constexpr (BF16) return dpas_bf(a, b, acc);
    else return dpas_hf(a, b, acc);
}

}  // namespace xe2

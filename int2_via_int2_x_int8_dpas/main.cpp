// SYCL driver for int2_int8_dpas.hpp; same methodology as TernOCL's OpenCL
// driver (and xetla int2_fp16_dpas_fp16scales_fast_test it mirrors):
//   - A (--dtype fp16|bf16) in [-5, 5] fake-quantized per (row, 128-group) so
//     that A = q / SA exactly, SA = DT(127 / absmax) recomputed from the
//     fake-quantized A, B codes {0, 1, 3} = {0, +1, -1}, SB DT in [0.75, 15.75];
//   - host gold: q = sat_int8(trunc(A * SA)), int32 dot per group,
//     acc += float(dot) * (SB * (1 / SA)), epilogue, cast to DT;
//   - rotating distinct weight sets (>= --weights-gib, default 2 GiB), every
//     set warmed up, device events -- the A pre-kernel and the GEMM
//     separately, like xetla's "AbsMax" and "GEMM" times;
//   - pass rule of xetla_buff_cmp: fp16 (ulp 64, abs 8), bf16 (32, 16).
// Kernels are templates; tiles and qmode are picked at run time from a
// compiled-in table (--list-tiles), JIT-compiled per kernel on first use.

#include "int2_int8_dpas.hpp"

#include "driver.hpp"
#include "dt16.hpp"
#include "epilogue.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <utility>
#include <vector>

static constexpr int kGS = 128;
static int ldsa(int M) { return int8dpas::ldsa(M); }

struct Args {
    const dt16 *A;
    const int8_t *Aq;
    const dt16 *SA;
    const uint32_t *B;
    const dt16 *SB;
    void *C;
    Epi epi;
    int M, N, K;
};
using LaunchFn = sycl::event (*)(sycl::queue &, const Args &);

template <bool BF16, int Q, int SGM, int NSG, int LS, int U>
static sycl::event launch_gemv(sycl::queue &q, const Args &a) {
    using Kern = int8dpas::Gemv<BF16, Q, SGM, NSG, LS, U>;
    const size_t wgn = 16 * NSG;
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((a.M + SGM - 1) / SGM, (a.N + wgn - 1) / wgn * Kern::WG);
    return q.parallel_for(sycl::nd_range<2>(global, local),
            Kern{a.A, (const signed char *)a.Aq, a.SA, a.B, a.SB, a.C, a.epi, a.M, a.N, a.K});
}

template <bool BF16, int Q, int MT_M, int MT_N, int WG_M, int WG_N>
static sycl::event launch_mt(sycl::queue &q, const Args &a) {
    const size_t tm = MT_M * WG_M, tn = MT_N * WG_N;
    const sycl::range<2> local(1, 16 * WG_M * WG_N);
    const sycl::range<2> global((a.M + tm - 1) / tm, (a.N + tn - 1) / tn * local[1]);
    auto go = [&](auto P, auto F) {
        return q.parallel_for(sycl::nd_range<2>(global, local),
                int8dpas::GemmMT<BF16, Q, MT_M, MT_N, WG_M, WG_N, decltype(P)::value, decltype(F)::value>{
                        a.A, (const signed char *)a.Aq, a.SA, a.B, a.SB, a.C, a.epi, a.M, a.N, a.K});
    };
    using std::integral_constant;
    using T = std::true_type;
    using F = std::false_type;
    switch (a.epi.postop * 2 + (a.epi.f32 ? 1 : 0)) {
    case 0: return go(integral_constant<int, 0>{}, F{});
    case 1: return go(integral_constant<int, 0>{}, T{});
    case 2: return go(integral_constant<int, 1>{}, F{});
    case 3: return go(integral_constant<int, 1>{}, T{});
    case 4: return go(integral_constant<int, 2>{}, F{});
    case 5: return go(integral_constant<int, 2>{}, T{});
    case 6: return go(integral_constant<int, 3>{}, F{});
    case 7: return go(integral_constant<int, 3>{}, T{});
    case 8: return go(integral_constant<int, 4>{}, F{});
    default: return go(integral_constant<int, 4>{}, T{});
    }
}

template <bool BF16, bool WQ>
static sycl::event launch_quant(sycl::queue &q, const dt16 *A, dt16 *SA, int8_t *Aq, int M, int K) {
    return q.parallel_for(sycl::nd_range<2>({(size_t)M, (size_t)(K / kGS) * 16}, {1, 16}),
            int8dpas::QuantA<BF16, WQ>{A, SA, (signed char *)Aq, M, K});
}

// compiled-in tile space: GEMV sgm x nsg x ls x u, large-M tiles (mt_m, mt_n, wg_m, wg_n)
constexpr int kSGM[] = {1, 2, 4, 8}, kNSG[] = {1, 2, 4}, kLS[] = {1, 2, 4, 8}, kU[] = {1, 2};
constexpr int kMT[][4] = {{8, 128, 8, 2}, {8, 128, 4, 2}, {8, 128, 4, 4}, {8, 128, 16, 1}, {8, 128, 2, 4},
        {16, 64, 4, 2}, {16, 64, 8, 2}, {8, 64, 8, 2}, {32, 32, 4, 2}};

struct GemvTile { int sgm, nsg, ls, u; LaunchFn f[2][2]; };  // f[bf16][qmode]
struct MtTile { int mt_m, mt_n, wg_m, wg_n; LaunchFn f[2][2]; };

template <size_t I> struct GemvAt {
    static constexpr int sgm = kSGM[I / 24], nsg = kNSG[I / 8 % 3], ls = kLS[I / 2 % 4], u = kU[I % 2];
    template <bool BF, int Q> static constexpr LaunchFn fn() { return &launch_gemv<BF, Q, sgm, nsg, ls, u>; }
};
template <size_t... I> static constexpr auto gemv_table(std::index_sequence<I...>) {
    return std::array<GemvTile, sizeof...(I)>{{GemvTile{GemvAt<I>::sgm, GemvAt<I>::nsg, GemvAt<I>::ls,
            GemvAt<I>::u, {{GemvAt<I>::template fn<false, 0>(), GemvAt<I>::template fn<false, 1>()},
                                  {GemvAt<I>::template fn<true, 0>(), GemvAt<I>::template fn<true, 1>()}}}...}};
}
template <size_t I> struct MtAt {
    static constexpr int mt_m = kMT[I][0], mt_n = kMT[I][1], wg_m = kMT[I][2], wg_n = kMT[I][3];
    template <bool BF, int Q> static constexpr LaunchFn fn() { return &launch_mt<BF, Q, mt_m, mt_n, wg_m, wg_n>; }
};
template <size_t... I> static constexpr auto mt_table(std::index_sequence<I...>) {
    return std::array<MtTile, sizeof...(I)>{{MtTile{MtAt<I>::mt_m, MtAt<I>::mt_n, MtAt<I>::wg_m, MtAt<I>::wg_n,
            {{MtAt<I>::template fn<false, 0>(), MtAt<I>::template fn<false, 1>()},
                    {MtAt<I>::template fn<true, 0>(), MtAt<I>::template fn<true, 1>()}}}...}};
}
static const auto kGemvTiles = gemv_table(std::make_index_sequence<4 * 3 * 4 * 2>{});
static const auto kMtTiles = mt_table(std::make_index_sequence<sizeof(kMT) / sizeof(kMT[0])>{});

struct RunConfig {
    int m = 1, n = 4096, k = 4096, iters = 50;
    int qmode = 1;  // 0 upfront int8 A, 1 xetla scheme (scale pre-kernel, quantize in GEMM)
    bool validate = true, distinct_sets = false;
    int num_sets = 0;
    double weights_gib = 2.0;
    int sgm = 0, nsg = 0, ls = 0, u = 0;  // GEMV kernel, 0 = default
    int mt_m = 0, mt_n = 128, wg_m = 8, wg_n = 2;  // large-M kernel
    Epilogue epi;
};

// Per (row, group) scale, xetla formula: DT(127 / max(FLT_EPSILON, absmax)).
static void compute_scale_a(const dt16 *A, dt16 *SA, int M, int K) {
#pragma omp parallel for
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < K / kGS; ++g) {
            float mx = FLT_EPSILON;
            for (int p = 0; p < kGS; ++p)
                mx = std::max(mx, std::fabs(tof(A[(size_t)m * K + g * kGS + p])));
            SA[(size_t)g * ldsa(M) + m] = fromf(127.0f / mx);
        }
}

// fp32 accumulator gold; the epilogue and output cast follow in epilogue_ref()
static void compute_gold(const dt16 *A, const uint32_t *B, const dt16 *SA,
        const dt16 *SB, float *C, int M, int K, int N) {
    const int groups = K / kGS;
    std::vector<int8_t> q((size_t)M * K);
#pragma omp parallel for
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            long v = (long)(tof(A[(size_t)m * K + k]) * tof(SA[(size_t)(k / kGS) * ldsa(M) + m]));
            q[(size_t)m * K + k] = (int8_t)std::min(127L, std::max(-128L, v));
        }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int g = 0; g < groups; ++g) {
                int32_t dot = 0;
                for (int p = 0; p < kGS; ++p) {
                    const int ik = g * kGS + p;
                    const uint32_t code = (B[(size_t)(ik / 16) * N + j] >> (2 * (ik % 16))) & 3u;
                    dot += (int)q[(size_t)i * K + ik] * code_to_value(code);
                }
                const float sa = tof(SA[(size_t)g * ldsa(M) + i]);
                const float inv = sa == 0.0f ? 0.0f : 1.0f / sa;
                acc += (float)dot * (tof(SB[(size_t)g * N + j]) * inv);
            }
            C[(size_t)i * N + j] = acc;
        }
}

static void default_tiles(RunConfig &c) {
    if (c.sgm == 0) c.sgm = c.m == 1 ? 1 : (c.m <= 2 ? 2 : (c.m <= 4 ? 4 : 8));
    if (c.nsg == 0) c.nsg = 2;
    if (c.ls == 0) c.ls = c.n <= 8192 ? 4 : 2;
    if (c.u == 0) c.u = 2;
}

static void list_tiles() {
    std::cout << "GEMV tiles (--sgm --nsg --ls --u):\n";
    for (auto &t : kGemvTiles) std::cout << "  " << t.sgm << " " << t.nsg << " " << t.ls << " " << t.u << "\n";
    std::cout << "large-M tiles (--mt-m --mt-n --wg-m --wg-n):\n";
    for (auto &t : kMtTiles) std::cout << "  " << t.mt_m << " " << t.mt_n << " " << t.wg_m << " " << t.wg_n << "\n";
}

static void run(RunConfig cfg) {
    const int M = cfg.m, N = cfg.n, K = cfg.k;
    if (K % kGS || N % 16) { std::cerr << "need K % 128 == 0 and N % 16 == 0\n"; std::exit(1); }
    if (cfg.qmode < 0 || cfg.qmode > 1) { std::cerr << "--qmode 0|1\n"; std::exit(1); }
    const bool mt = cfg.mt_m > 0 || M >= 64;
    if (mt && cfg.mt_m == 0) cfg.mt_m = 8;
    default_tiles(cfg);

    const int bf = dt_is_bf16(), qm = cfg.qmode;
    LaunchFn launch = nullptr;
    if (mt) {
        for (auto &t : kMtTiles)
            if (t.mt_m == cfg.mt_m && t.mt_n == cfg.mt_n && t.wg_m == cfg.wg_m && t.wg_n == cfg.wg_n)
                launch = t.f[bf][qm];
    } else {
        for (auto &t : kGemvTiles)
            if (t.sgm == cfg.sgm && t.nsg == cfg.nsg && t.ls == cfg.ls && t.u == cfg.u) launch = t.f[bf][qm];
    }
    if (!launch) { std::cerr << "tile not compiled in (see --list-tiles)\n"; std::exit(1); }
    auto quant = bf ? (qm == 0 ? &launch_quant<true, true> : &launch_quant<true, false>)
                    : (qm == 0 ? &launch_quant<false, true> : &launch_quant<false, false>);

    sycl::queue q = make_queue();
    static const char *qname[] = {"upfront (int8 A + scale pre-kernel)",
            "xetla scheme (scale pre-kernel, quantize in GEMM)"};
    std::cout << "Problem: M=" << M << " N=" << N << " K=" << K << " scale_gs=" << kGS
              << " dtype=" << dt_name() << " epilogue=" << cfg.epi.name() << "\n"
              << "A quant: qmode=" << qm << " " << qname[qm] << "\n";
    if (mt)
        std::cout << "Tile (mt): sg " << cfg.mt_m << "x" << cfg.mt_n << ", wg " << cfg.wg_m
                  << "x" << cfg.wg_n << " sub-groups, 256 GRF\n";
    else
        std::cout << "Tile: sg_m=" << cfg.sgm << " wg_n=" << 16 * cfg.nsg << " (nsg=" << cfg.nsg
                  << ") ls=" << cfg.ls << " u=" << cfg.u << "\n";

    const size_t size_a = (size_t)M * K, size_b = (size_t)(K / 16) * N;
    const size_t size_c = (size_t)M * N, size_sb = (size_t)(K / kGS) * N;
    const size_t size_sa = (size_t)(K / kGS) * ldsa(M);
    const Epilogue &ep = cfg.epi;
    const size_t os = ep.out_size();
    const double bytes_set = size_a * 3.0 + size_b * 4.0 + size_c * (double)os + (size_sb + size_sa) * 2.0
            + (ep.needs_other() ? size_c * 2.0 : 0.0) + (ep.needs_bias() ? N * (double)os : 0.0);
    const double wbytes_set = size_b * 4.0 + size_sb * 2.0;
    int sets = cfg.num_sets > 0 ? cfg.num_sets
                                : std::max(1, (int)std::ceil(cfg.weights_gib * (1 << 30) / wbytes_set));
    std::cout << "Per-set bytes: " << bytes_set / (1 << 20) << " MiB (weights " << wbytes_set / (1 << 20)
              << " MiB); using " << sets << " distinct sets -> weights " << std::fixed
              << std::setprecision(3) << wbytes_set * sets / (1 << 30) << " GiB, total "
              << bytes_set * sets / (1 << 30) << " GiB\n";

    std::vector<dt16> A(size_a), SA(size_sa), SB(size_sb), Oth;
    std::vector<unsigned char> Bias, Ch(size_c * os);
    std::vector<uint32_t> B(size_b);
    static std::mt19937 egen(std::random_device{}());
    auto fill = [&] {
#pragma omp parallel for
        for (size_t i = 0; i < size_a; ++i) A[i] = fromf(rnd_f(-5.0f, 5.0f));
#pragma omp parallel for
        for (size_t i = 0; i < size_b; ++i) B[i] = rnd_ternary_word();
        for (size_t i = 0; i < size_sb; ++i) SB[i] = fromf(rnd_f(0.0f, 15.0f) + 0.75f);
        // fake-quantize A (RTE, as xetla), then recompute SA from the result
        compute_scale_a(A.data(), SA.data(), M, K);
#pragma omp parallel for
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                const float sa = tof(SA[(size_t)(k / kGS) * ldsa(M) + m]);
                long v = std::lrintf(tof(A[(size_t)m * K + k]) * sa);
                v = std::min(127L, std::max(-128L, v));
                A[(size_t)m * K + k] = fromf((float)v * (1.0f / sa));
            }
        compute_scale_a(A.data(), SA.data(), M, K);
        fill_epilogue_inputs(ep, Oth, Bias, M, N, egen);
    };
    fill();

    const int ngold = (cfg.validate && cfg.distinct_sets) ? sets : 1;
    std::vector<std::vector<unsigned char>> gold(cfg.validate ? ngold : 0);
    std::vector<float> acc_gold;
    auto make_gold = [&](std::vector<unsigned char> &g) {
        acc_gold.resize(size_c);
        compute_gold(A.data(), B.data(), SA.data(), SB.data(), acc_gold.data(), M, K, N);
        g.resize(size_c * os);
        epilogue_ref(ep, acc_gold.data(), Oth.data(), Bias.data(), g.data(), M, N);
    };
    if (cfg.validate) make_gold(gold[0]);

    std::vector<Args> args(sets);
    std::vector<void *> allocs;
    auto dev_alloc = [&](const void *src, size_t bytes) {
        void *p = sycl::malloc_device(std::max<size_t>(bytes, 64), q);
        if (!p) { std::cerr << "device allocation failed\n"; std::exit(1); }
        if (src && bytes) q.memcpy(p, src, bytes);
        allocs.push_back(p);
        return p;
    };
    for (int s = 0; s < sets; ++s) {
        if (cfg.distinct_sets && s > 0) {
            fill();
            if (cfg.validate) make_gold(gold[s]);
        }
        Args &a = args[s];
        a.A = (const dt16 *)dev_alloc(A.data(), size_a * 2);
        a.B = (const uint32_t *)dev_alloc(B.data(), size_b * 4);
        a.SB = (const dt16 *)dev_alloc(SB.data(), size_sb * 2);
        // SA / Aq are produced on the device (junk-filled so a missing write
        // cannot pass validation)
        a.SA = (const dt16 *)dev_alloc(nullptr, size_sa * 2);
        a.Aq = (const int8_t *)dev_alloc(nullptr, qm == 0 ? size_a : 64);
        q.memset((void *)a.SA, 0x5a, size_sa * 2);
        q.memset((void *)a.Aq, 0x5a, qm == 0 ? size_a : 64);
        a.C = dev_alloc(nullptr, size_c * os);
        q.memset(a.C, 0, size_c * os);
        a.epi.other = (const unsigned short *)dev_alloc(Oth.data(), Oth.size() * 2);
        a.epi.bias = dev_alloc(Bias.data(), Bias.size());
        a.epi.postop = ep.postop;
        a.epi.f32 = ep.out_f32;
        a.M = M, a.N = N, a.K = K;
        q.wait();
    }

    double gemm_ns = 0.0, pre_ns = 0.0, host_ms = 0.0;
    int timed = 0;
    for (int it = 0; it < sets + cfg.iters; ++it) {
        const int s = it % sets;
        const Args &a = args[s];
        auto t0 = std::chrono::high_resolution_clock::now();
        sycl::event eq = quant(q, a.A, (dt16 *)a.SA, (int8_t *)a.Aq, M, K);
        sycl::event e = launch(q, a);
        e.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        if (it >= sets) {
            gemm_ns += ev_ns(e);
            pre_ns += ev_ns(eq);
            host_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++timed;
        }
    }
    if (timed) {
        const double gms = gemm_ns / 1e6 / timed, pms = pre_ns / 1e6 / timed, hms = host_ms / timed;
        const double tms = gms + pms;
        const double bytes = (double)M * K * 2 + (double)K * N / 4.0 + (double)M * N * os
                + (double)(K / kGS) * (N + M) * 2 + (ep.needs_other() ? (double)M * N * 2 : 0.0)
                + (ep.needs_bias() ? (double)N * os : 0.0);
        const double gib = 1024.0 * 1024.0 * 1024.0, flop = 2.0 * M * N * K;
        std::cout << std::fixed << std::setprecision(5)
                  << "Avg host  time: " << hms << " ms (" << flop / (hms * 1e-3) / 1e9 << " GFLOPS)\n"
                  << "Avg dev   pre : " << pms << " ms\n"
                  << "Avg dev   gemm: " << gms << " ms (" << flop / (gms * 1e-3) / 1e9 << " GFLOPS, "
                  << bytes / (gms * 1e-3) / gib << " GiB/s)\n"
                  << "Avg dev   time: " << tms << " ms (" << flop / (tms * 1e-3) / 1e9 << " GFLOPS, "
                  << bytes / (tms * 1e-3) / gib << " GiB/s)\n"
                  << "Bytes/GEMM: " << bytes / (1 << 20) << " MiB\n";
    }

    if (cfg.validate) {
        int passed = 0;
        std::vector<unsigned char> C0;
        {  // device scale_a of the last set vs host (last filled) one
            std::vector<dt16> SAh(size_sa);
            q.memcpy(SAh.data(), args[sets - 1].SA, size_sa * 2).wait();
            size_t diff = 0;
            for (int g = 0; g < K / kGS; ++g)
                for (int m = 0; m < M; ++m) {
                    const size_t i = (size_t)g * ldsa(M) + m;
                    diff += SAh[i] != SA[i];
                }
            std::cout << "scale_a: " << diff << "/" << (size_t)(K / kGS) * M << " differ from host\n";
        }
        for (int s = 0; s < sets; ++s) {
            q.memcpy(Ch.data(), args[s].C, size_c * os).wait();
            const auto &g = gold[cfg.distinct_sets ? s : 0];
            const std::string label = "validation [set " + std::to_string(s) + "/" + std::to_string(sets) + "]";
            bool ok;
            if (!cfg.distinct_sets && s > 0 && std::memcmp(Ch.data(), C0.data(), size_c * os) == 0)
                ok = true;
            else if (ep.out_f32)
                ok = compare_f32((const float *)Ch.data(), (const float *)g.data(), size_c, label);
            else
                ok = compare_dt((const dt16 *)Ch.data(), (const dt16 *)g.data(), size_c, label,
                        ep.abs_tol(), ep.ulp_tol());
            if (s == 0) C0 = Ch;
            passed += ok;
            if (!ok) std::cout << "FAILED at set " << s << "\n";
            if (!cfg.distinct_sets && s == 0 && !ok) break;
        }
        std::cout << "Validation summary: " << passed << "/" << sets << " sets PASSED\n";
    }

    for (void *p : allocs) sycl::free(p, q);
}

int main(int argc, char **argv) {
    RunConfig cfg;
    auto ival = [&](int &i) { return std::atoi(argv[++i]); };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i + 1 < argc) cfg.m = ival(i);
        else if (a == "--n" && i + 1 < argc) cfg.n = ival(i);
        else if (a == "--k" && i + 1 < argc) cfg.k = ival(i);
        else if (a == "--dtype" && i + 1 < argc) {
            const std::string d = argv[++i];
            if (d != "fp16" && d != "bf16") { std::cerr << "--dtype fp16|bf16\n"; return 1; }
            dt_is_bf16() = d == "bf16";
        }
        else if (a == "--qmode" && i + 1 < argc) cfg.qmode = ival(i);
        else if (a == "--postop" && i + 1 < argc) {
            cfg.epi.postop = ival(i);
            if (cfg.epi.postop < 0 || cfg.epi.postop > 4) { std::cerr << "--postop 0..4\n"; return 1; }
        }
        else if (a == "--out-f32") cfg.epi.out_f32 = true;
        else if (a == "--iters" && i + 1 < argc) cfg.iters = ival(i);
        else if (a == "--sets" && i + 1 < argc) cfg.num_sets = ival(i);
        else if (a == "--weights-gib" && i + 1 < argc) cfg.weights_gib = std::atof(argv[++i]);
        else if (a == "--no-validate") cfg.validate = false;
        else if (a == "--distinct-sets") cfg.distinct_sets = true;
        else if (a == "--sgm" && i + 1 < argc) cfg.sgm = ival(i);
        else if (a == "--nsg" && i + 1 < argc) cfg.nsg = ival(i);
        else if (a == "--wgn" && i + 1 < argc) cfg.nsg = ival(i) / 16;
        else if (a == "--ls" && i + 1 < argc) cfg.ls = ival(i);
        else if (a == "--u" && i + 1 < argc) cfg.u = ival(i);
        else if (a == "--mt-m" && i + 1 < argc) cfg.mt_m = ival(i);
        else if (a == "--mt-n" && i + 1 < argc) cfg.mt_n = ival(i);
        else if (a == "--wg-m" && i + 1 < argc) cfg.wg_m = ival(i);
        else if (a == "--wg-n" && i + 1 < argc) cfg.wg_n = ival(i);
        else if (a == "--list-tiles") { list_tiles(); return 0; }
        else {
            std::cout << "Usage: " << argv[0]
                      << " [--m M] [--n N] [--k K] [--dtype fp16|bf16] [--qmode 0|1] [--iters N] [--no-validate]\n"
                         "       [--distinct-sets] [--sets N | --weights-gib G]\n"
                         "       [--sgm 1|2|4|8] [--wgn W | --nsg S] [--ls L] [--u U]\n"
                         "       [--mt-m 8k --mt-n 16k --wg-m W --wg-n W] [--list-tiles]\n"
                         "       [--postop 0|1|2|3|4] [--out-f32]   epilogue: 0 none, 1 silu(acc)*other,\n"
                         "       2 acc+other, 3 acc+bias[n], 4 sigmoid(acc); --out-f32 = fp32 C/bias\n"
                         "qmode: 0 upfront int8 A, 1 xetla scheme (quantize A in the GEMM)\n";
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }
    run(cfg);
    return 0;
}

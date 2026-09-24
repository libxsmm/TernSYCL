// SYCL driver for int2_fp16_upcvt.hpp; same methodology as TernOCL's OpenCL
// driver (and the xetla int2_fp16_upcvt_dpas_fast_test it mirrors):
//   - same operation, layouts, input generation and host gold (fp32
//     accumulate per 128-group, x scale, epilogue, cast to the output type);
//   - rotate over enough distinct device buffer sets until the distinct
//     weights (B + S) reach --weights-gib (default 2 GiB), warm up every set
//     once, so every timed call streams its weights from DRAM;
//   - device profiling events; same byte model for GiB/s;
//   - same pass rule as xetla_buff_cmp: fp16 (ulp 64, abs 8), bf16 (32, 16).
// Kernels are templates; the tiles are picked at run time from a compiled-in
// table (--list-tiles), JIT-compiled per kernel on first use.

#include "int2_fp16_upcvt.hpp"

#include "driver.hpp"
#include "dt16.hpp"
#include "epilogue.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <utility>
#include <vector>

static constexpr int kGS = 128;

struct Args {
    const dt16 *A;
    const uint32_t *B;
    const dt16 *S;
    void *C;
    Epi epi;
    int M, N, K;
};
using LaunchFn = sycl::event (*)(sycl::queue &, const Args &);

template <bool BF16, int SGM, int NSG, int LS, int U>
static sycl::event launch_gemv(sycl::queue &q, const Args &a) {
    using Kern = upcvt::Gemv<BF16, SGM, NSG, LS, U>;
    const size_t wgn = 16 * NSG;
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((a.M + SGM - 1) / SGM, (a.N + wgn - 1) / wgn * Kern::WG);
    return q.parallel_for(sycl::nd_range<2>(global, local), Kern{a.A, a.B, a.S, a.C, a.epi, a.M, a.N, a.K});
}

template <bool BF16, int MT_M, int MT_N, int WG_M, int WG_N>
static sycl::event launch_mt(sycl::queue &q, const Args &a) {
    using Kern = upcvt::GemmMT<BF16, MT_M, MT_N, WG_M, WG_N>;
    const size_t tm = MT_M * WG_M, tn = MT_N * WG_N;
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((a.M + tm - 1) / tm, (a.N + tn - 1) / tn * Kern::WG);
    return q.parallel_for(sycl::nd_range<2>(global, local), Kern{a.A, a.B, a.S, a.C, a.epi, a.M, a.N, a.K});
}

// compiled-in tile space: GEMV sgm x nsg x ls x u, large-M tile x wg
constexpr int kSGM[] = {1, 2, 4, 8}, kNSG[] = {1, 2, 4}, kLS[] = {1, 2, 4, 6, 8}, kU[] = {1, 2};
constexpr int kMT[][2] = {{16, 16}, {32, 16}, {64, 16}, {96, 16}, {128, 16}, {16, 32}, {32, 32}, {64, 32}};
constexpr int kWG[][2] = {{1, 4}, {1, 8}, {2, 2}, {2, 4}, {4, 2}, {4, 4}, {8, 1}};

struct GemvTile { int sgm, nsg, ls, u; LaunchFn f[2]; };
struct MtTile { int mt_m, mt_n, wg_m, wg_n; LaunchFn f[2]; };

template <size_t I> struct GemvAt {
    static constexpr int sgm = kSGM[I / 30], nsg = kNSG[I / 10 % 3], ls = kLS[I / 2 % 5], u = kU[I % 2];
};
template <size_t... I> static constexpr auto gemv_table(std::index_sequence<I...>) {
    return std::array<GemvTile, sizeof...(I)>{{GemvTile{GemvAt<I>::sgm, GemvAt<I>::nsg, GemvAt<I>::ls,
            GemvAt<I>::u, {&launch_gemv<false, GemvAt<I>::sgm, GemvAt<I>::nsg, GemvAt<I>::ls, GemvAt<I>::u>,
                    &launch_gemv<true, GemvAt<I>::sgm, GemvAt<I>::nsg, GemvAt<I>::ls, GemvAt<I>::u>}}...}};
}
template <size_t I> struct MtAt {
    static constexpr int mt_m = kMT[I / 7][0], mt_n = kMT[I / 7][1], wg_m = kWG[I % 7][0], wg_n = kWG[I % 7][1];
};
template <size_t... I> static constexpr auto mt_table(std::index_sequence<I...>) {
    return std::array<MtTile, sizeof...(I)>{{MtTile{MtAt<I>::mt_m, MtAt<I>::mt_n, MtAt<I>::wg_m, MtAt<I>::wg_n,
            {&launch_mt<false, MtAt<I>::mt_m, MtAt<I>::mt_n, MtAt<I>::wg_m, MtAt<I>::wg_n>,
                    &launch_mt<true, MtAt<I>::mt_m, MtAt<I>::mt_n, MtAt<I>::wg_m, MtAt<I>::wg_n>}}...}};
}
static const auto kGemvTiles = gemv_table(std::make_index_sequence<4 * 3 * 5 * 2>{});
static const auto kMtTiles = mt_table(std::make_index_sequence<8 * 7>{});

struct RunConfig {
    int m = 1, n = 4096, k = 4096, iters = 50;
    bool validate = true, distinct_sets = false;
    int num_sets = 0;
    double weights_gib = 2.0;
    int sgm = 0, nsg = 0, ls = 0, u = 0;  // 0 = default dispatch
    // large-M kernel: sub-group tile mt_m x mt_n, work-group wg_m x wg_n
    // sub-groups; used when mt_m > 0 or M >= 64
    int mt_m = 0, mt_n = 16, wg_m = 2, wg_n = 4;
    Epilogue epi;
};

// fp32 accumulator gold; the epilogue and output cast follow in epilogue_ref()
static void compute_gold(const dt16 *A, const uint32_t *B, const dt16 *S,
        float *C, int M, int K, int N) {
    const int groups = K / kGS;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int g = 0; g < groups; ++g) {
                float partial = 0.0f;
                for (int p = 0; p < kGS; ++p) {
                    int ik = g * kGS + p;
                    uint32_t code = (B[(size_t)(ik / 16) * N + j] >> (2 * (ik % 16))) & 3u;
                    int bv = code_to_value(code);
                    if (bv) partial += tof(A[(size_t)i * K + ik]) * (float)bv;
                }
                acc += partial * tof(S[(size_t)g * N + j]);
            }
            C[(size_t)i * N + j] = acc;
        }
}

// Default tiles (TernOCL's B70 table). M = 1: keyed by (K, N) like the
// plugin's dispatch; M > 1: SGM-row tile.
static void default_tiles(RunConfig &c) {
    if (c.sgm == 0) c.sgm = c.m == 1 ? 1 : (c.m <= 2 ? 2 : (c.m <= 4 ? 4 : 8));
    int wgn = 64, ls = 1, u = 2;
    if (c.m == 1) {
        const int K = c.k, N = c.n;
        if (K == 5120 && N == 34816)       { wgn = 16; ls = 4; u = 2; }  // gate_up
        else if (K == 17408 && N == 5120)  { wgn = 32; ls = 4; u = 2; }  // down
        else if (K == 5120 && N == 16384)  { wgn = 16; ls = 2; u = 1; }  // in_proj_qkvz
        else if (K == 6144 && N == 5120)   { wgn = 32; ls = 6; u = 1; }  // out_proj
        else if (K == 5120 && N == 14336)  { wgn = 32; ls = 2; u = 1; }  // qkv
        else if (K == 5120 && N == 248320) { wgn = 16; ls = 4; u = 2; }  // lm_head
        else if (N <= 8192)                { wgn = 32; ls = 4; u = 2; }
        else                               { wgn = 16; ls = 2; u = 1; }
    }
    if (c.nsg == 0) c.nsg = wgn / 16;
    if (c.ls == 0) c.ls = ls;
    if (c.u == 0) c.u = u;
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
    const bool mt = cfg.mt_m > 0 || M >= 64;
    if (mt && cfg.mt_m == 0) cfg.mt_m = 64;
    default_tiles(cfg);

    LaunchFn launch = nullptr;
    const int bf = dt_is_bf16();
    if (mt) {
        for (auto &t : kMtTiles)
            if (t.mt_m == cfg.mt_m && t.mt_n == cfg.mt_n && t.wg_m == cfg.wg_m && t.wg_n == cfg.wg_n) launch = t.f[bf];
    } else {
        for (auto &t : kGemvTiles)
            if (t.sgm == cfg.sgm && t.nsg == cfg.nsg && t.ls == cfg.ls && t.u == cfg.u) launch = t.f[bf];
    }
    if (!launch) { std::cerr << "tile not compiled in (see --list-tiles)\n"; std::exit(1); }

    sycl::queue q = make_queue();
    std::cout << "Problem: M=" << M << " N=" << N << " K=" << K << " scale_gs=" << kGS
              << " dtype=" << dt_name() << " epilogue=" << cfg.epi.name() << "\n";
    if (mt)
        std::cout << "Tile (mt): sg " << cfg.mt_m << "x" << cfg.mt_n << ", wg " << cfg.wg_m
                  << "x" << cfg.wg_n << " sub-groups, 256 GRF\n";
    else
        std::cout << "Tile: sg_m=" << cfg.sgm << " wg_n=" << 16 * cfg.nsg << " (nsg=" << cfg.nsg
                  << ") ls=" << cfg.ls << " u=" << cfg.u << "\n";

    const Epilogue &ep = cfg.epi;
    const size_t os = ep.out_size();
    const size_t size_a = (size_t)M * K, size_b = (size_t)(K / 16) * N;
    const size_t size_c = (size_t)M * N, size_s = (size_t)(K / kGS) * N;
    const double bytes_set = size_a * 2.0 + size_b * 4.0 + size_c * (double)os + size_s * 2.0
            + (ep.needs_other() ? size_c * 2.0 : 0.0) + (ep.needs_bias() ? N * (double)os : 0.0);
    const double wbytes_set = size_b * 4.0 + size_s * 2.0;
    int sets = cfg.num_sets > 0 ? cfg.num_sets
                                : std::max(1, (int)std::ceil(cfg.weights_gib * (1 << 30) / wbytes_set));
    std::cout << "Per-set bytes: " << bytes_set / (1 << 20) << " MiB (weights " << wbytes_set / (1 << 20)
              << " MiB); using " << sets << " distinct sets -> weights " << std::fixed
              << std::setprecision(3) << wbytes_set * sets / (1 << 30) << " GiB, total "
              << bytes_set * sets / (1 << 30) << " GiB\n";

    std::vector<dt16> A(size_a), S(size_s), Oth;
    std::vector<unsigned char> Bias, Ch(size_c * os);
    std::vector<uint32_t> B(size_b);
    static std::mt19937 egen(std::random_device{}());
    auto fill = [&] {
#pragma omp parallel for
        for (size_t i = 0; i < size_a; ++i) A[i] = fromf(rnd_f(-5.0f, 5.0f));
#pragma omp parallel for
        for (size_t i = 0; i < size_b; ++i) B[i] = rnd_ternary_word();
        for (size_t i = 0; i < size_s; ++i) S[i] = fromf(rnd_f(0.0f, 15.0f) + 0.75f);
        fill_epilogue_inputs(ep, Oth, Bias, M, N, egen);
    };
    fill();

    const int ngold = (cfg.validate && cfg.distinct_sets) ? sets : 1;
    std::vector<std::vector<unsigned char>> gold(cfg.validate ? ngold : 0);
    std::vector<float> acc_gold;
    auto make_gold = [&](std::vector<unsigned char> &g) {
        acc_gold.resize(size_c);
        compute_gold(A.data(), B.data(), S.data(), acc_gold.data(), M, K, N);
        g.resize(size_c * os);
        epilogue_ref(ep, acc_gold.data(), Oth.data(), Bias.data(), g.data(), M, N);
    };
    if (cfg.validate) make_gold(gold[0]);

    std::vector<Args> args(sets);
    std::vector<void *> allocs;
    auto dev_copy = [&](const void *src, size_t bytes) {
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
        a.A = (const dt16 *)dev_copy(A.data(), size_a * 2);
        a.B = (const uint32_t *)dev_copy(B.data(), size_b * 4);
        a.S = (const dt16 *)dev_copy(S.data(), size_s * 2);
        a.C = dev_copy(nullptr, size_c * os);
        q.memset(a.C, 0, size_c * os);
        a.epi.other = (const unsigned short *)dev_copy(Oth.data(), Oth.size() * 2);
        a.epi.bias = dev_copy(Bias.data(), Bias.size());
        a.epi.postop = ep.postop;
        a.epi.f32 = ep.out_f32;
        a.M = M, a.N = N, a.K = K;
        q.wait();
    }

    double dev_ns = 0.0, host_ms = 0.0;
    int timed = 0;
    for (int it = 0; it < sets + cfg.iters; ++it) {
        const int s = it % sets;
        auto t0 = std::chrono::high_resolution_clock::now();
        sycl::event e = launch(q, args[s]);
        e.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        if (it >= sets) {
            dev_ns += ev_ns(e);
            host_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++timed;
        }
    }
    if (timed) {
        const double dms = dev_ns / 1e6 / timed, hms = host_ms / timed;
        const double bytes = (double)M * K * 2 + (double)K * N / 4.0 + (double)M * N * os
                + (double)(K / kGS) * N * 2 + (ep.needs_other() ? (double)M * N * 2 : 0.0)
                + (ep.needs_bias() ? (double)N * os : 0.0);
        const double gib = 1024.0 * 1024.0 * 1024.0, flop = 2.0 * M * N * K;
        std::cout << std::fixed << std::setprecision(5)
                  << "Avg host  time: " << hms << " ms (" << flop / (hms * 1e-3) / 1e9
                  << " GFLOPS, " << bytes / (hms * 1e-3) / gib << " GiB/s)\n"
                  << "Avg dev   time: " << dms << " ms (" << flop / (dms * 1e-3) / 1e9
                  << " GFLOPS, " << bytes / (dms * 1e-3) / gib << " GiB/s)\n"
                  << "Bytes/GEMM: " << bytes / (1 << 20) << " MiB\n";
    }

    if (cfg.validate) {
        int passed = 0;
        std::vector<unsigned char> C0;
        for (int s = 0; s < sets; ++s) {
            q.memcpy(Ch.data(), args[s].C, size_c * os).wait();
            const auto &g = gold[cfg.distinct_sets ? s : 0];
            const std::string label = "validation [set " + std::to_string(s) + "/" + std::to_string(sets) + "]";
            bool ok;
            // shared inputs: every set must reproduce set 0 bit for bit
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
        else if (a == "--postop" && i + 1 < argc) {
            cfg.epi.postop = ival(i);
            if (cfg.epi.postop < 0 || cfg.epi.postop > 4) { std::cerr << "--postop 0..4\n"; return 1; }
        }
        else if (a == "--out-f32") cfg.epi.out_f32 = true;
        else if (a == "--list-tiles") { list_tiles(); return 0; }
        else {
            std::cout << "Usage: " << argv[0]
                      << " [--m M] [--n N] [--k K] [--dtype fp16|bf16] [--iters N] [--no-validate] [--distinct-sets]\n"
                         "       [--sets N | --weights-gib G] [--sgm 1|2|4|8] [--wgn W | --nsg S] [--ls L] [--u U]\n"
                         "       [--mt-m 8k --mt-n 16k --wg-m W --wg-n W] [--list-tiles]\n"
                         "       [--postop 0|1|2|3|4] [--out-f32]   epilogue: 0 none, 1 silu(acc)*other,\n"
                         "       2 acc+other, 3 acc+bias[n], 4 sigmoid(acc); --out-f32 = fp32 C/bias\n";
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }
    run(cfg);
    return 0;
}

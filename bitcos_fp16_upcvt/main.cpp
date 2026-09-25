// SYCL driver for bitcos_fp16_upcvt.hpp; same methodology as TernOCL's OpenCL
// BITCOS driver: random ternary weights at zero density --z packed as
// xetla_vllm_plugin.pack_bitcos does (common/bitcos.hpp), rotation over enough
// distinct device buffer sets that the distinct weights (BITCOS buffer +
// scales) reach --weights-gib (default 2 GiB), device-event timing, fp32 host
// gold (fp32 accumulate per 128-group, x scale), the xetla_buff_cmp pass rule.
// Kernels are templates; tiles and the scale apply are picked at run time from
// a compiled-in table (--list-tiles).

#include "bitcos_fp16_upcvt.hpp"

#include "bitcos.hpp"
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
    const uint32_t *SR;
    void *C;
    Epi epi;
    int M, N, K;
};
using LaunchFn = sycl::event (*)(sycl::queue &, const Args &);

// kernel variants per tile: fp16 visa-hmul, fp16 int-and, fp16 simt-hmul, bf16 int-and
constexpr int kVariants = 4;
static int variant(bool bf16, int ap) { return bf16 ? 3 : ap; }

template <bool BF16, int AP, int SGM, int NSG, int LS>
static sycl::event launch_gemv(sycl::queue &q, const Args &a) {
    using Kern = bitcos::Gemv<BF16, AP, SGM, NSG, LS>;
    const size_t wgn = 16 * NSG;
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((a.M + SGM - 1) / SGM, (a.N + wgn - 1) / wgn * Kern::WG);
    return q.parallel_for(sycl::nd_range<2>(global, local),
            Kern{a.A, a.B, a.S, a.SR, a.C, a.epi, a.M, a.N, a.K});
}

template <bool BF16, int AP, int MT_M, int MT_N, int WG_M, int WG_N>
static sycl::event launch_mt(sycl::queue &q, const Args &a) {
    using Kern = bitcos::GemmMT<BF16, AP, MT_M, MT_N, WG_M, WG_N>;
    const size_t tm = MT_M * WG_M, tn = MT_N * WG_N;
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((a.M + tm - 1) / tm, (a.N + tn - 1) / tn * Kern::WG);
    return q.parallel_for(sycl::nd_range<2>(global, local),
            Kern{a.A, a.B, a.S, a.SR, a.C, a.epi, a.M, a.N, a.K});
}

// compiled-in tile space: GEMV sgm x nsg x ls (work-group <= 1024), M-tiled tile x wg
constexpr int kSGM[] = {1, 2, 4, 8}, kNSG[] = {1, 2, 4, 8, 16}, kLS[] = {1, 2, 4, 8, 16};
constexpr int kMT[][2] = {{16, 16}, {32, 16}, {64, 16}, {16, 32}, {32, 32}, {16, 64}};
constexpr int kWG[][2] = {{1, 4}, {1, 8}, {2, 2}, {2, 4}, {4, 2}};

struct GemvTile { int sgm, nsg, ls; LaunchFn f[kVariants]; };
struct MtTile { int mt_m, mt_n, wg_m, wg_n; LaunchFn f[kVariants]; };

template <size_t I> struct GemvAt {
    static constexpr int sgm = kSGM[I / 25], nsg = kNSG[I / 5 % 5], ls = kLS[I % 5];
    static constexpr bool ok = nsg * ls <= 64;
    template <bool BF, int AP> static constexpr LaunchFn fn() {
        if constexpr (ok) return &launch_gemv<BF, AP, sgm, nsg, ls>;
        else return nullptr;
    }
};
template <size_t... I> static constexpr auto gemv_table(std::index_sequence<I...>) {
    using bitcos::INT, bitcos::SIMT, bitcos::VISA;
    return std::array<GemvTile, sizeof...(I)>{{GemvTile{GemvAt<I>::sgm, GemvAt<I>::nsg, GemvAt<I>::ls,
            {GemvAt<I>::template fn<false, VISA>(), GemvAt<I>::template fn<false, INT>(),
                    GemvAt<I>::template fn<false, SIMT>(), GemvAt<I>::template fn<true, INT>()}}...}};
}
template <size_t I> struct MtAt {
    static constexpr int mt_m = kMT[I / 5][0], mt_n = kMT[I / 5][1], wg_m = kWG[I % 5][0], wg_n = kWG[I % 5][1];
    template <bool BF, int AP> static constexpr LaunchFn fn() { return &launch_mt<BF, AP, mt_m, mt_n, wg_m, wg_n>; }
};
template <size_t... I> static constexpr auto mt_table(std::index_sequence<I...>) {
    using bitcos::INT, bitcos::SIMT, bitcos::VISA;
    return std::array<MtTile, sizeof...(I)>{{MtTile{MtAt<I>::mt_m, MtAt<I>::mt_n, MtAt<I>::wg_m, MtAt<I>::wg_n,
            {MtAt<I>::template fn<false, VISA>(), MtAt<I>::template fn<false, INT>(),
                    MtAt<I>::template fn<false, SIMT>(), MtAt<I>::template fn<true, INT>()}}...}};
}
static const auto kGemvTiles = gemv_table(std::make_index_sequence<4 * 5 * 5>{});
static const auto kMtTiles = mt_table(std::make_index_sequence<6 * 5>{});

struct RunConfig {
    int m = 1, n = 4096, k = 4096, iters = 50;
    double z = 0.40;
    bool validate = true, distinct_sets = false, int_apply = false, simt_mul = false;
    int num_sets = 0;
    double weights_gib = 2.0;
    int sgm = 0, nsg = 0, ls = 0;                  // 0 = default
    int mt_m = 0, mt_n = 16, wg_m = 1, wg_n = 4;  // mt_m > 0: M-tiled GEMM
    Epilogue epi;
};

// fp32 accumulator gold from the decoded codes; epilogue_ref() follows
static void compute_gold(const dt16 *A, const Bitcos &b, const dt16 *S, float *C, int M) {
    const int K = b.K, N = b.N;
#pragma omp parallel
    {
        std::vector<int8_t> codes(K);
#pragma omp for schedule(dynamic, 16)
        for (int j = 0; j < N; ++j) {
            bitcos_decode_col(b, j, codes.data());
            for (int i = 0; i < M; ++i) {
                float acc = 0.0f;
                for (int g = 0; g < K / kGS; ++g) {
                    float partial = 0.0f;
                    for (int p = 0; p < kGS; ++p) {
                        const int k = g * kGS + p;
                        if (codes[k]) partial += tof(A[(size_t)i * K + k]) * (float)codes[k];
                    }
                    acc += partial * tof(S[(size_t)g * N + j]);
                }
                C[(size_t)i * N + j] = acc;
            }
        }
    }
}

static void default_tiles(RunConfig &c) {
    if (c.sgm == 0) c.sgm = c.m == 1 ? 1 : (c.m <= 2 ? 2 : (c.m <= 4 ? 4 : 8));
    if (c.nsg == 0) c.nsg = 2;
    if (c.ls == 0) c.ls = 4;
    while (c.ls > 1 && c.k % (64 * c.ls)) c.ls /= 2;
}

static void list_tiles() {
    std::cout << "GEMV tiles (--sgm --nsg --ls):\n";
    for (auto &t : kGemvTiles)
        if (t.f[0]) std::cout << "  " << t.sgm << " " << t.nsg << " " << t.ls << "\n";
    std::cout << "M-tiled tiles (--mt-m --mt-n --wg-m --wg-n):\n";
    for (auto &t : kMtTiles) std::cout << "  " << t.mt_m << " " << t.mt_n << " " << t.wg_m << " " << t.wg_n << "\n";
    std::cout << "scale apply: default visa-hmul (fp16), --int-apply, --simt-mul; bf16 always int-and\n";
}

static void run(RunConfig cfg) {
    const int M = cfg.m, N = cfg.n, K = cfg.k;
    const bool mt = cfg.mt_m > 0;
    default_tiles(cfg);
    if (K % kGS || N % 16 || (!mt && K % (64 * cfg.ls))) {
        std::cerr << "need K % 128 == 0, N % 16 == 0 and (GEMV) K % (64 * LS) == 0\n";
        std::exit(1);
    }
    const bool bf = dt_is_bf16();
    const int ap = cfg.int_apply || bf ? bitcos::INT : cfg.simt_mul ? bitcos::SIMT : bitcos::VISA;
    const int v = variant(bf, ap);

    LaunchFn launch = nullptr;
    if (mt) {
        for (auto &t : kMtTiles)
            if (t.mt_m == cfg.mt_m && t.mt_n == cfg.mt_n && t.wg_m == cfg.wg_m && t.wg_n == cfg.wg_n) launch = t.f[v];
    } else {
        for (auto &t : kGemvTiles)
            if (t.sgm == cfg.sgm && t.nsg == cfg.nsg && t.ls == cfg.ls) launch = t.f[v];
    }
    if (!launch) { std::cerr << "tile not compiled in (see --list-tiles)\n"; std::exit(1); }

    sycl::queue q = make_queue();
    std::cout << "Problem: M=" << M << " N=" << N << " K=" << K << " scale_gs=" << kGS << " z=" << cfg.z
              << " dtype=" << dt_name() << " epilogue=" << cfg.epi.name() << "\n";
    if (mt)
        std::cout << "Tile (mt): sg " << cfg.mt_m << "x" << cfg.mt_n << ", wg " << cfg.wg_m << "x" << cfg.wg_n
                  << " sub-groups, 256 GRF";
    else
        std::cout << "Tile: sg_m=" << cfg.sgm << " wg_n=" << 16 * cfg.nsg << " ls=" << cfg.ls;
    std::cout << " apply=" << (ap == bitcos::INT ? "int-and" : ap == bitcos::SIMT ? "simt-hmul" : "visa-hmul") << "\n";

    const Epilogue &ep = cfg.epi;
    const size_t os = ep.out_size();
    const size_t size_a = (size_t)M * K, size_c = (size_t)M * N, size_s = (size_t)(K / kGS) * N;
    static std::mt19937 egen(std::random_device{}());
    std::vector<dt16> A(size_a), S(size_s), Oth;
    std::vector<unsigned char> Bias, Ch(size_c * os);
    Bitcos B;
    std::vector<uint32_t> SR;
    uint64_t seed = std::random_device{}();
    const int ls = mt ? 1 : cfg.ls;
    auto fill = [&] {
#pragma omp parallel for
        for (size_t i = 0; i < size_a; ++i) A[i] = fromf(rnd_f(-5.0f, 5.0f));
        for (size_t i = 0; i < size_s; ++i) S[i] = fromf(rnd_f(0.0f, 15.0f) + 0.75f);
        B = bitcos_random(K, N, cfg.z, seed++);
        SR = bitcos_slice_ranks(B, ls);
        if (SR.empty()) SR.assign(1, 0);
        fill_epilogue_inputs(ep, Oth, Bias, M, N, egen);
    };
    fill();
    const double wbytes_set = (double)B.bytes() + size_s * 2.0;
    const int sets = cfg.num_sets > 0 ? cfg.num_sets
                                      : std::max(1, (int)std::ceil(cfg.weights_gib * (1 << 30) / wbytes_set));
    std::cout << "BITCOS: measured z " << std::setprecision(4) << bitcos_zero_density(B) << ", "
              << 8.0 * B.bytes() / ((double)K * N) << " bits/weight (+scales); weights "
              << wbytes_set / (1 << 20) << " MiB/set, " << sets << " distinct sets -> " << std::fixed
              << std::setprecision(3) << wbytes_set * sets / (1 << 30) << " GiB\n";

    const int ngold = (cfg.validate && cfg.distinct_sets) ? sets : 1;
    std::vector<std::vector<unsigned char>> gold(cfg.validate ? ngold : 0);
    std::vector<float> acc_gold;
    auto make_gold = [&](std::vector<unsigned char> &g) {
        acc_gold.resize(size_c);
        compute_gold(A.data(), B, S.data(), acc_gold.data(), M);
        g.resize(size_c * os);
        epilogue_ref(ep, acc_gold.data(), Oth.data(), Bias.data(), g.data(), M, N);
    };
    if (cfg.validate) make_gold(gold[0]);

    std::vector<Args> args(sets);
    std::vector<size_t> bbytes(sets);
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
        bbytes[s] = B.bytes();
        a.A = (const dt16 *)dev_copy(A.data(), size_a * 2);
        a.B = (const uint32_t *)dev_copy(B.buf.data(), B.bytes());
        a.S = (const dt16 *)dev_copy(S.data(), size_s * 2);
        a.SR = (const uint32_t *)dev_copy(SR.data(), SR.size() * 4);
        a.C = dev_copy(nullptr, size_c * os);
        q.memset(a.C, 0, size_c * os);
        a.epi.other = (const unsigned short *)dev_copy(Oth.data(), Oth.size() * 2);
        a.epi.bias = dev_copy(Bias.data(), Bias.size());
        a.epi.postop = ep.postop;
        a.epi.f32 = ep.out_f32;
        a.M = M, a.N = N, a.K = K;
        q.wait();
    }

    double dev_ns = 0.0, host_ms = 0.0, bsum = 0.0;
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
            bsum += (double)bbytes[s];
            ++timed;
        }
    }
    if (timed) {
        const double dms = dev_ns / 1e6 / timed, hms = host_ms / timed;
        // actual bytes: A + BITCOS buffer + scales + C (+ epilogue operands)
        const double bytes = (double)M * K * 2 + bsum / timed + (double)(K / kGS) * N * 2
                + (double)M * N * os + (ep.needs_other() ? (double)M * N * 2 : 0.0)
                + (ep.needs_bias() ? (double)N * os : 0.0);
        // int2-equivalent bytes (2 bits/weight), the throughput at the same weight count
        const double bytes_i2 = bytes - bsum / timed + (double)K * N / 4.0;
        const double gib = 1024.0 * 1024.0 * 1024.0, flop = 2.0 * M * N * K;
        std::cout << std::fixed << std::setprecision(5)
                  << "Avg host  time: " << hms << " ms (" << flop / (hms * 1e-3) / 1e9 << " GFLOPS)\n"
                  << "Avg dev   time: " << dms << " ms (" << flop / (dms * 1e-3) / 1e9
                  << " GFLOPS, " << bytes / (dms * 1e-3) / gib << " GiB/s, int2-equiv "
                  << bytes_i2 / (dms * 1e-3) / gib << " GiB/s)\n"
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
        else if (a == "--z" && i + 1 < argc) cfg.z = std::atof(argv[++i]);
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
        else if (a == "--int-apply") cfg.int_apply = true;
        else if (a == "--simt-mul") cfg.simt_mul = true;
        else if (a == "--sgm" && i + 1 < argc) cfg.sgm = ival(i);
        else if (a == "--nsg" && i + 1 < argc) cfg.nsg = ival(i);
        else if (a == "--wgn" && i + 1 < argc) cfg.nsg = ival(i) / 16;
        else if (a == "--ls" && i + 1 < argc) cfg.ls = ival(i);
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
                      << " [--m M] [--n N] [--k K] [--z Z] [--dtype fp16|bf16] [--iters N] [--no-validate]\n"
                         "       [--distinct-sets] [--sets N | --weights-gib G] [--sgm 1|2|4|8] [--wgn W | --nsg S]\n"
                         "       [--ls L] [--int-apply | --simt-mul] [--mt-m 8k --mt-n 16k --wg-m W --wg-n W]\n"
                         "       [--list-tiles] [--postop 0|1|2|3|4] [--out-f32]\n";
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }
    run(cfg);
    return 0;
}

// Validation + timing driver for hadamard_fwht.hpp: device output must match a
// host fp32 reference with the same butterfly order bit for bit.
//   ./build/hadamard_sycl [--rows R] [--k K] [--dtype fp16|bf16] [--no-signs] [--iters N]

#include "hadamard_fwht.hpp"

#include "driver.hpp"
#include "dt16.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <vector>

static void ref_block(const dt16 *x, const int8_t *s, dt16 *y, bool inv) {
    const int8_t *si = inv ? s : nullptr;
    if (inv) s = nullptr;
    float v[1024];
    for (int i = 0; i < 1024; ++i) v[i] = tof(x[i]) * (s ? (float)s[i] : 1.0f);
    // stages 1..8 within 8, then strides 8 and 64 (radix-8 passes), then 512
    auto radix8 = [&](int stride) {
        for (int b = 0; b < 1024; ++b) {
            if ((b / stride) % 8) continue;
            float t[8];
            for (int j = 0; j < 8; ++j) t[j] = v[b + j * stride];
            hadamard::wht8(t);
            for (int j = 0; j < 8; ++j) v[b + j * stride] = t[j];
        }
    };
    radix8(1);
    radix8(8);
    radix8(64);
    for (int i = 0; i < 512; ++i) {
        const float a = v[i], b = v[i + 512];
        float lo = (a + b) * (1.0f / 32.0f), hi = (a - b) * (1.0f / 32.0f);
        if (si) {
            lo *= (float)si[i];
            hi *= (float)si[i + 512];
        }
        y[i] = fromf(lo);
        y[i + 512] = fromf(hi);
    }
}

int main(int argc, char **argv) {
    int rows = 1024, K = 5120, iters = 20;
    bool use_signs = true, inv = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (a == "--k" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--dtype" && i + 1 < argc) dt_is_bf16() = std::string(argv[++i]) == "bf16";
        else if (a == "--no-signs") use_signs = false;
        else if (a == "--inverse") inv = true;
        else {
            std::cout << "Usage: " << argv[0] << " [--rows R] [--k K] [--dtype fp16|bf16] [--no-signs] [--inverse] [--iters N]\n";
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }
    if (K % 1024) { std::cerr << "need K % 1024 == 0\n"; return 1; }
    sycl::queue q = make_queue();
    std::cout << "rows=" << rows << " K=" << K << " dtype=" << dt_name() << " signs=" << use_signs
              << " inverse=" << inv << "\n";

    const size_t n = (size_t)rows * K;
    std::vector<dt16> x(n), y(n), g(n);
    std::vector<int8_t> s(K);
    for (auto &v : x) v = fromf(rnd_f(-5.0f, 5.0f));
    for (auto &v : s) v = (rnd_u32() & 1) ? 1 : -1;
    const size_t bpr = K / 1024;
    for (size_t b = 0; b < (size_t)rows * bpr; ++b) {
        const size_t off = (b / bpr) * K + (b % bpr) * 1024;
        ref_block(x.data() + off, use_signs ? s.data() + (b % bpr) * 1024 : nullptr, g.data() + off, inv);
    }

    auto *dx = sycl::malloc_device<dt16>(n, q), *dy = sycl::malloc_device<dt16>(n, q);
    auto *ds = sycl::malloc_device<int8_t>(K, q);
    q.memcpy(dx, x.data(), n * 2).wait();
    q.memcpy(ds, s.data(), K).wait();
    const sycl::nd_range<1> r((size_t)rows * bpr * 128, 128);
    const signed char *sp = use_signs ? (const signed char *)ds : nullptr;
    auto launch = [&] {
        return dt_is_bf16() ? q.parallel_for(r, hadamard::Fwht1024<true>{dx, sp, dy, K, inv})
                            : q.parallel_for(r, hadamard::Fwht1024<false>{dx, sp, dy, K, inv});
    };
    launch().wait();
    double ns = 0.0;
    for (int it = 0; it < iters; ++it) {
        sycl::event e = launch();
        e.wait();
        ns += ev_ns(e);
    }
    q.memcpy(y.data(), dy, n * 2).wait();
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) bad += y[i] != g[i];
    const double us = ns / iters / 1e3;
    std::cout << std::fixed << std::setprecision(3) << "Avg dev   time: " << us / 1e3 << " ms ("
              << 4.0 * n / (us * 1e-6) / (1 << 30) << " GiB/s)\n"
              << "bit mismatches vs host: " << bad << "/" << n << "\n"
              << "Validation summary: " << (bad ? "FAILED" : "1/1 sets PASSED") << "\n";
    sycl::free(dx, q);
    sycl::free(dy, q);
    sycl::free(ds, q);
    return bad != 0;
}

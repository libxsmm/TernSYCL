// Host helpers shared by the drivers: SYCL queue, event timing, random inputs.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <iostream>
#include <random>
#include <string>

inline sycl::queue make_queue() {
    sycl::queue q{sycl::gpu_selector_v,
            {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    return q;
}

inline double ev_ns(const sycl::event &e) {
    return (double)(e.get_profiling_info<sycl::info::event_profiling::command_end>()
            - e.get_profiling_info<sycl::info::event_profiling::command_start>());
}

inline uint32_t rnd_u32() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return std::uniform_int_distribution<uint32_t>(0, UINT32_MAX)(gen);
}
inline float rnd_f(float lo, float hi) {
    static thread_local std::mt19937 gen(std::random_device{}());
    return std::uniform_real_distribution<float>(lo, hi)(gen);
}
// random ternary word: 16 codes in {0, 1, 3} = {0, +1, -1}
inline uint32_t rnd_ternary_word() {
    const uint32_t r1 = rnd_u32(), r2 = rnd_u32();
    const uint32_t lo = r1 & 0x55555555u, hi = (r1 & r2) & 0x55555555u;
    return lo | (hi << 1);
}
inline int code_to_value(uint32_t c) { return (int)(int8_t)((int8_t)(c << 6) >> 6); }

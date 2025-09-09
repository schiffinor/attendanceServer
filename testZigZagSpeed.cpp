//
// Created by schif on 9/4/2025.
//
#include <cassert>
#include <iostream>
#include <utility>
#include <cstddef>
#include <cstdint>

static std::ptrdiff_t computeZigzagOffset(std::size_t idx, std::size_t off) noexcept {
    using S = std::make_signed_t<std::size_t>;
    const S i   = static_cast<S>(idx);
    const S o   = static_cast<S>(off);
    const S pos = (i + 2 * o + 1) / 2;    // floor((i + 2*off + 1)/2)
    return (i & 1u) ? -pos : pos;
}

static std::pair<std::ptrdiff_t, std::ptrdiff_t>
computeZigzagOffsetPair_orig(std::size_t nth, std::size_t l_cnt, std::size_t r_cnt) noexcept {
    return { computeZigzagOffset(2 * nth,     l_cnt),
            computeZigzagOffset(2 * nth + 1, r_cnt) };
}

static std::pair<std::ptrdiff_t, std::ptrdiff_t>
computeZigzagOffsetPair_fast(std::size_t idx, std::size_t lcnt, std::size_t rcnt) noexcept {
    using S = std::ptrdiff_t;
    return { static_cast<S>(idx + lcnt),
            -static_cast<S>(idx + rcnt + 1) };
}

#include <chrono>
int main() {
    //SUPER AVGS
    int cnter = 0;
    double total_orig = 0.0;
    double total_fast = 0.0;
    double avg_avg_orig = 0.0;
    double avg_avg_fast = 0.0;
    for (std::size_t i = 0; i < 2000; ++i) {
        ++cnter;
        // time keeper variables
        // compare original and fast versions
        int cnt = 0;
        double sum_orig = 0.0;
        double sum_fast = 0.0;
        double avg_orig = 0.0;
        double avg_fast = 0.0;
        // Test a modest but broad range
        for (std::size_t nth = 0; nth < 1000; ++nth) {
            for (std::size_t l = 0; l < 100; ++l) {
                for (std::size_t r = 0; r < 100; ++r) {
                    ++cnt;
                    // lets time these to measure speed
                    auto start = std::chrono::high_resolution_clock::now();
                    auto a = computeZigzagOffsetPair_orig(nth, l, r);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::micro> elapsed_orig = end - start;
                    sum_orig += elapsed_orig.count();

                    auto start2 = std::chrono::high_resolution_clock::now();
                    auto b = computeZigzagOffsetPair_fast(nth, l, r); // idx == nth
                    auto end2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::micro> elapsed2 = end2 - start2;
                    sum_fast += elapsed2.count();

                    if (a != b) {
                        std::cerr << "Mismatch at nth=" << nth
                                  << " l=" << l << " r=" << r
                                  << " orig=(" << a.first << "," << a.second << ")"
                                  << " fast=(" << b.first << "," << b.second << ")\n";
                        return 1;
                    }
                }
            }
        }
        std::cout << "All good: pairs are identical over tested ranges.\n";
        avg_orig = sum_orig / cnt;
        avg_fast = sum_fast / cnt;
        std::cout << "Original total time: " << sum_orig << " us over "<< cnt << " calls\n";
        std::cout << "Fast total time:     " << sum_fast << " us over "<< cnt << " calls\n";
        std::cout << "\n";
        std::cout << "Original avg time: " << avg_orig << " us over "<< cnt << " calls\n";
        std::cout << "Fast avg time:     " << avg_fast << " us over "<< cnt << " calls\n";
        std::cout << "Speedup: " << (avg_orig / avg_fast) << "x\n";
        std::cout << "Time_diff: " << (avg_orig - avg_fast) << " us\n";
        total_orig += sum_orig;
        total_fast += sum_fast;
        avg_avg_orig += avg_orig;
        avg_avg_fast += avg_fast;
    }
    std::cout << "\n\nSUPER AVGS over " << cnter << " runs:\n";
    std::cout << "Original total time: " << total_orig << " us\n";
    std::cout << "Fast total time:     " << total_fast << " us\n";
    std::cout << "\n";
    std::cout << "Overall Speedup: " << (total_orig / total_fast) << "x\n";
    std::cout << "Overall Time_diff: " << (total_orig - total_fast) << " us\n";
    std::cout << "\n";
    std::cout << "Original avg time: " << (avg_avg_orig / cnter) << " us\n";
    std::cout << "Fast avg time:     " << (avg_avg_fast / cnter) << " us\n";
    std::cout << "Avg Speedup: " << (avg_avg_orig / avg_avg_fast) << "x\n";
    std::cout << "Avg Time_diff: " << ((avg_avg_orig / cnter) - (avg_avg_fast / cnter)) << " us\n";



    return 0;
}

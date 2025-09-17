//
// Created by schif on 9/16/2025.
//

#ifndef ATTENDANCESERVER_CPU_HPP
#define ATTENDANCESERVER_CPU_HPP

#include <cstdint>

namespace simd {

enum class Level : std::int32_t {
    Scalar = 0,
    SSE42  = 1,
    AVX2   = 2,
};

struct Caps {
    bool sse42 {false};
    bool avx2 {false};
    bool avx512f {false};
    bool os_xmm {false}; // XMM state enabled by OS (XCR0[1] == 1)
    bool os_ymm {false}; // YMM state enabled by OS (XCR0[2] == 1)
    bool os_zmm {false}; // ZMM state enabled by OS (XCR0[7:5] == 111)
};

/**
 * @brief Returns the cached CPU/OS SIMD capability bits. Thread-safe, runs detection once.
 *
 * @return const Caps&
 */
const Caps &cpu_caps() noexcept;

/**
 * @brief Highest usable SIMD level for kernels (based on cpu_caps()).
 *
 * @return Level
 */
Level active_level() noexcept;

/**
 * @brief Test hook to override the detected level (e.g., SIMDX_FORCE=avx2)
 *
 * @return void
 */
void force_level(Level lvl) noexcept;

// ----- convenience helpers -----
[[nodiscard]]
inline bool has(Level lvl) noexcept {
    return active_level() == lvl;
}

[[nodiscard]]
inline bool at_least(Level lvl) noexcept {
    const auto a = active_level();
    return static_cast<int>(a) >= static_cast<int>(lvl);
}

} // namespace simd

#endif // ATTENDANCESERVER_CPU_HPP

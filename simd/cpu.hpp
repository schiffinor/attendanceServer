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

enum class ADX_Level : std::int32_t {
    None = 0,
    ADX  = 1,
};

struct Caps {
    bool sse42 {false};
    bool avx2 {false};
    bool avx512f {false};
    bool os_xmm {false}; // XMM state enabled by OS (XCR0[1] == 1)
    bool os_ymm {false}; // YMM state enabled by OS (XCR0[2] == 1)
    bool os_zmm {false}; // ZMM state enabled by OS (XCR0[7:5] == 111)

    bool adx {false};   // EBX.ADX
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
 * @brief Returns whether ADX instructions are supported by CPU/OS
 *
 * @return ADX_Level
 */
ADX_Level active_adx_level() noexcept;

/**
 * @brief Test hook to override the detected level (e.g., SIMDX_FORCE=avx2)
 *
 * @return void
 */
void force_level(Level lvl) noexcept;

/**
 * @brief Test hook to override the detected ADX support (e.g., ADX_FORCE=1)
 *
 * @return void
 */
void force_adx_level(ADX_Level lvl) noexcept;

// ----- convenience helpers -----

// Cheap accessor for ADX support
/**
 * @brief Check if ADX is enabled
 * @return true if ADX is enabled, false otherwise
 */
[[nodiscard]]
inline bool adx_enabled() noexcept {
    return cpu_caps().adx;
}
/**
 * @brief Check if the active level is exactly `lvl`
 * @param lvl level to check
 * @return true if the active level is exactly `lvl`, false otherwise
 */
[[nodiscard]]
inline bool has(const Level lvl) noexcept {
    return active_level() == lvl;
}

/**
 * @brief Check if ADX is supported
 * @return true if ADX is supported, false otherwise
 */
[[nodiscard]]
inline bool has_adx() noexcept {
    return active_adx_level() == ADX_Level::ADX;
}

/**
 * @brief Check if the active level is at least `lvl`
 * @param lvl level to check
 * @return true if the active level is at least `lvl`, false otherwise
 */
[[nodiscard]]
inline bool at_least(Level lvl) noexcept {
    const auto a = active_level();
    return static_cast<int>(a) >= static_cast<int>(lvl);
}

// Realistically this is pointless as its equivalent to has_adx() but for symmetry...
/**
 * @brief Check if ADX is supported
 * @return true if ADX is supported, false otherwise
 */
[[nodiscard]]
inline bool at_least_adx() noexcept {
    return active_adx_level() == ADX_Level::ADX;
}

} // namespace simd

#endif // ATTENDANCESERVER_CPU_HPP

//
// Created by schif on 9/16/2025.
//

#include <atomic>
#include <cpuid.h> // GCC/Clang on x86/x64
#include <cstdint>
#include <cstdlib> // getenv (optional)
#include <cstring>
#include <mutex>

#include "cpu.hpp"

namespace simd {

namespace {
    /**
     * @brief Wrapper for xgetbv assembly instruction. Reads the value of an extended control register (XCR).
     *
     * @param index The index of the XCR to read (default is 0 for XCR0).
     * @return std::uint64_t The value of the specified XCR.
     */
    std::uint64_t xgetbv(std::uint32_t index = 0) noexcept {
        std::uint32_t eax, edx;
        // xgetbv: opcode 0F 01 D0
        __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(index));
        return (static_cast<std::uint64_t>(edx) << 32) | eax;
    }

    Caps detect_caps() noexcept {
        Caps caps {};

        std::uint32_t eax {}, ebx {}, ecx {}, edx {};
        if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
            return caps;

        caps.sse42 = (ecx & (1u << 20)) != 0; // ECX.SSE4.2

        // AVX requires OS support for XSAVE/XRESTOR and AVX registers
        if ((ecx & (1u << 27)) != 0) {
            const std::uint64_t xcr0 = xgetbv(0);
            caps.os_xmm              = (xcr0 & (1ull << 1)) != 0;
            caps.os_ymm              = (xcr0 & (1ull << 2)) != 0;
            // AVX-512 requires ZMM state enabled by OS (bits 7:5)
            caps.os_zmm = (xcr0 & (1ull << 5)) && (xcr0 & (1ull << 6)) && (xcr0 & (1ull << 7));
        }

        // Leaf 7, sub-leaf 0: AVX2 / AVX-512 feature bits.
        if (const std::uint32_t max_leaf = __get_cpuid_max(0, nullptr); max_leaf >= 7) {
            __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
            const bool cpu_avx2    = (ebx & (1u << 5)) != 0;  // EBX.AVX2
            const bool cpu_avx512f = (ebx & (1u << 16)) != 0; // EBX.AVX512F
            caps.avx2              = cpu_avx2 && caps.os_xmm && caps.os_ymm;
            caps.avx512f           = cpu_avx512f && caps.os_xmm && caps.os_ymm && caps.os_zmm;
        }

        return caps;
    }

    // Override for testing at runtime
    auto forced = static_cast<Level>(-1);

    const Caps &cached_caps() noexcept {
        static const Caps caps = [] {
            Caps _caps = detect_caps();
            // Optional override via environment variable SIMDX_FORCE
            if (const char *env = std::getenv("SIMDX_FORCE")) {
                if (std::strcmp(env, "avx2") == 0) {
                } else if (std::strcmp(env, "sse42") == 0) {
                    _caps.avx2 = false;
                } else if (std::strcmp(env, "scalar") == 0 || std::strcmp(env, "none") == 0) {
                    _caps = {}; // disable all
                }
            };
            return _caps;
        }();
        return caps;
    }
    // Forced override (unset means "no override")
    enum class OptLevel : std::int32_t { Unset = -1 };
    std::atomic g_forced{static_cast<int32_t>(OptLevel::Unset)};

    // Cached effective level (computed once if not forced)
    std::once_flag g_level_once;

    auto g_cached_level = Level::Scalar; // default; will be set on first use

    Level compute_level_from_caps(const Caps& c) noexcept {
        if (c.avx2)  return Level::AVX2;
        if (c.sse42) return Level::SSE42;
        return Level::Scalar;
    }

}

/**
 * @brief Highest usable SIMD level for kernels (based on cpu_caps()).
 *
 * @return
 */
const Caps &cpu_caps() noexcept { return cached_caps(); }

Level active_level() noexcept {
    const std::int32_t f = g_forced.load(std::memory_order_acquire);
    if (f != static_cast<std::int32_t>(OptLevel::Unset)) {
        return static_cast<Level>(f);
    }
    std::call_once(g_level_once, [] {
        g_cached_level = compute_level_from_caps(cached_caps());
    });
    return g_cached_level;
}

/**
 * @brief Test hook to override the detected level (e.g., SIMDX_FORCE=avx2)
 *
 * @param lvl
 */
void force_level(Level lvl) noexcept {
    g_forced.store(static_cast<std::int32_t>(lvl), std::memory_order_release);
}

} // namespace simd

/**
 *
 */

#ifndef POLY_HPP
#define POLY_HPP

#pragma once

#include <algorithm>
#include <array>
#include <cpuid.h>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <numeric>
#include <type_traits>

// put these once near your includes
#include <cinttypes>
#include <cstdio>

/**
 *
 */
namespace simd {
using u64 = std::uint64_t;

static constexpr unsigned char operator""_uc(const unsigned long long arg) noexcept {
    return static_cast<unsigned char>(arg);
}

static constexpr char operator""_c(const unsigned long long arg) noexcept { return static_cast<char>(arg); }

static constexpr long long operator""_ll(const unsigned long long arg) noexcept { return static_cast<long long>(arg); }

//
inline void cpuid(int out[4], int leaf, int sub = 0) noexcept {
    unsigned eax, ebx, ecx, edx;
    __cpuid_count(leaf, sub, eax, ebx, ecx, edx);
    out[0] = eax;
    out[1] = ebx;
    out[2] = ecx;
    out[3] = edx;
}

enum class Level : int { SSE42 = 0, AVX2 = 1 };

inline Level detect() noexcept {
    int regs[4];
    cpuid(regs, 0);
    if (regs[0] < 7) {
        return Level::SSE42; // No AVX2 support
    }

    cpuid(regs, 7, 0);
    const bool avx2_supported = regs[1] & (1 << 5);
    return avx2_supported ? Level::AVX2 : Level::SSE42;
}

inline Level active() noexcept {
    static const Level level = detect();
    return level;
}

template<std::size_t N>
struct alignas(32) PolyBase {
    using value_type                  = std::uint16_t;
    static constexpr std::size_t size = N;

    std::array<value_type, N> v; // 32-byte aligned thanks to alignas(32)

    value_type &operator[](std::size_t i) noexcept { return v[i]; }

    value_type operator[](std::size_t i) const noexcept { return v[i]; }
};

namespace debug {
#if defined(__AVX2__)
    static void print_u64x4(const char *lbl, const __m256i v, const bool hex = false) {
        alignas(32) uint64_t t[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(t), v);
        const char *frmt_string = hex ? "%s u64x4 : [%016llx | %016llx | %016llx | %016llx]\n"
                                      : "%s u64x4 : [%llu | %llu | %llu | %llu]\n";
        std::printf(frmt_string,
                    lbl,
                    static_cast<unsigned long long>(t[0]),
                    static_cast<unsigned long long>(t[1]),
                    static_cast<unsigned long long>(t[2]),
                    static_cast<unsigned long long>(t[3]));
    }

    static void print_s64x4(const char *lbl, const __m256i v) {
        alignas(32) int64_t t[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(t), v);
        const char *frmt_string = "%s s64x4 : [%lld | %lld | %lld | %lld]\n";
        std::printf(frmt_string,
                    lbl,
                    static_cast<long long>(t[0]),
                    static_cast<long long>(t[1]),
                    static_cast<long long>(t[2]),
                    static_cast<long long>(t[3]));
    }

    static void print_u32x8_256(const char *lbl, const __m256i v, const bool hex = false) {
        alignas(32) uint32_t t[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(t), v);
        const char *frmt_string = hex ? "%s u32x8 : [%08x %08x %08x %08x | %08x %08x %08x %08x]\n"
                                        : "%s u32x8 : [%u %u %u %u | %u %u %u %u]\n";
        std::printf(frmt_string,
                    lbl,
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    t[5],
                    t[6],
                    t[7]);
    }
#endif

#if defined(__SSE4_2__) || defined(__AVX2__)
    static void print_u32x4_128(const char *lbl, const __m128i v, const bool hex = false) {
        alignas(16) uint32_t t[4];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(t), v);
        const char *frmt_string = hex ? "%s u32x4 : [%08x %08x %08x %08x]\n" : "%s u32x4 : [%u %u %u %u]\n";
        std::printf(frmt_string, lbl, t[0], t[1], t[2], t[3]);
    }
#endif

#if defined(__AVX2__)
    static void print_u16x16(const char *lbl, const __m256i v, const bool hex = false) {
        alignas(32) uint16_t t[16];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(t), v);
        const char *frmt_string = hex ? "%s u16x16 : [%04x %04x %04x %04x | %04x %04x %04x %04x | "
                                        "%04x %04x %04x %04x | %04x %04x %04x %04x]\n"
                                      : "%s u16x16 : [%u %u %u %u | %u %u %u %u | "
                                        "%u %u %u %u | %u %u %u %u]\n";
        std::printf(frmt_string,
                    lbl,
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    t[5],
                    t[6],
                    t[7],
                    t[8],
                    t[9],
                    t[10],
                    t[11],
                    t[12],
                    t[13],
                    t[14],
                    t[15]);
    }
#endif

#if defined(__AVX2__) || defined(__SSE4_2__)
    // ReSharper disable once CppDFAConstantParameter
    static void print_u16x8_128(const char *lbl, const __m128i v, const bool hex = false) {
        alignas(16) uint16_t t[8];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(t), v);
        const char *frmt_string = hex ? "%s u16x8 : [%04x %04x %04x %04x | %04x %04x %04x %04x]\n"
                                      : "%s u16x8 : [%u %u %u %u | %u %u %u %u]\n";
        std::printf(frmt_string,
                    lbl,
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    t[5],
                    t[6],
                    t[7]);
    }

    static void print_s64x2(const char *lbl, const __m128i v, const bool hex = false) {
        alignas(16) int64_t t[2];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(t), v);
        std::printf("%s s64x2 : [%lld | %lld]\n",
                    lbl,
                    static_cast<long long>(t[0]),
                    static_cast<long long>(t[1]));
    }

    static void print_u64x2(const char *lbl, const __m128i v, const bool hex = false) {
        alignas(16) uint64_t t[2];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(t), v);
        const char *frmt_string = hex ? "%s u64x2 : [%016llx | %016llx]\n" : "%s u64x2 : [%llu | %llu]\n";
        std::printf(frmt_string,
                    lbl,
                    static_cast<unsigned long long>(t[0]),
                    static_cast<unsigned long long>(t[1]));
    }


#endif

} // namespace debug

namespace detail {

    /** Compile-time Barrett reduction for polynomial coefficients.
     *  This is used to reduce the coefficients of a polynomial modulo a prime.
     *  The prime must be a 32-bit integer.
     */

    template<std::uint32_t Q>
    [[nodiscard]]
    constexpr std::uint32_t log_2_q() {
        return static_cast<std::uint32_t>(32 - __builtin_clz(Q));
    }

    template<std::uint32_t Q>
    constexpr std::uint32_t LOG2_Q = log_2_q<Q>();

    template<std::uint32_t Q>
    [[nodiscard]]
    constexpr std::uint32_t barrett_mu() {
        return static_cast<uint32_t>((1ull << 32) / Q);
    }

    template<std::uint32_t Q>
    [[nodiscard]]
    constexpr std::uint64_t barrett_mu64() {
        return (static_cast<unsigned __int128>(1) << 64) / Q;
    }

    template<std::uint32_t Q>
    constexpr std::uint32_t BARRETT_MU = barrett_mu<Q>();

    template<std::uint32_t Q>
    constexpr std::uint64_t BARRETT_MU64 = barrett_mu64<Q>();

    template<std::uint32_t Q, std::uint32_t bit_width>
    [[nodiscard]]
    constexpr std::uint32_t neg_adj() {
        // no bit-width above 64 allowed
        static_assert(bit_width <= 64, "bit_width must be <= 64");
        // 2^bit_width mod Q
        constexpr unsigned __int128 maxVal = static_cast<unsigned __int128>(1) << bit_width;
        constexpr auto m                   = static_cast<std::uint32_t>(maxVal % static_cast<unsigned __int128>(Q));
        // -(2^bit_width) mod Q = (Q - (2^bit_width) % Q)) = (Q - m)
        // for efficiency reasons return m basically congruence and fitting in 64 bits
        return m;
    }

    template<std::uint32_t Q, std::uint32_t bit_width>
    constexpr std::uint32_t NEG_ADJ = neg_adj<Q, bit_width>();

    template<std::uint32_t Q>
    constexpr std::uint32_t NEG_ADJ_16 = neg_adj<Q, 16>();

    template<std::uint32_t Q>
    constexpr std::uint32_t NEG_ADJ_32 = neg_adj<Q, 32>();

    template<std::uint32_t Q>
    constexpr std::uint32_t NEG_ADJ_64 = neg_adj<Q, 64>();

    template<std::uint32_t Q>
    struct barrett_s64_consts {
        static constexpr u64 MU64       = detail::BARRETT_MU64<Q>;
        static constexpr u64 NEG_ADJ_64 = detail::NEG_ADJ_64<Q>;

#if defined(__AVX2__)
        // __m256i constants for AVX2 Signed 64 bit Barrett reduction
        alignas(32) static inline const __m256i zero_256    = _mm256_setzero_si256();
        alignas(32) static inline const __m256i neg_one_256 = _mm256_set1_epi64x(-1); // 64-bit lanes for one
        alignas(32) static inline const __m256i negAdj_256  = _mm256_set1_epi64x(NEG_ADJ_64);
        alignas(32) static inline const __m256i Q256_64     = _mm256_set1_epi64x(Q); // 64-bit lanes for Q
        alignas(32) static inline const __m256i MU256_64 = _mm256_set1_epi64x(MU64); // 64-bit lanes for multiplication
#endif

#if defined(__SSE4_2__)
        // __m128i constants for SSE4.2 Signed 64 bit Barrett reduction
        alignas(16) static inline const __m128i zero_128    = _mm_setzero_si128();
        alignas(16) static inline const __m128i neg_one_128 = _mm_set1_epi64x(-1); // 64-bit lanes for one
        alignas(16) static inline const __m128i negAdj_128  = _mm_set1_epi64x(NEG_ADJ_64);
        alignas(16) static inline const __m128i Q128_64     = _mm_set1_epi64x(Q);    // 64-bit lanes for Q
        alignas(16) static inline const __m128i MU128_64    = _mm_set1_epi64x(MU64); // 64-bit lanes for multiplication
        alignas(16) static inline const __m128i bit_shuffle_mask_1_128 =
                _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, 0x80_c, 0x80_c, 0x80_c, 0x80_c, 0x80_c, 0x80_c, 0x80_c, 0x80_c);
        alignas(16) static inline const __m128i bit_shuffle_mask_2_128 =
                _mm_setr_epi8(0, 1, 8, 9, 4, 5, 12, 13, 0x80_c, 0x80_c, 0x80_c, 0x80_c, 0x80_c, 0x80_c, 0x80_c, 0x80_c);
#endif
    };


    /**
     * @brief Barrett reduction for polynomial coefficients.
     * Constant-time w.r.t. value not secret.
     *
     * @param x The value to reduce.
     * @return The reduced value.
     */
    template<std::uint32_t Q>
    [[nodiscard]]
    constexpr std::uint16_t reduce(const std::uint32_t x) {
        const std::uint64_t t = (static_cast<std::uint64_t>(x) * detail::BARRETT_MU<Q>) >> 32;
        auto r                = static_cast<std::int32_t>(x - t * Q);
        r -= static_cast<std::int32_t>(Q);
        r += (r >> 31) & static_cast<std::int32_t>(Q); // Ensure r is non-negative
        return static_cast<std::uint16_t>(r);
    }

    template<std::uint32_t Q>
    [[nodiscard]]
    constexpr std::uint16_t barrett_s64(const std::int64_t x) {
        using u64  = std::uint64_t;
        using u128 = unsigned __int128;

        // 1. static cast x as unsigned, because bit pattern maintained this equiv to x mod 2^64
        const u64 ux      = static_cast<u64>(x);
        const u64 neg_adj = detail::NEG_ADJ_64<Q>;
        const u64 pre_mod = ux - (neg_adj & (x >> 63));
        // 2. Barrett reduction
        const u128 prod = static_cast<u128>(pre_mod) * static_cast<u128>(detail::BARRETT_MU64<Q>);
        const u64 t     = static_cast<u64>(prod >> 64);
        u64 r           = pre_mod - t * Q;
        // 3. Ensure r is in range [0, Q)
        // 5. branchless reduce: subtract Q when r>=Q, else subtract 0
        //    (bool(r>=Q) → 0/1, negate to 0 or 0xFFFF…; &Q → Q or 0)
        const u64 ge = r >= Q;
        const u64 m  = 0 - ge;
        r -= (Q & m);
        return static_cast<std::uint16_t>(r);
    }
} // namespace detail

namespace kernels {

    /* ---------------------------------------------------------------- */
    /* 1.  Constant vectors                                             */
    /* ---------------------------------------------------------------- */
    template<std::uint32_t Q>
    struct vec_consts16 {
        static constexpr std::uint16_t q16 = static_cast<std::uint16_t>(Q);

#if defined(__AVX2__)
        static inline const __m256i Q256 = _mm256_set1_epi16(q16);
#endif
        static inline const __m128i Q128 = _mm_set1_epi16(q16);
    };

    template<std::uint32_t Q>
    struct vec_consts32 {
        static constexpr std::uint32_t mu32 = detail::BARRETT_MU<Q>;

#if defined(__AVX2__)
        static inline const __m256i MU256 = _mm256_set1_epi32(mu32);
        static inline const __m256i Q256  = _mm256_set1_epi32(Q);
#endif
#if defined(__SSE4_2__)
        static inline const __m128i MU128 = _mm_set1_epi32(mu32);
        static inline const __m128i Q128  = _mm_set1_epi32(Q);
#endif
    };

#if defined(__AVX2__)
    static __m128i cvtepi64_epi32_avx(const __m256i v) {
        const __m256 vf = _mm256_castsi256_ps(v);       // free
        const __m128 hi = _mm256_extractf128_ps(vf, 1); // vextractf128
        const __m128 lo = _mm256_castps256_ps128(vf);   // also free
        // take the bottom 32 bits of each 64-bit chunk in lo and hi
        const __m128 packed = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0)); // shufps
        return _mm_castps_si128(packed);                                       // if you want
    }

    // Emulate _mm256_mulhi_epu64(a,b):
    //   for each 64-bit lane i:
    //     uint128_t prod = (uint128_t)A[i] * (uint128_t)B[i];
    //     result[i] = uint64_t(prod >> 64);
    static __m256i mm256_mulhi_epu64(const __m256i A, const __m256i B) {
        // mask for low 32 bits of each 64-bit lane
        const __m256i mask32   = _mm256_set1_epi64x(0xFFFFFFFFULL);
        const __m256i signbit  = _mm256_set1_epi64x(0x8000000000000000_ll);
        const __m256i one_lane = _mm256_set1_epi64x(1);

        // split A and B into high/low 32-bit halves
        const __m256i A_lo = _mm256_and_si256(A, mask32);
        const __m256i A_hi = _mm256_srli_epi64(A, 32);
        const __m256i B_lo = _mm256_and_si256(B, mask32);
        const __m256i B_hi = _mm256_srli_epi64(B, 32);

        // four partial products (each is 32×32→64 bits)
        const __m256i lo_lo = _mm256_mul_epu32(A_lo, B_lo); // low×low
        const __m256i lo_hi = _mm256_mul_epu32(A_lo, B_hi); // low×high
        const __m256i hi_lo = _mm256_mul_epu32(A_hi, B_lo); // high×low
        const __m256i hi_hi = _mm256_mul_epu32(A_hi, B_hi); // high×high

        // cross terms sum
        const __m256i cross = _mm256_add_epi64(hi_lo, lo_hi);
        // upper 32 bits of cross contribute to the high 64 bits
        const __m256i cross_hi = _mm256_srli_epi64(cross, 32);

        // but the low‐half of cross, shifted into bits [32..63], can overflow
        const __m256i cross_lo     = _mm256_and_si256(cross, mask32);
        const __m256i cross_lo_shl = _mm256_slli_epi64(cross_lo, 32);
        const __m256i sum_lo       = _mm256_add_epi64(lo_lo, cross_lo_shl);

        // detect unsigned carry: sum_lo < lo_lo ?
        //    sum_lo < lo_lo  ⟺  carry out of the low-64 addition
        const __m256i sum_lo_xor = _mm256_xor_si256(sum_lo, signbit);
        const __m256i lo_lo_xor  = _mm256_xor_si256(lo_lo, signbit);
        const __m256i cmp        = _mm256_cmpgt_epi64(lo_lo_xor, sum_lo_xor);
        const __m256i carry      = _mm256_and_si256(cmp, one_lane);

        // final high half = hi_hi + cross_hi + carry
        return _mm256_add_epi64(hi_hi, _mm256_add_epi64(cross_hi, carry));
    }

    // Emulate _mm256_mullo_epu64(a,b) on AVX2
    static __m256i mm256_mullo_epu64(const __m256i A, const __m256i B) noexcept {
        // mask to extract the low 32 bits of each 64-bit lane
        const __m256i mask32 = _mm256_set1_epi64x(0xFFFF'FFFFULL);

        // split A and B into their low/high 32-bit halves
        const __m256i A_lo = _mm256_and_si256(A, mask32);
        const __m256i A_hi = _mm256_srli_epi64(A, 32);
        const __m256i B_lo = _mm256_and_si256(B, mask32);
        const __m256i B_hi = _mm256_srli_epi64(B, 32);

        // compute the four partial 32×32→64 products
        const __m256i lo_lo = _mm256_mul_epu32(A_lo, B_lo); // low(A) × low(B)
        const __m256i lo_hi = _mm256_mul_epu32(A_lo, B_hi); // low(A) × high(B)
        const __m256i hi_lo = _mm256_mul_epu32(A_hi, B_lo); // high(A) × low(B)
        // high(A)×high(B) contributes only to bits ≥64, so IGNORE for the low 64 bits

        // sum the two cross‐terms
        const __m256i cross = _mm256_add_epi64(lo_hi, hi_lo);
        // keep only their low 32 bits and shift into the 64-bit high half
        const __m256i cross_lo     = _mm256_and_si256(cross, mask32);
        const __m256i cross_lo_shl = _mm256_slli_epi64(cross_lo, 32);

        // final low-64 = lo_lo + (cross_lo << 32)  (mod 2^64)
        return _mm256_add_epi64(lo_lo, cross_lo_shl);
    }
#endif

#if defined(__SSE4_2__)
    // Emulate _mm256_mulhi_epu64 on SSE: high 64 bits of each 64×64→128 product
    static __m128i mm_mulhi_epu64(const __m128i A, const __m128i B) noexcept {
        // mask for low 32 bits
        const __m128i mask32 = _mm_set1_epi64x(0xFFFF'ffffULL);
        const __m128i signbit = _mm_set1_epi64x(0x8000000000000000_ll);
        const __m128i one64  = _mm_set1_epi64x(1);

        // split each 64-bit lane into two 32-bit halves
        const __m128i A_lo = _mm_and_si128(A, mask32);
        const __m128i A_hi = _mm_srli_epi64(A, 32);
        const __m128i B_lo = _mm_and_si128(B, mask32);
        const __m128i B_hi = _mm_srli_epi64(B, 32);

        // four partial 32×32→64 multiplies
        const __m128i lo_lo = _mm_mul_epu32(A_lo, B_lo); // low×low
        const __m128i lo_hi = _mm_mul_epu32(A_lo, B_hi); // low×high
        const __m128i hi_lo = _mm_mul_epu32(A_hi, B_lo); // high×low
        const __m128i hi_hi = _mm_mul_epu32(A_hi, B_hi); // high×high

        // sum cross-terms (fits in 64 bits before shifting)
        const __m128i cross        = _mm_add_epi64(lo_hi, hi_lo);
        const __m128i cross_hi     = _mm_srli_epi64(cross, 32);
        const __m128i cross_lo     = _mm_and_si128(cross, mask32);
        const __m128i cross_lo_shl = _mm_slli_epi64(cross_lo, 32);

        // now compute low-half sum to detect carry
        const __m128i sum_lo = _mm_add_epi64(lo_lo, cross_lo_shl);
        // carry if lo_lo + … overflowed: sum_lo < lo_lo
        const __m128i lo_lo_xor  = _mm_xor_si128(lo_lo, signbit);
        const __m128i sum_lo_xor = _mm_xor_si128(sum_lo, signbit);
        const __m128i carry_mask = _mm_cmpgt_epi64(lo_lo_xor, sum_lo_xor);
        const __m128i carry      = _mm_and_si128(carry_mask, one64);

        // final high 64 bits = hi_hi + cross_hi + carry
        return _mm_add_epi64(hi_hi, _mm_add_epi64(cross_hi, carry));
    }

    // Emulate _mm256_mullo_epu64 on SSE: low  64 bits of each 64×64→128 product
    static __m128i mm_mullo_epu64(const __m128i A, const __m128i B) noexcept {
        const __m128i mask32 = _mm_set1_epi64x(0xFFFF'ffffULL);

        const __m128i A_lo = _mm_and_si128(A, mask32);
        const __m128i A_hi = _mm_srli_epi64(A, 32);
        const __m128i B_lo = _mm_and_si128(B, mask32);
        const __m128i B_hi = _mm_srli_epi64(B, 32);

        const __m128i lo_lo = _mm_mul_epu32(A_lo, B_lo);
        const __m128i lo_hi = _mm_mul_epu32(A_lo, B_hi);
        const __m128i hi_lo = _mm_mul_epu32(A_hi, B_lo);

        // cross terms contribute to bits [32..95]
        const __m128i cross    = _mm_add_epi64(lo_hi, hi_lo);
        const __m128i cross_lo = _mm_and_si128(cross, mask32);
        const __m128i cross_sh = _mm_slli_epi64(cross_lo, 32);

        // low half = lo_lo + (cross_lo << 32)
        return _mm_add_epi64(lo_lo, cross_sh);
    }
#endif


    /* ---------------------------------------------------------------- */
    /* 2.  Barrett reduction helpers                                    */
    /* ---------------------------------------------------------------- */
#if defined(__AVX2__)
    template<std::uint32_t Q>
    __m256i reduce8_avx2(const __m256i x32) noexcept {
        using C = vec_consts32<Q>; // Q256  and MU256

        /* -------- multiply: even + odd lanes separately ---------------- */
        __m256i prod_even     = _mm256_mul_epu32(x32, C::MU256);     // lanes 0 2 4 6 (64-bit)
        const __m256i x32_odd = _mm256_srli_epi64(x32, 32);          // move odd words to even pos
        __m256i prod_odd      = _mm256_mul_epu32(x32_odd, C::MU256); // lanes 0 2 4 6 (odd inputs)

        /* -------- keep the HIGH half of each 64-bit product ----------- */
        prod_even = _mm256_srli_epi64(prod_even, 32); // hi(even)
        prod_odd  = _mm256_srli_epi64(prod_odd, 32);  // hi(odd)

        /* -------- merge even/odd into one packed 32-bit vector -------- */
        const __m256i t = _mm256_blend_epi32(prod_even, _mm256_slli_epi64(prod_odd, 32), 0b10101010);
        /*  now  t = ⌊x·µ / 2³²⌋  for all 8 lanes                         */

        /* -------- one-subtract Barrett correction --------------------- */
        const __m256i r  = _mm256_sub_epi32(x32, _mm256_mullo_epi32(t, C::Q256));
        const __m256i r2 = _mm256_sub_epi32(r, C::Q256);
        const __m256i m  = _mm256_cmpgt_epi32(_mm256_setzero_si256(), r2); // mask (r2<0)
        return _mm256_blendv_epi8(r2, r, m);                               // exact remainder in [0,Q)
    }
#endif

#if defined(__SSE4_2__)
    inline __m128i unsigned_gt(const __m128i a, const __m128i b) { // SSE2 helper
        return _mm_cmpgt_epi32(_mm_xor_si128(b, _mm_set1_epi32(static_cast<int>(0x8000'0000))),
                               _mm_xor_si128(a, _mm_set1_epi32(static_cast<int>(0x8000'0000))));
    }


    template<std::uint32_t Q>
    __m128i reduce4_sse2(const __m128i x32) noexcept {
        using C      = vec_consts32<Q>;
        __m128i even = _mm_mul_epu32(x32, C::MU128); // lanes 0,2 → 64-bit
        __m128i odd  = _mm_mul_epu32(_mm_srli_epi64(x32, 32), C::MU128);

        even = _mm_srli_epi64(even, 32);
        odd  = _mm_srli_epi64(odd, 32);

        /* shuffle to interleave even/odd 32-bit halves */
        const __m128i t = _mm_or_si128(even, _mm_slli_epi64(odd, 32));

        const __m128i r  = _mm_sub_epi32(x32, _mm_mullo_epi32(t, C::Q128));
        const __m128i r2 = _mm_sub_epi32(r, C::Q128);
        const __m128i m  = unsigned_gt(r2, r); // keep r if r<Q
        return _mm_or_si128(_mm_and_si128(m, r), _mm_andnot_si128(m, r2));
    }
#endif

#if defined(__AVX2__)
    template<std::uint32_t Q, std::size_t N, bool DEBUG = false>
    void reduce_s64_avx2(const std::int64_t *in, std::uint16_t *out) noexcept {
        using C = detail::barrett_s64_consts<Q>;

        static_assert(N % 8 == 0, "N must be multiple of 8 for AVX2 reduce_s64");

        for (std::size_t i = 0; i < N; i += 8) {
            // 1. Load eight int64_t values as two 256-bit registers 4 lanes each
            const __m256i x_low  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));
            const __m256i x_high = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i + 4));

            // 2. Compute mask = (x < 0) ? 0xFFFFFFFFFFFFFFFF : 0
            const __m256i m_low  = _mm256_cmpgt_epi64(C::zero_256, x_low);  // mask for low lanes
            const __m256i m_high = _mm256_cmpgt_epi64(C::zero_256, x_high); // mask for high lanes

            // inputs and masks
            if constexpr (DEBUG) {
                debug::print_s64x4("x_low         ", x_low);
                debug::print_s64x4("x_high        ", x_high);
                debug::print_u64x4("m_low (mask)  ", m_low);
                debug::print_u64x4("m_high(mask)  ", m_high);
            }
            // 3. ux = bitcast(x) as unsigned 64-bit integers
            //    pre_mod = ux - (negAdj & mask)
            const __m256i p_low  = _mm256_sub_epi64(x_low, _mm256_and_si256(C::negAdj_256, m_low));
            const __m256i p_high = _mm256_sub_epi64(x_high, _mm256_and_si256(C::negAdj_256, m_high));

            // pre_mod
            if constexpr (DEBUG) {
                debug::print_u64x4("p_low         ", p_low);
                debug::print_u64x4("p_high        ", p_high);
            }

            // 4. Multiply by MU64 -> high 64 bits of 64*64=128 product

            const __m256i t64_low  = mm256_mulhi_epu64(p_low, C::MU256_64);
            const __m256i t64_high = mm256_mulhi_epu64(p_high, C::MU256_64);

            // 7. Compute raw remainder r = pre_mod - t * Q
            const __m256i tq_low  = mm256_mullo_epu64(t64_low, C::Q256_64);  // t * Q for low lanes
            const __m256i tq_high = mm256_mullo_epu64(t64_high, C::Q256_64); // t * Q for high lanes


            // t and t*Q
            if constexpr (DEBUG) {
                debug::print_u64x4("t64_low       ", t64_low);
                debug::print_u64x4("t64_high      ", t64_high);
                debug::print_u64x4("tq_low        ", tq_low);
                debug::print_u64x4("tq_high       ", tq_high);
            }

            const __m256i r_low  = _mm256_sub_epi64(p_low, tq_low);
            const __m256i r_high = _mm256_sub_epi64(p_high, tq_high);


            // Since there is no greater than or equal to comparison for 64-bit integers in AVX2,
            // we will use a trick:
            // We will switch the order of r and q in _mm256_cmpgt_epi64 and then negate the result.
            const __m256i gt_mask_low  = _mm256_cmpgt_epi64(C::Q256_64, r_low);  // r < Q
            const __m256i gt_mask_high = _mm256_cmpgt_epi64(C::Q256_64, r_high); // r < Q
            const __m256i f_mask_ge_low =
                    _mm256_xor_si256(gt_mask_low, C::neg_one_256); // 0x0000...0001 if r < Q, else 0
            const __m256i f_mask_ge_high =
                    _mm256_xor_si256(gt_mask_high, C::neg_one_256); // 0x0000...0001 if r < Q, else 0
            const __m256i a_mask_low =
                    _mm256_and_si256(C::Q256_64, f_mask_ge_low); // if r < Q, then a_mask_low = Q, else 0
            const __m256i a_mask_high =
                    _mm256_and_si256(C::Q256_64, f_mask_ge_high); // if r < Q, then a_mask_high = Q, else 0

            const __m256i r_final_low  = _mm256_sub_epi64(r_low, a_mask_low);
            const __m256i r_final_high = _mm256_sub_epi64(r_high, a_mask_high);

            // r and final r
            if constexpr (DEBUG) {
                debug::print_u64x4("r_low         ", r_low);
                debug::print_u64x4("r_high        ", r_high);
                debug::print_u64x4("r_final_low   ", r_final_low);
                debug::print_u64x4("r_final_high  ", r_final_high);
            }

            // 9. Pack the results into 16-bit integers
            // 9.a. Convert the 64-bit results to 32-bit integers
            const __m128i r32_low  = cvtepi64_epi32_avx(r_final_low);
            const __m128i r32_high = cvtepi64_epi32_avx(r_final_high);

            if constexpr (DEBUG) {
                debug::print_u32x4_128("r32_low       ", r32_low);
                debug::print_u32x4_128("r32_high      ", r32_high);
            }

            // 9.d) pack 4×32→8×16 and store
            const __m128i r16 = _mm_packus_epi32(r32_low, r32_high); // 8×u16

            if constexpr (DEBUG) {
                debug::print_u16x8_128("r16           ", r16);
            }

            _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), r16);
        }
    }

    template<std::uint32_t Q, std::size_t N>
    void reduce_s64_avx2_dbg(const int64_t *in, uint16_t *out) {
        reduce_s64_avx2<Q, N, true>(in, out);
    }
#endif

#if defined(__SSE4_2__)
    template<std::uint32_t Q, std::size_t N, bool DEBUG = false>
    void reduce_s64_sse(const std::int64_t *in, std::uint16_t *out) noexcept {
        using C = detail::barrett_s64_consts<Q>;

        static_assert(N % 4 == 0, "N must be multiple of 4 for SSE4.2 reduce_s64");

        for (std::size_t i = 0; i < N; i += 4) {
            // 1. Load four int64_t values as two 128-bit registers 2 lanes each
            const __m128i x_low  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
            const __m128i x_high = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i + 2));
            if constexpr (DEBUG) {
                debug::print_s64x2("x_low        ", x_low);
                debug::print_s64x2("x_high       ", x_high);
            }

            // 2. Compute mask = (x < 0) ? 0xFFFFFFFFFFFFFFFF : 0
            const __m128i m_low  = _mm_cmpgt_epi64(C::zero_128, x_low);  // mask for low lanes
            const __m128i m_high = _mm_cmpgt_epi64(C::zero_128, x_high); // mask for high lanes
            if constexpr (DEBUG) {
                debug::print_u64x2("m_low (mask) ", m_low);
                debug::print_u64x2("m_high(mask) ", m_high);
            }

            // 3. ux = bitcast(x) as unsigned 64-bit integers
            //    pre_mod = ux - (negAdj & mask)
            const __m128i p_low  = _mm_sub_epi64(x_low, _mm_and_si128(C::negAdj_128, m_low));
            const __m128i p_high = _mm_sub_epi64(x_high, _mm_and_si128(C::negAdj_128, m_high));
            if constexpr (DEBUG) {
                debug::print_u64x2("p_low        ", p_low);
                debug::print_u64x2("p_high       ", p_high);
            }

            // 4. Multiply by MU64 -> high 64 bits of 64*64=128 product

            const __m128i t64_low  = mm_mulhi_epu64(p_low, C::MU128_64);
            const __m128i t64_high = mm_mulhi_epu64(p_high, C::MU128_64);

            // 7. Compute raw remainder r = pre_mod - t * Q
            const __m128i tq_low  = mm_mullo_epu64(t64_low, C::Q128_64);  // t * Q for low lanes
            const __m128i tq_high = mm_mullo_epu64(t64_high, C::Q128_64); // t * Q for high lanes
            if constexpr (DEBUG) {
                debug::print_u64x2("t64_low      ", t64_low);
                debug::print_u64x2("t64_high     ", t64_high);
                debug::print_u64x2("tq_low       ", tq_low);
                debug::print_u64x2("tq_high      ", tq_high);
            }


            const __m128i r_low  = _mm_sub_epi64(p_low, tq_low);
            const __m128i r_high = _mm_sub_epi64(p_high, tq_high);
            if constexpr (DEBUG) {
                debug::print_u64x2("r_low        ", r_low);
                debug::print_u64x2("r_high       ", r_high);
            }

            // Since there is no greater than or equal to comparison for 64-bit integers in AVX2,
            // we will use a trick:
            // We will switch the order of r and q in _mm256_cmpgt_epi64 and then negate the result.
            const __m128i gt_mask_low   = _mm_cmpgt_epi64(C::Q128_64, r_low);         // r < Q
            const __m128i gt_mask_high  = _mm_cmpgt_epi64(C::Q128_64, r_high);        // r < Q
            const __m128i f_mask_ge_low = _mm_xor_si128(gt_mask_low, C::neg_one_128); // 0x0000...0000 if r < Q, else 0x1111...1111
            const __m128i f_mask_ge_high =
                    _mm_xor_si128(gt_mask_high, C::neg_one_128); // 0x0000...0000 if r < Q, else 0x1111...1111
            const __m128i a_mask_low =
                    _mm_and_si128(C::Q128_64, f_mask_ge_low); // if r < Q, then a_mask_low = Q, else 0
            const __m128i a_mask_high =
                    _mm_and_si128(C::Q128_64, f_mask_ge_high); // if r < Q, then a_mask_high = Q, else 0

            const __m128i r_final_low  = _mm_sub_epi64(r_low, a_mask_low);
            const __m128i r_final_high = _mm_sub_epi64(r_high, a_mask_high);
            if constexpr (DEBUG) {
                debug::print_u64x2("r_final_low  ", r_final_low);
                debug::print_u64x2("r_final_high ", r_final_high);
            }

            // 9. Pack the results into 16-bit integers
            // 9.a. Convert the 64-bit results to 32-bit integers
            // THIS MUST BE UNPACKLO TO GET THE RIGHT ORDER OTHERWISE COMPLETELY FALLS APART
            const __m128i r32_shuffled = _mm_unpacklo_epi32(_mm_shuffle_epi8(r_final_low, C::bit_shuffle_mask_1_128),
                                                            _mm_shuffle_epi8(r_final_high, C::bit_shuffle_mask_1_128));
            if constexpr (DEBUG) {
                debug::print_u32x4_128("r32_shuffled ", r32_shuffled);
            }
            const __m128i r32_trunc    = _mm_shuffle_epi8(r32_shuffled, C::bit_shuffle_mask_2_128);
            if constexpr (DEBUG) {
                debug::print_u32x4_128("r32_trunc    ", r32_trunc);
                debug::print_u16x8_128("r32_trunc as u16", r32_trunc);
            }
            // 9.d) pack 4×32→8×16 and store
            // This intrinsic stores only the lower 64 bits (4×16)
            _mm_storel_epi64(reinterpret_cast<__m128i *>(out + i), r32_trunc);
        }
    }

    template<std::uint32_t Q, std::size_t N>
    void reduce_s64_sse_dbg(const int64_t *in, uint16_t *out) {
        reduce_s64_sse<Q, N, true>(in, out);
    }
#endif


    /* Wrapper that reduces N uint32’s → uint16’s */
    template<std::uint32_t Q, std::size_t N>
    void reduce_vec(const std::uint32_t *in, std::uint16_t *out) noexcept {
#if defined(__AVX2__)
        if (active() == Level::AVX2 && (N % 8 == 0)) // stride = 8
        {
            for (std::size_t i = 0; i < N; i += 8) {
                /* load 8×u32 -------------------------------------------------- */
                __m256i v32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in + i));

                /* Barrett on 8 lanes ----------------------------------------- */
                const __m256i q32 = reduce8_avx2<Q>(v32); // 8×u32 remainders

                /* split 256→128+128, pack each dword to word ----------------- */
                const __m128i lo32 = _mm256_castsi256_si128(q32);      // lanes 0‥3
                const __m128i hi32 = _mm256_extracti128_si256(q32, 1); // lanes 4‥7
                const __m128i r16  = _mm_packus_epi32(lo32, hi32);     // 8×u16, correct order

                /* store all eight coefficients ------------------------------- */
                _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), r16);
            }
            return;
        }
#endif
        /* ---- SSE4.2 path (8 lanes) ---- */
#if defined(__SSE4_2__)
        if ((N % 4) == 0) {
            for (std::size_t i = 0; i < N; i += 4) {
                __m128i v         = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + i));
                const __m128i r32 = reduce4_sse2<Q>(v);
                const __m128i r16 = _mm_packus_epi32(r32, r32);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), r16);
            }
            return;
        }
#endif
        /* ---- Scalar fallback ---- */
        for (std::size_t i = 0; i < N; ++i)
            out[i] = detail::reduce<Q>(in[i]);
    }

    /* Wrapper that reduces N uint32’s → uint16’s */
    template<std::uint32_t Q, std::size_t N, bool DEBUG = false>
    void reduce_vec_s64(const std::int64_t *in, std::uint16_t *out) noexcept {
#if defined(__AVX2__)
        if (active() == Level::AVX2 && (N % 8) == 0) {
            kernels::reduce_s64_avx2<Q, N, DEBUG>(in, out);
            return;
        }
#endif
#if defined(__SSE4_2__)
        if ((N % 4) == 0) {
            kernels::reduce_s64_sse<Q, N, DEBUG>(in, out);
            return;
        }
#endif
        // scalar fallback
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = detail::barrett_s64<Q>(in[i]);
        }
    }

    /* ---------------------------------------------------------------- */
    /* 3.  Add-mod SIMD kernels                                         */
    /* ---------------------------------------------------------------- */
#if defined(__SSE4_2__)
    template<std::size_t N, std::uint32_t Q>
    void add_mod_sse4_2(const std::uint16_t *a, const std::uint16_t *b, std::uint16_t *r) noexcept {
        static_assert(N % 8 == 0, "N must be multiple of 8 for SSE4.2 add");
        const __m128i k = vec_consts16<Q>::Q128;

        for (std::size_t i = 0; i < N; i += 8) {
            const __m128i va   = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i));
            const __m128i vb   = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i));
            const __m128i sum  = _mm_add_epi16(va, vb);
            const __m128i tmp  = _mm_sub_epi16(sum, k);
            const __m128i mask = _mm_cmpgt_epi16(_mm_setzero_si128(), tmp);
            const __m128i res  = _mm_or_si128(_mm_and_si128(mask, sum), _mm_andnot_si128(mask, tmp));
            _mm_store_si128(reinterpret_cast<__m128i *>(r + i), res);
        }
    }
#endif

#if defined(__AVX2__)
    template<std::size_t N, std::uint32_t Q>
    void add_mod_avx2(const std::uint16_t *a, const std::uint16_t *b, std::uint16_t *r) noexcept {
        static_assert(N % 16 == 0, "N must be multiple of 16 for AVX2 add");
        const __m256i k = vec_consts16<Q>::Q256;

        for (std::size_t i = 0; i < N; i += 16) {
            const __m256i va   = _mm256_load_si256(reinterpret_cast<const __m256i *>(a + i));
            const __m256i vb   = _mm256_load_si256(reinterpret_cast<const __m256i *>(b + i));
            const __m256i sum  = _mm256_add_epi16(va, vb);
            const __m256i tmp  = _mm256_sub_epi16(sum, k);
            const __m256i mask = _mm256_cmpgt_epi16(_mm256_setzero_si256(), tmp);
            const __m256i res  = _mm256_or_si256(_mm256_and_si256(mask, sum), _mm256_andnot_si256(mask, tmp));
            _mm256_store_si256(reinterpret_cast<__m256i *>(r + i), res);
        }
    }
#endif

    /* ---------------------------------------------------------------- */
    /* 3.  Mult-mod SIMD kernels                                         */
    /* ---------------------------------------------------------------- */


} // namespace kernels

template<std::size_t N, std::uint32_t Q>
class Poly : public PolyBase<N> {
    using Base = PolyBase<N>;

public:
    using value_type = typename Base::value_type;

    // Constructors
    Poly() = default;
    explicit Poly(value_type x) noexcept { fill(x); }
    template<typename It>
    Poly(It first, It last) {
        assign(first, last);
    }

    void fill(value_type x) noexcept {
        const value_type reduced_value = detail::reduce<Q>(x);
        this->v.fill(reduced_value);
    }

    template<typename It>
    void assign(It first, It last) {
        std::size_t i = 0;
        for (; first != last && i < N; ++first, ++i) {
            this->v[i] = detail::reduce<Q>(*first);
        }
    }

    static void add(const Poly &a, const Poly &b, Poly &result) noexcept {
        using namespace kernels;
        if (active() == Level::AVX2) {
#if defined(__AVX2__)
            add_mod_avx2<N, Q>(a.v.data(), b.v.data(), result.v.data()); // AVX2 path
#else
#if defined(__SSE4_2__)
            add_mod_sse4_2<N, Q>(a.v.data(), b.v.data(), result.v.data()); // fallback for non-AVX2
#else
            add_schoolbook(a, b, result); // fallback for non-SSE4.2 or AVX2
#endif
#endif
        } else if (active() == Level::SSE42) {
#if defined(__SSE4_2__)
            add_mod_sse4_2<N, Q>(a.v.data(), b.v.data(), result.v.data());
#else
            add_schoolbook(a, b, result); // fallback for non-SSE4.2
#endif
        } else {
            add_schoolbook(a, b, result);
        }
    }

    static void add_schoolbook(const Poly &a, const Poly &b, Poly &result) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            result.v[i] = detail::reduce<Q>(static_cast<std::uint32_t>(a.v[i] + b.v[i]));
        }
    }


    /* ------------------------------------------------------------------ */
    /*  O(N²) school-book multiply (reference / test path)                */
    /* ------------------------------------------------------------------ */
    static void mult_schoolbook(const Poly &a, const Poly &b, Poly &out, const bool debug = false) noexcept {
        constexpr std::size_t log_n = std::countr_zero(N); // N power–of-two

        std::array<std::int64_t, N> accumulator {}; // signed

        /* 1.  accumulate with per-step Barrett                                    */
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                std::size_t k = (i + j) & (N - 1); // wrap index
                // print values of i, j, k
                if (debug) std::cout << "\ni: " << i << ", j: " << j << ", k: " << k;
                const std::int32_t s = 1 - static_cast<std::int32_t>((i + j) >> log_n) * 2; // ±1
                // print value of s
                if (debug) std::cout << "\ns: " << s;

                // print values of a.v[i] and b.v[j]
                if (debug) std::cout << "\na.v[" << i << "]: " << a.v[i];
                if (debug) std::cout << "\nb.v[" << j << "]: " << b.v[j];
                const std::int64_t prod = s * static_cast<std::int64_t>(a.v[i]) * static_cast<std::int64_t>(b.v[j]);

                // check value of prod is in range [-Q, Q)
                if (debug) std::cout << "\nprod: " << prod;
                // check value of acc[k] is in range [-Q, Q)
                if (debug) std::cout << "\nacc[" << k << "]: " << accumulator[k];
                accumulator[k] = accumulator[k] + static_cast<std::int64_t>(prod);
                // accumulator[k] = detail::barrett_s64<Q>(static_cast<int64_t>(accumulator[k]) + prod); // |acc| < Q
                //  check value of acc[k] is in range [-Q, Q)
                if (debug) std::cout << "\nacc[" << k << "] after reduce: " << accumulator[k];
            }
        };

        // check value of acc[k] is in range [-Q, Q)
        if (debug)
            std::cout << "\naccumulator[k] values: ";
        if (debug)
            for (const auto &v : accumulator) {
                std::cout << v << " ";
            }

        /* 3.  SIMD Barrett (inputs already < Q, so this is just a copy)         */
        kernels::reduce_vec_s64<Q, N>(accumulator.data(), out.v.data());
    }
};
} // namespace simd


#endif // POLY_HPP

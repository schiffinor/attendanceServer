//
// Created by schif on 9/17/2025.
//

#ifndef ATTENDANCESERVER_BIG_INT_HPP
#define ATTENDANCESERVER_BIG_INT_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <type_traits>

// Config: Enable runtime masks (vs. only consteval masks)
// 0 means off, should be set to 0 for maximum optimization, will be 1 temp for testing and dev
#ifndef BIGINT_ENABLE_RUNTIME_MASKS
#define BIGINT_ENABLE_RUNTIME_MASKS 1
#endif
namespace endian {
    enum class Endianness {
        Little,
        Big
    };

    // Function to determine the system's endianness at runtime
    consteval Endianness get_system_endianness() {
        std::uint16_t number = 0x1;
        const auto numPtr   = std::bit_cast<char *>(&number);
        return (numPtr[0] == 1) ? Endianness::Little : Endianness::Big;
    }
} // namespace endian

namespace bigint {

// ----- Limb Requirements -----

/**
 * @brief A concept that checks if a type is an unsigned integer type (excluding bool).
 *
 * This concept ensures that the type T is an integral type, is unsigned, and is not bool.
 * This is important for types used as "limbs" in big integer implementations, as they need to
 * represent non-negative integer values.
 *
 * @tparam T The type to be checked.
 */
template<class T>
concept UnsignedLimb = std::is_integral_v<T> && std::is_unsigned_v<T> && !std::is_same_v<T, bool>;

/**
 * @brief A concept that checks if a type is an unsigned integer type with a size that is a multiple of 8 bits (1 byte).
 *
 * This concept ensures that the type T is an unsigned integer and that its size in bits is a multiple of 8.
 * This is important for types used as "limbs" in big integer implementations, as they need to be byte-aligned
 * for efficient memory access and operations.
 *
 * As a side note, I understand that ∀ x ∈ ℕ, x*8 % 8 == 0, but this concept is more about explicitly stating
 * the requirement for clarity and intent in the code, rather than relying on mathematical tautologies.
 *
 * @tparam T The type to be checked.
 */
template<class T>
concept NibAlignedUnsignedLimb = UnsignedLimb<T> && ((8 * sizeof(T)) % 8 == 0);

// This is just a shorthand for the above concept. It's just a short alias.
/**
 * Just an alias for @see NibAlignedUnsignedLimb
 * @brief A concept that checks if a type is an unsigned integer type with a size that is a multiple of 8 bits (1 byte).
 *
 * This concept ensures that the type T is an unsigned integer and that its size in bits is a multiple of 8.
 * This is important for types used as "limbs" in big integer implementations, as they need to be byte-aligned
 * for efficient memory access and operations.
 *
 * As a side note, I understand that ∀ x ∈ ℕ, x*8 % 8 == 0, but this concept is more about explicitly stating
 * the requirement for clarity and intent in the code, rather than relying on mathematical tautologies.
 *
 * @tparam T The type to be checked.
 */
template<class T>
concept NAULimb = NibAlignedUnsignedLimb<T>;

namespace limb_utils {
    // ----- Helper functions -----

    // ---------- Bit and Byte Width of Limb Types ----------

    /**
     * @brief Get the byte width of a given Limb type at compile time.
     *
     * @tparam Limb class of limb, should be an unsigned integer type, but not enforced
     * @return std::size_t byte width of the Limb type
     */
    template<NAULimb Limb>
    consteval std::size_t byte_width() noexcept {
        return sizeof(Limb);
    }

    /**
     * @brief Get the bit width of a given Limb type at compile time.
     *
     * @tparam Limb class of limb, should be an unsigned integer type, but not enforced
     * @return std::size_t bit width of the Limb type
     */
    template<NAULimb Limb>
    consteval std::size_t bit_width() noexcept {
        return byte_width<Limb>() * 8;
    }

    // ---------- MSB_pos ----------

    /**
     * @brief Get the position of the most significant bit (MSB) in a given Limb type at compile time.
     * The position is 0-based, so for an 8-bit Limb, the MSB is at position 7.
     *
     * @tparam Limb class of limb, should be an unsigned integer type, but not enforced
     * @return std::size_t position of the MSB (0-based)
     */
    template<NAULimb Limb>
    consteval std::size_t msb_pos() noexcept {
        return bit_width<Limb>() - 1;
    }
} // namespace limb_utils

template<class T>
inline constexpr std::size_t limb_bits_v = limb_utils::bit_width<T>();

template<class T>
inline constexpr std::size_t limb_bytes_v = limb_utils::byte_width<T>();

// ----- BigInt Requirements -----

/**
 * @brief A concept that checks if a Limb type and bit size are suitable for use in a BigInt implementation.
 *
 * This concept ensures that the Limb type is an unsigned integer type with a size that is a multiple of 8 bits,
 * and that the specified number of bits (NumBits) meets certain criteria:
 * - NumBits must be at least 8.
 * - NumBits must be greater than or equal to the bit width of the Limb type.
 * - NumBits must be a multiple of the bit width of the Limb type.
 * - NumBits must be a power of two.
 *
 * These requirements are important for ensuring efficient memory usage and operations in big integer implementations.
 *
 * @tparam NumBits The total number of bits for the big integer.
 * @tparam Limb The type used as a limb in the big integer representation.
 */
template<std::size_t NumBits, class Limb>
concept BigIntReq = NAULimb<Limb> &&                      // Limb is an uint with size multiple of 8 bits
                    (NumBits >= 8) &&                     // minimum 1 byte
                    (NumBits >= limb_bits_v<Limb>) &&     // at least one limb
                    (NumBits % limb_bits_v<Limb> == 0) && // whole number of limbs
                    ((NumBits & (NumBits - 1)) == 0);     // power of two

namespace limb_utils::mask_bases {
    // ----- Various Common Bit Masks -----

    // ---------- Canonical Masks ----------

    /**
     * @brief All-zeros value for a given Limb type
     *
     * @tparam Limb class of limb, should be an unsigned integer type, but not enforced
     * @return Limb with all bits zero
     */
    template<NAULimb Limb>
    consteval Limb all_zeros() noexcept {
        return Limb {0};
    }

    /**
     * @brief All-ones value for a given Limb type
     *
     * @tparam Limb class of limb, should be an unsigned integer type, but not enforced
     * @return Limb with all bits one
     */
    template<NAULimb Limb>
    consteval Limb all_ones() noexcept {
        return ~Limb {0};
    }

    // ---------- Single-bit / Range-based Masks (UB-safe) ----------

    /**
     * @brief Get a Limb with only the k-th bit set, or all-zeros if k is out of range.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @param k index of the bit to set (0-based)
     * @return Limb with only the k-th bit set, or all-zeros if k is out of range
     */
    template<NAULimb Limb>
    constexpr Limb kth_bit(const std::size_t k) noexcept {
        return (k >= bit_width<Limb>()) ? all_zeros<Limb>() : (Limb {1} << k);
    }

    /**
     * @brief Get a Limb with the lowest k bits set, or all-ones if k is out of range.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @param k number of lowest bits to set
     * @return Limb with the lowest k bits set, or all-ones if k is out of range
     */
    template<NAULimb Limb>
    constexpr Limb k_lowest_bits(const std::size_t k) noexcept {
        const std::size_t W = bit_width<Limb>();
        if (k == 0) {
            return all_zeros<Limb>();
        }
        if (k >= W) {
            return all_ones<Limb>();
        }
        return (Limb {1} << k) - Limb {1};
    }

    /**
     * @brief Get a Limb with the highest k bits set, or all-ones if k is out of range.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @param k number of highest bits to set (0-based)
     * @return Limb with the highest k bits set, or all-ones if k is out of range
     */
    template<NAULimb Limb>
    constexpr Limb k_highest_bits(const std::size_t k) noexcept {
        const std::size_t W = bit_width<Limb>();
        if (k == 0) {
            return all_zeros<Limb>();
        }
        if (k >= W) {
            return all_ones<Limb>();
        }
        return static_cast<Limb>(~k_lowest_bits<Limb>(W - k));
    }

    /**
     * @brief Get a Limb with only the most significant bit set.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with only the most significant bit set
     */
    template<NAULimb Limb>
    consteval Limb most_significant_bit() noexcept {
        return kth_bit<Limb>(msb_pos<Limb>());
    }

    /**
     * @brief Get a Limb with only the least significant bit set.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with only the least significant bit set
     */
    template<NAULimb Limb>
    consteval Limb least_significant_bit() noexcept {
        return kth_bit<Limb>(0);
    }

    // ---------- Repeated Patterns ----------

    /**
     * @brief Get a Limb with the pattern 0x11111111... (4-bit nibble repeated) across the entire width of the Limb.
     * (e.g. 0x11111111 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x1 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_repunit() noexcept {
        const Limb rep = (~Limb {0}) / 0xF; // e.g. 0x11111111 for 32-bit Limb
        return rep;
    }

    /**
     * @brief Repeat a 4-bit nibble across the entire width of a Limb type. (e.g. 0xA → 0xAAAAAAAA... for
     * 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @param nibble 4-bit value to repeat, can be any value, only the lowest 4 bits are used, must be const
     * unsigned 32 bit int
     * @return Limb with repeated nibble
     */
    template<NAULimb Limb>
    constexpr Limb repeat_nibble(const std::uint32_t nibble) noexcept {
        return Limb {nibble & 0xF} * hex_repunit<Limb>();
    }

    /**
     * @brief Canonical alternating bit mask (…1010) across the whole Limb
     *
     * @details Get a Limb with the pattern 0xAAAAAAAA... (4-bit nibble 0xA repeated) across the entire width of the
     * Limb. (e.g. 0xAAAAAAAA for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0xA nibble
     */
    template<NAULimb Limb>
    consteval Limb alt_10_mask() noexcept {
        return repeat_nibble<Limb>(0xA);
    }

    /**
     * @brief Canonical alternating bit mask (…0101) across the whole Limb
     *
     * @details Get a Limb with the pattern 0x55555555... (4-bit nibble 0x5 repeated) across the entire width of the
     * Limb. (e.g. 0x55555555 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x5 nibble
     */
    template<NAULimb Limb>
    consteval Limb alt_01_mask() noexcept {
        return repeat_nibble<Limb>(0x5);
    }

    // ---------- Hexadecimal Repeated Nibbles ----------

    /**
     * @brief Get a Limb with the pattern 0x00000000... (4-bit nibble 0x0 repeated) across the entire width of the
     * Limb. (e.g. 0x00000000 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x0 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_0() noexcept { // 0x00000000... 0b00000000...
        return all_zeros<Limb>();
    }

    /**
     * @brief Get a Limb with the pattern 0x11111111... (4-bit nibble 0x1 repeated) across the entire width of the
     * Limb. (e.g. 0x11111111 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x1 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_1() noexcept { // 0x11111111... 0b00010001...
        return hex_repunit<Limb>();
    }

    /**
     * @brief Get a Limb with the pattern 0x22222222... (4-bit nibble 0x2 repeated) across the entire width of the
     * Limb. (e.g. 0x22222222 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x2 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_2() noexcept { // 0x22222222... 0b00100010...
        return repeat_nibble<Limb>(0x2);
    }

    /**
     * @brief Get a Limb with the pattern 0x33333333... (4-bit nibble 0x3 repeated) across the entire width of the
     * Limb. (e.g. 0x33333333 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x3 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_3() noexcept { // 0x33333333... 0b00110011...
        return repeat_nibble<Limb>(0x3);
    }

    /**
     * @brief Get a Limb with the pattern 0x44444444... (4-bit nibble 0x4 repeated) across the entire width of the
     * Limb. (e.g. 0x44444444 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x4 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_4() noexcept { // 0x44444444... 0b01000100...
        return repeat_nibble<Limb>(0x4);
    }

    /**
     * @brief Get a Limb with the pattern 0x55555555... (4-bit nibble 0x5 repeated) across the entire width of the
     * Limb. (e.g. 0x55555555 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x5 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_5() noexcept { // 0x55555555... 0b01010101...
        return repeat_nibble<Limb>(0x5);
    }

    /**
     * @brief Get a Limb with the pattern 0x66666666... (4-bit nibble 0x6 repeated) across the entire width of the
     * Limb. (e.g. 0x66666666 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x6 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_6() noexcept { // 0x66666666... 0b01100110...
        return repeat_nibble<Limb>(0x6);
    }

    /**
     * @brief Get a Limb with the pattern 0x77777777... (4-bit nibble 0x7 repeated) across the entire width of the
     * Limb. (e.g. 0x77777777 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x7 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_7() noexcept { // 0x77777777... 0b01110111...
        return repeat_nibble<Limb>(0x7);
    }

    /**
     * @brief Get a Limb with the pattern 0x88888888... (4-bit nibble 0x8 repeated) across the entire width of the
     * Limb. (e.g. 0x88888888 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x8 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_8() noexcept { // 0x88888888... 0b10001000...
        return repeat_nibble<Limb>(0x8);
    }

    /**
     * @brief Get a Limb with the pattern 0x99999999... (4-bit nibble 0x9 repeated) across the entire width of the
     * Limb. (e.g. 0x99999999 for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0x9 nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_9() noexcept { // 0x99999999... 0b10011001...
        return repeat_nibble<Limb>(0x9);
    }

    /**
     * @brief Get a Limb with the pattern 0xAAAAAAAA... (4-bit nibble 0xA repeated) across the entire width of the
     * Limb. (e.g. 0xAAAAAAAA for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0xA nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_A() noexcept { // 0xAAAAAAAA... 0b10101010...
        return alt_10_mask<Limb>();
    }

    /**
     * @brief Get a Limb with the pattern 0xBBBBBBBB... (4-bit nibble 0xB repeated) across the entire width of the
     * Limb. (e.g. 0xBBBBBBBB for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0xB nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_B() noexcept { // 0xBBBBBBBB... 0b10111011...
        return repeat_nibble<Limb>(0xB);
    }

    /**
     * @brief Get a Limb with the pattern 0xCCCCCCCC... (4-bit nibble 0xC repeated) across the entire width of the
     * Limb. (e.g. 0xCCCCCCCC for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0xC nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_C() noexcept { // 0xCCCCCCCC... 0b11001100...
        return repeat_nibble<Limb>(0xC);
    }

    /**
     * @brief Get a Limb with the pattern 0xDDDDDDDD... (4-bit nibble 0xD repeated) across the entire width of the
     * Limb. (e.g. 0xDDDDDDDD for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0xD nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_D() noexcept { // 0xDDDDDDDD... 0b11011101...
        return repeat_nibble<Limb>(0xD);
    }

    /**
     * @brief Get a Limb with the pattern 0xEEEEEEEE... (4-bit nibble 0xE repeated) across the entire width of the
     * Limb. (e.g. 0xEEEEEEEE for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0xE nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_E() noexcept { // 0xEEEEEEEE... 0b11101110...
        return repeat_nibble<Limb>(0xE);
    }

    /**
     * @brief Get a Limb with the pattern 0xFFFFFFFF... (4-bit nibble 0xF repeated) across the entire width of the
     * Limb. (e.g. 0xFFFFFFFF for 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @return Limb with repeated 0xF nibble
     */
    template<NAULimb Limb>
    consteval Limb hex_rep_F() noexcept { // 0xFFFFFFFF... 0b11111111...
        return repeat_nibble<Limb>(0xF);
    }

} // namespace limb_utils::mask_bases

namespace limb_utils::masks {
    // The only difference between this and mask_bases is that k dependent masks and repeat_nibble are consteval.
    // Thus, we will just re-expose that namespace and override those functions with consteval versions.
    using namespace mask_bases;

    // ----- Various Common Bit Masks -----
    // ---------- Single-bit / Range-based Masks (UB-safe) ----------

    /**
     * @brief Get a Limb with only the k-th bit set, or all-zeros if k is out of range.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @param k index of the bit to set (0-based)
     * @return Limb with only the k-th bit set, or all-zeros if k is out of range
     */
    template<NAULimb Limb>
    consteval Limb kth_bit(const std::size_t k) noexcept {
        return mask_bases::kth_bit<Limb>(k);
    }

    /**
     * @brief Get a Limb with the lowest k bits set, or all-ones if k is out of range.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @param k number of lowest bits to set
     * @return Limb with the lowest k bits set, or all-ones if k is out of range
     */
    template<NAULimb Limb>
    consteval Limb k_lowest_bits(const std::size_t k) noexcept {
        return mask_bases::k_lowest_bits<Limb>(k);
    }

    /**
     * @brief Get a Limb with the highest k bits set, or all-ones if k is out of range.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @param k number of highest bits to set (0-based)
     * @return Limb with the highest k bits set, or all-ones if k is out of range
     */
    template<NAULimb Limb>
    consteval Limb k_highest_bits(const std::size_t k) noexcept {
        return mask_bases::k_highest_bits<Limb>(k);
    }


    // ---------- Repeated Patterns ----------

    /**
     * @brief Repeat a 4-bit nibble across the entire width of a Limb type. (e.g. 0xA → 0xAAAAAAAA... for
     * 32-bit Limb)
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @param nibble 4-bit value to repeat, can be any value, only the lowest 4 bits are used, must be const
     * unsigned 32 bit int
     * @return Limb with repeated nibble
     */
    template<NAULimb Limb>
    consteval Limb repeat_nibble(const std::uint32_t nibble) noexcept {
        return mask_bases::repeat_nibble<Limb>(nibble);
    }

} // namespace limb_utils::masks

// ARE YOU SURE YOU NEED THESE?
// Can you not use the compile time versions above?
// If you can use a compile time constant for k, use the compile time versions above.
// They're much faster and better optimized by the compiler.
#if BIGINT_ENABLE_RUNTIME_MASKS
namespace limb_utils::masks_rt {
    // Here we would really just forward every function directly to mask_bases thus letsjust expose it as an alias.
    using namespace limb_utils::mask_bases;
} // namespace limb_utils::masks_rt
#endif

template<bool IsSigned, std::size_t NumBits, NAULimb Limb>
    requires BigIntReq<NumBits, Limb>
struct big_traits {

    // ----- Important constants -----
    static constexpr bool is_signed        = IsSigned;
    static constexpr std::size_t nBits     = NumBits;
    static constexpr std::size_t nLimbBits = limb_bits_v<Limb>;
    static constexpr std::size_t nLimbs    = NumBits / limb_bits_v<Limb>;

    // ----- Type aliases -----
    using limb_type  = Limb;
    using array_type = std::array<Limb, nLimbs>;

    // ----- Indexes of various kinds -----
    static constexpr std::size_t msb_index   = nBits - 1;
    static constexpr std::size_t msb_limb    = nLimbs - 1;
    static constexpr std::size_t msb_bit_pos = limb_utils::msb_pos<Limb>();

    // ----- Masks of various kinds -----
    // ---------- Limb-level ----------
    // ---------- MSB-based ----------

    /**
     * @brief Mask with only the most significant bit (MSB) set, e.g. 0x80000000 / 0b1000...0000 for 32-bit Limb
     *
     * @return Limb with only MSB set
     */
    static consteval Limb msb_limb_mask() noexcept {
        return limb_utils::masks::most_significant_bit<Limb>();
    }

    /**
     * @brief Mask with all but the most significant bit (MSB) set, e.g. 0x7FFFFFFF / 0b0111...1111 for 32-bit Limb
     *
     * @return Limb with all but MSB set
     */
    static consteval Limb lower_limb_mask() noexcept {
        return ~msb_limb_mask();
    }

    /**
     * @brief Mask with only the most significant bit (MSB) set, e.g. 0x80000000 / 0b1000...0000 for 32-bit Limb
     *
     * Alias for msb_limb_mask()
     *
     * @return Limb with only MSB set
     */
    static consteval Limb upper_limb_mask() noexcept {
        return msb_limb_mask();
    }

    // ---------- Canonical Masks ----------

    /**
     * @brief Mask with all bits set to 0
     *
     * @return Limb with all bits 0
     */
    static consteval Limb all_zeros() noexcept {
        return Limb {0};
    }

    /**
     * @brief Mask with all bits set to 1
     *
     * @return Limb with all bits 1
     */
    static consteval Limb all_ones() noexcept {
        return ~all_zeros();
    }

    // ---------- Dynamic Bit Masks ----------
    // These are not allowed to use runtime values, only compile-time constants
    // If you need runtime values, use the functions in limb_utils::masks_rt, not recommended for performance reasons

    /**
     * @brief k-th bit mask (0-based, 0 = least significant bit) (e.g. k=31 → 0x80000000 for 32-bit Limb)
     *
     * @tparam K bit index
     * @return Limb with only k-th bit set, or 0 if k is out of range
     */
    template<std::size_t K>
    static consteval Limb kth_bit_mask() noexcept {
        return limb_utils::masks::kth_bit<Limb>(K);
    }

    /**
     * @brief Lower-k-bits mask (handles k==0 and k==bitwidth without UB)
     *
     * @tparam K number of lower bits to set
     * @return Limb with lower k bits set, or all bits 0 if k==0, or all bits 1 if k >= bitwidth
     */
    template<std::size_t K>
    static consteval Limb mask_lo_bits() noexcept {
        return limb_utils::masks::k_lowest_bits<Limb>(K);
    }

    /**
     * @brief Upper-k-bits mask
     *
     * @tparam K number of upper bits to set
     * @return Limb with upper k bits set, or all bits 0 if k==0, or all bits 1 if k >= bitwidth
     */
    template<std::size_t K>
    static consteval Limb mask_hi_bits() noexcept {
        return limb_utils::masks::k_highest_bits<Limb>(K);
    }
};

// Alias namespace if you want to use simd::bigint
namespace simd {
    using namespace ::bigint;
}

namespace detail {



    /**
     * @brief Pair of indexes: limb index and bit index within that limb
     *
     * @details Corresponds to a single bit index in a big integer represented as an array of limbs.
     * The limb index is 0-based, with 0 being the least significant limb.
     * The bit index is also 0-based, with 0 being the least significant bit within the limb
     */
    struct index_pair {
        std::size_t limb;
        // bit could technically be uint8_t for all standard int types,
        // but technically if someone were to define a custom uint512_t type,
        // with std::is_integral_v and std::is_unsigned_v true,
        // it could be larger than 255 bits, so we use size_t for safety.
        // Actually, we will only allow 8 bits to be used for bit index,
        // as half of everything breaks down if we allow more than 8 bits.
        // Mainly, if we allow types with more than 128 bits, then
        // none of the builtins from <bit> will work, and we would have to
        // implement our own versions of countl_zero, countr_zero, popcount, etc.
        // So we will just limit bit index to 0-255, which is more than enough for any reasonable limb size.
        // This also allows us to use uint8_t for bit index, which is more memory efficient.
        // If we ever need to support limbs with more than 256 bits, we can always change this later.
        // Also, technically speaking it is best practice to define bit before limb,
        // as it is of smaller size, but for clarity we will define limb first.
        // Also, it literally makes no difference in practice as there are only two members.
        std::uint8_t bit;
    };

    /**
     * @brief Internal helper functions that map our logical indices to physical indices based on limb endianness.
     *
     * Everything should be endian agnostic, so these functions help with that.
     * Also, they shouldn't even be concerned with the actual endianness of the system,
     * as the limbs are always stored in little-endian order in memory and these operate on ints not bytes.
     * The only thing that matters is the logical endianness of the limbs.
     */
    namespace order_conv {

        // ----- Endianness Conversion -----
        template<NAULimb Limb, std::size_t NumLimbs>
        constexpr index_pair bit_to_index_pair(const std::size_t bit_index) noexcept {
            if (bit_index >= NumLimbs * limb_bits_v<Limb>) {
                return index_pair {0, 0}; // out of range
            }
            if (bit_index == 0) {
                return index_pair {0, 0}; // special case for zero
            }
            if (bit_index < limb_bits_v<Limb>) {
                return index_pair {0, static_cast<std::uint8_t>(bit_index)}; // special case for first limb
            }
            // general case
            const std::size_t limb_index = bit_index / limb_bits_v<Limb>;
            const std::size_t bit_pos    = bit_index % limb_bits_v<Limb>;
            return index_pair {limb_index, static_cast<std::uint8_t>(bit_pos)};
        }

        template<NAULimb Limb, std::size_t NumLimbs>
        constexpr std::size_t index_pair_to_bit(const index_pair &idx) noexcept {
            if (idx.limb >= NumLimbs || idx.bit >= limb_bits_v<Limb>) {
                return 0; // out of range
            }
            return idx.limb * limb_bits_v<Limb> + idx.bit;
        }


    } // namespace order_conv

    /**
     * @brief index of most significant bit set or unset, depends only on endianness.
     *
     * @tparam Limb class of limb, must be an unsigned integer type
     * @tparam NumLimbs number of limbs in the array
     * @return
     */
    template<NAULimb Limb, std::size_t NumLimbs>
    consteval std::size_t msb_index() noexcept {
        return NumLimbs * limb_bits_v<Limb> - 1;
    }


    template<NAULimb Limb, std::size_t NumLimbs>
    consteval index_pair msb_index_pair() noexcept {
        return index_pair {NumLimbs - 1, limb_bits_v<Limb> - 1};
    }

    template<NAULimb Limb, std::size_t NumLimbs>
    constexpr bool is_zero(const std::array<Limb, NumLimbs> &a) noexcept {
        if (std::ranges::all_of(a, [](const Limb &l) { return l == Limb {0}; })) {
            return true;
        }
        return false;
    }

    template<NAULimb Limb>
    constexpr bool _limb_get_nth_bit(const Limb l, const std::uint32_t n) noexcept {
        if (n >= limb_bits_v<Limb>) {
            return false;
        }
        return (l >> n) & Limb {1};
    }

    template<NAULimb Limb>
    constexpr bool _limb_nth_bit_is_set(const Limb l, const std::uint32_t n) noexcept {
        return _limb_get_nth_bit(l, n);
    }

    template<NAULimb Limb>
    constexpr bool _limb_nth_bit_is_unset(const Limb l, const std::uint32_t n) noexcept {
        return !(_limb_get_nth_bit(l, n));
    }

    template<NAULimb Limb>
    constexpr bool _limb_lsb_set(Limb l) noexcept {
        return _limb_get_nth_bit(l, 0);
    }

    template<NAULimb Limb>
    constexpr bool _limb_msb_set(Limb l) noexcept {
        return _limb_get_nth_bit(l, limb_bits_v<Limb> - 1);
    }

    template<NAULimb Limb, std::size_t NumLimbs>
    constexpr bool msb_is_set(const std::array<Limb, NumLimbs> &a) noexcept {
        return _limb_msb_set(a[NumLimbs - 1]);
    }

    template<NAULimb Limb>
    constexpr std::uint32_t _limb_countl_zero(const Limb l) noexcept {
        return static_cast<std::uint32_t>(std::countl_zero(l));
    }

    template<NAULimb Limb>
    constexpr std::uint32_t _limb_countr_zero(const Limb l) noexcept {
        return static_cast<std::uint32_t>(std::countr_zero(l));
    }

    template<NAULimb Limb>
    constexpr std::uint32_t _limb_countl_one(const Limb l) noexcept {
        return static_cast<std::uint32_t>(std::countl_one(l));
    }

    template<NAULimb Limb>
    constexpr std::uint32_t _limb_countr_one(const Limb l) noexcept {
        return static_cast<std::uint32_t>(std::countr_one(l));
    }

    template<NAULimb Limb>
    constexpr std::uint32_t _limb_popcount(const Limb l) noexcept {
        return static_cast<std::uint32_t>(std::popcount(l));
    }

    template<NAULimb Limb>
    constexpr bool _limb_has_single_bit(const Limb l) noexcept {
        return std::has_single_bit(l);
    }

    template<NAULimb Limb>
    constexpr bool _limb_is_pow2(const Limb l) noexcept {
        return _limb_has_single_bit(l);
    }

    template<NAULimb Limb>
    constexpr bool _limb_is_pow_2_or_zero(const Limb l) noexcept {
        return (l == Limb {0}) || _limb_has_single_bit(l);
    }

    template<NAULimb Limb>
    constexpr Limb _limb_bit_ceil(const Limb l) noexcept {
        return std::bit_ceil(l);
    }

    template<NAULimb Limb>
    constexpr Limb _limb_bit_floor(const Limb l) noexcept {
        return std::bit_floor(l);
    }

    template<NAULimb Limb>
    constexpr Limb _limb_bit_width(const Limb l) noexcept {
        return std::bit_width(l);
    }


    template<NAULimb Limb, std::size_t NumLimbs>
    constexpr std::int8_t cmp_abs(const std::array<Limb, NumLimbs> &a, const std::array<Limb, NumLimbs> &b) noexcept {
        for (std::size_t i = NumLimbs; i-- > 0;) {
            if (a[i] < b[i])
                return -1;
            if (a[i] > b[i])
                return 1;
        }
        return 0;
    }

    template<NAULimb Limb, std::size_t NumLimbs>
    constexpr std::int8_t cmp_signed(const std::array<Limb, NumLimbs> &a,
                                      const std::array<Limb, NumLimbs> &b) noexcept {
        const bool a_neg = msb_is_set(a);
        if (const bool b_neg = msb_is_set(b); a_neg != b_neg) {
            return a_neg ? -1 : 1; // if signs differ, negative is less
        }
        // Both same sign, compare absolute values
        const std::int8_t abs_cmp = cmp_abs<Limb, NumLimbs>(a, b);
        return a_neg ? -abs_cmp : abs_cmp; // if negative, reverse result
    }

    namespace arith_utils {



        template <NAULimb Limb>
        constexpr bool add_with_carry(Limb &res, const Limb a, const Limb b, const Limb carry_in = Limb {0}) noexcept {
            const bool carry1 = __builtin_add_overflow(a, b, &res);
            const bool carry2 = __builtin_add_overflow(res, carry_in, &res);
            const bool carry_out      = carry1 || carry2;
            return carry_out;
        }
    }

    template<NAULimb Limb, std::size_t NumLimbs>
    constexpr Limb add_with_carry(std::array<Limb, 2 * NumLimbs> &res,
                                  const std::array<Limb, NumLimbs> &a,
                                  const std::array<Limb, NumLimbs> &b,
                                  const Limb carry_in = Limb {0}) noexcept {
        Limb carry = carry_in;
        for (std::size_t i = 0; i < NumLimbs; ++i) {
            const auto [sum1, carry1] =
            res[i]                    = sum1;
            carry                     = carry1;
        }
        res[NumLimbs] = carry; // final carry out
        return carry;
    }

} // namespace detail

template<bool IsSigned = false, std::size_t NumBits = 256, typename Limb = std::uint32_t>
struct BigInt {};
} // namespace bigint


#endif // ATTENDANCESERVER_BIG_INT_HPP

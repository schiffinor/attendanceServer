//
// Created by schif on 9/30/2025.
//

#ifndef ATTENDANCESERVER_ARITH_OPS_HPP
#define ATTENDANCESERVER_ARITH_OPS_HPP
#include <cstdint>
#include <type_traits>

#include "../simd/cpu.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#define ARITH_x86_64 1
#elif defined(__i386__) || defined(_M_IX86)
#define ARITH_x86_32 1
#else
#define ARITH_SCALAR 1
#endif

#if ARITH_x86_64 || ARITH_x86_32
#include <immintrin.h>
#endif

#ifndef ARITH_HAS_BUILTIN
#  if defined(__has_builtin)
#    define ARITH_HAS_BUILTIN(x) __has_builtin(x)
#  else
#    define ARITH_HAS_BUILTIN(x) 0
#  endif
#endif

namespace simd::arith_utils {

// ----- Allowed limb types (UNSIGNED ONLY) -----
template<class T>
concept UnsignedLimb = std::is_unsigned_v<T> && (sizeof(T) == 1 ||
                                                 sizeof(T) == 2 ||
                                                 sizeof(T) == 4 ||
                                                 sizeof(T) == 8
                                                );

template<class T>
concept Limb8 = UnsignedLimb<T> && (sizeof(T) == 1);
template<class T>
concept Limb16 = UnsignedLimb<T> && (sizeof(T) == 2);
template<class T>
concept Limb32 = UnsignedLimb<T> && (sizeof(T) == 4);
template<class T>
concept Limb64 = UnsignedLimb<T> && (sizeof(T) == 8);

template<class T>
concept LimbUT16 = Limb8<T> || Limb16<T>;
template<class T>
concept LimbUT32 = Limb8<T> || Limb16<T> || Limb32<T>;
template<class T>
concept LimbUT64 = Limb8<T> || Limb16<T> || Limb32<T> || Limb64<T>;


// ----- Function Pointer Signatures -----

template<UnsignedLimb U>
using AddCarryFn = std::uint8_t (*)(U &out, U a, U b, U carry_in) noexcept;

template <UnsignedLimb U>
using SubBorrowFn = std::uint8_t (*)(U &out, U a, U b, U borrow_in) noexcept;


// ----- Per-width op set & kernel table -----

template <UnsignedLimb U>
struct OpSet {
    AddCarryFn<U> add_carry {};
    SubBorrowFn<U> sub_borrow {};
};

struct KernelTable {
    OpSet<std::uint8_t> u8_ops {};
    OpSet<std::uint16_t> u16_ops {};
    OpSet<std::uint32_t> u32_ops {};
    OpSet<std::uint64_t> u64_ops {};
};

// ----- All available implementations -----

// ---------- Portable fallbacks (all architectures) ----------

template<UnsignedLimb U>
constexpr std::uint8_t addcarry_portable(U &res, const U a, const U b, const U carry_in = 0) noexcept {
    const auto tmp = static_cast<U>(a + b);
    const auto c1 = static_cast<std::uint8_t>(tmp < a);
    res = static_cast<U>(tmp + carry_in);
    const auto c2 = static_cast<std::uint8_t>(res < tmp);
    return static_cast<std::uint8_t>(c1 | c2);
}

template<UnsignedLimb U>
constexpr std::uint8_t subborrow_portable(U &res, const U a, const U b, const U borrow_in = 0) noexcept {
    const auto tmp = static_cast<U>(a - b);
    const auto b1 = static_cast<std::uint8_t>(tmp > a);
    res = static_cast<U>(tmp - borrow_in);
    const auto b2 = static_cast<std::uint8_t>(res > tmp);
    return static_cast<std::uint8_t>(b1 | b2);
}

// ---------- Compiler builtins ----------

#if defined(__clang__) || defined(__GNUC__)

// --------------- __builtin_add_overflow (Clang/GCC) ---------------
#if ARITH_HAS_BUILTIN(__builtin_add_overflow)

template<UnsignedLimb U>
constexpr std::uint8_t addcarry_builtin_add_ovf(U &res, const U a, const U b, const U carry_in = 0) noexcept {
    const auto c1 = static_cast<std::uint8_t>(__builtin_add_overflow(a, b, &res));
    const auto c2 = static_cast<std::uint8_t>(__builtin_add_overflow(res, carry_in, &res));
    return c1 || c2;
}

#endif
// --------------- __builtin_sub_overflow (Clang/GCC) ---------------
#if ARITH_HAS_BUILTIN(__builtin_sub_overflow)
template<UnsignedLimb U>
constexpr std::uint8_t subborrow_builtin_sub_ovf(U &res, const U a, const U b, const U borrow_in = 0) noexcept {
    const auto b1 = static_cast<std::uint8_t>(__builtin_sub_overflow(a, b, &res));
    const auto b2 = static_cast<std::uint8_t>(__builtin_sub_overflow(res, borrow_in, &res));
    return b1 || b2;
}
#endif

// --------------- __builtin_addc / __builtin_addcl / __builtin_addcll (Clang/GCC) ---------------
#if ARITH_HAS_BUILTIN(__builtin_addc) && ARITH_HAS_BUILTIN(__builtin_addcl) && ARITH_HAS_BUILTIN(__builtin_addcll)

#if defined(__clang__)

#if ARITH_HAS_BUILTIN(__builtin_addcb) && ARITH_HAS_BUILTIN(__builtin_addcs)

template<Limb8 U>
constexpr std::uint8_t addcarry_builtin_addc(U &res, const U a, const U b, const U carry_in = 0) noexcept {
    U carry_out;
    res = _builtin_addcb(a, b, carry_in, &carry_out);
    return carry_out;
}

template <Limb16 U>
constexpr std::uint8_t addcarry_builtin_addc(U &res, const U a, const U b, const U carry_in = 0) noexcept {
    U carry_out;
    res = __builtin_addcs(a, b, carry_in, &carry_out);
    return static_cast<std::uint8_t>(carry_out);
}
#endif

#if ARITH_HAS_BUILTIN(__builtin_subcb) && ARITH_HAS_BUILTIN(__builtin_subcs)
template<Limb8 U>
constexpr std::uint8_t subborrow_builtin_subc(U &res, const U a, const U b, const U borrow_in = 0) noexcept {
    U borrow_out;
    res = __builtin_subcb(a, b, borrow_in, &borrow_out);
    return borrow_out;
}

template<Limb16 U>
constexpr std::uint8_t subborrow_builtin_subc(U &res, const U a, const U b, const U borrow_in = 0) noexcept {
    U borrow_out;
    res = __builtin_subcs(a, b, borrow_in, &borrow_out);
    return static_cast<std::uint8_t>(borrow_out);
}

#endif

#endif



#endif

#endif





} // namespace simd::arith_utils

#endif

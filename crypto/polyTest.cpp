//
// Created by schif on 6/7/2025.
//

#include <cmath>
#include <iostream>


#include "poly.hpp"
using namespace simd;
template<std::uint32_t Q_>
using Poly16 = Poly<16, Q_>;

int main() {
    std::array<uint32_t, 16> tmp1 =
            {1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, 9ul, 10ul, 11ul, 12ul, 13ul, 14ul, 15ul, 16ul};
    std::array<uint32_t, 16> tmp2 = {1ul,
                                     2ul,
                                     4ul,
                                     8ul,
                                     16ul,
                                     32ul,
                                     64ul,
                                     128ul,
                                     256ul,
                                     512ul,
                                     1024ul,
                                     2048ul,
                                     4096ul,
                                     8192ul,
                                     16384ul,
                                     32768ul};

    const auto poly1 = Poly16<3329ul>(tmp1.begin(), tmp1.end());
    const auto poly2 = Poly16<3329ul>(tmp2.begin(), tmp2.end());
    auto poly3 = Poly16<3329ul>(0);
    auto poly4 = Poly16<3329ul>(0);
    std::cout << "log_2_n: " << 32 - __builtin_clz(3329ul) << "\n";
    // print out BARRETT_MU64
    std::cout << "BARRETT_MU64: " << detail::BARRETT_MU64<3329ul> << "\n";
    // pre modified values to check
    std::cout << "Initial Poly1: ";
    for (const auto &val : poly1.v) {
        std::cout << val << " ";
    }
    std::cout << "\nInitial Poly2: ";
    for (const auto &val : poly2.v) {
        std::cout << val << " ";
    }
    std::cout << "\nInitial Poly3 (result): ";
    for (const auto &val : poly3.v) {
        std::cout << val << " ";
    }
    std::cout << "\nInitial Poly4 (result): ";
    for (const auto &val : poly4.v) {
        std::cout << val << " ";
    }

    Poly16<3329ul>::add(poly1, poly2, poly3);
    // Display poly1
    std::cout << "\nPoly1: ";
    for (const auto &val : poly1.v) {
        std::cout << val << " ";
    }
    std::cout << "\nPoly2: ";
    for (const auto &val : poly2.v) {
        std::cout << val << " ";
    }
    std::cout << "\nPoly3 (result): ";
    for (const auto &val : poly3.v) {
        std::cout << val << " ";
    }

    Poly16<3329ul>::mult_schoolbook(poly1, poly2, poly4);
    // Display poly1
    std::cout << "\nPoly1: ";
    for (const auto &val : poly1.v) {
        std::cout << val << " ";
    }
    std::cout << "\nPoly2: ";
    for (const auto &val : poly2.v) {
        std::cout << val << " ";
    }
    std::cout << "\nPoly4 (result): ";
    for (const auto &val : poly4.v) {
        std::cout << val << " ";
    }

    // test reduce_vec on poly3
    std::array<std::uint32_t, 16> tmp32;
    for (std::size_t i = 0; i < 16; ++i)
        tmp32[i] = static_cast<std::uint32_t>(1) << i; // 1, 2, 4, 8, ..., 32768)
    std::array<std::uint16_t, 16> reduced;
    kernels::reduce_vec<3329ul, 16>(tmp32.data(), reduced.data());
    std::cout << "\ninitial Poly3 coefficients: ";
    for (const auto &val : tmp32) {
        std::cout << val << " ";
    }

    std::cout << "\nReduced Poly3: ";
    for (const auto &val : reduced) {
        std::cout << val << " ";
    }

    //---------------------------------------------------
    //  Tiny sanity harness – paste into main()
    //---------------------------------------------------
    auto refPoly = [](const std::array<uint16_t, 16> &a, const std::array<uint16_t, 16> &b) -> std::array<int32_t, 16> {
        std::array<int32_t, 16> acc {};
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                constexpr int Q = 3329;
                const int k    = (i + j) & 15;
                const int s    = ((i + j) & 16) ? -1 : 1; // sign flip (x^16 = -1)
                const int prod = s * static_cast<int>(a[i]) * static_cast<int>(b[j]);
                acc[k]   = (acc[k] + prod) % Q; // true mod, may be negative
                if (acc[k] < -Q)
                    acc[k] += Q; // keep in (-Q,Q)
            }
        return acc;
    };

    const auto ref = refPoly(poly1.v, poly2.v); // ground-truth

    // run your current multiply (already prints accumulator)
    Poly16<3329>::mult_schoolbook(poly1, poly2, poly4);

    // now dump ref and diff
    std::cout << "\n\nidx  ref  acc  diff\n";
    for (int i = 0; i < 16; ++i) {
        const int32_t accVal = poly4.v[i]; // after reduction
        std::cout << i << ' ' << ref[i] << ' ' << accVal << ' ' << (accVal - ref[i]) << "\n";
    }

    // lets do another test with just 1 1 1 ... 1 for both polys
    const simd::Poly<16, 3329ul> poly5(1);
    const simd::Poly<16, 3329ul> poly6(1);
    simd::Poly<16, 3329ul> poly7(0);
    std::cout << "\n\nTesting with 1 1 1 ... 1 for both polys\n";
    std::cout << "Poly5: ";
    for (const auto &val : poly5.v) {
        std::cout << val << " ";
    }
    std::cout << "\nPoly6: ";
    for (const auto &val : poly6.v) {
        std::cout << val << " ";
    }
    simd::Poly<16, 3329ul>::mult_schoolbook(poly5, poly6, poly7, true);
    std::cout << "\nPoly7 (result): ";
    for (const auto &val : poly7.v) {
        std::cout << val << " ";
    }

    // test barrett_s64 against basic mod 3329
    for (std::int64_t i = -10000; i < 10000; ++i) {
        std::uint16_t reduced = detail::barrett_s64<3329ul>(i);
        std::int64_t mod      = i % 3329;
        if (mod < 0)
            mod += 3329; // ensure non-negative
        if (reduced != static_cast<std::uint16_t>(mod)) {
            std::cout << "Mismatch for " << i << ": "
                      << "reduced = " << reduced << ", expected = " << mod << "\n";
        }
    }
    constexpr unsigned __int128 maxVal = static_cast<unsigned __int128>(1) << 64;
    std::cout << "\nNEG_ADJ manual calc: " << (static_cast<std::uint32_t>(maxVal % 3329ul)) << "\n";
    std::cout << "NEG_ADJ_64: " << detail::NEG_ADJ_64<3329ul> << "\n";

    // Test reduce_vec_s64
    constexpr std::array<std::int64_t, 16> testVec = {10000, -10000, 5000, -5000, 2000, -2000, 1000, -1000,
                                             500, -500, 250, -250, 125, -125, 62, -62};
    // Display the original test vector
    std::cout << "\nOriginal test vector:\n";
    for (const auto &val : testVec) {
        std::cout << val << " ";
    }
    std::array<std::uint16_t, 16> reducedVec;
    kernels::reduce_vec_s64<3329ul, 16>(testVec.data(), reducedVec.data());
    std::cout << "\nReduced vector from reduce_vec_s64:\n";
    for (const auto &val : reducedVec) {
        std::cout << val << " ";
    }



    return 0;
};

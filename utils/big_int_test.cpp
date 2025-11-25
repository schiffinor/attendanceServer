//
// Created by schif on 9/23/2025.
//

#include "big_int.hpp"

#include <bitset>
#include <iostream>
#include <string_view>

bool smdGPT() {
    using namespace bigint;

    // code is not done yet so lets just test the runtime masks.
    // we will do two tests one on k dependent and one on k independent
    // both will depend on some runtime value provided by cin
    bool test1_passed = false;
    bool test2_passed = false;
    std::cout << "Enter a value of k for a kth-bit mask: ";
    std::size_t k;
    std::cin >> k;
    std::cout << "k = " << k << "\n";
    detail::_limb_countl_zero(k);
    const auto test1 = limb_utils::masks_rt::kth_bit<std::uint32_t>(k);
    if (test1 == std::uint32_t {1} << k) {
        std::cout << "kth-bit mask test passed ✔\n";
        test1_passed = true;
    } else {
        std::cout << "kth-bit mask test FAILED\n";
        std::cout << "Expected: " << std::hex << (std::uint32_t {1} << k) << ", got: \nHex: " << std::hex << test1
                  << "\nBinary: " << std::bitset<32>(test1) << "\n";
    }
    std::cout << "kth-bit mask: \nHex: " << std::hex << test1 << "\nBinary: " << std::bitset<32>(test1) << "\n";
    std::uint32_t test2;
    if (k < 16) {
        test2 = limb_utils::masks_rt::all_ones<std::uint32_t>();
        std::cout << "all-ones mask: \nHex: " << std::hex << test2 << "\nBinary: " << std::bitset<32>(test2) << "\n";

    } else {
        test2 = limb_utils::masks_rt::all_zeros<std::uint32_t>();
        std::cout << "all-zeros mask: \nHex: " << std::hex << test2 << "\nBinary: " << std::bitset<32>(test2) << "\n";
    }
    if (test2 == (k < 16 ? std::uint32_t {0xFFFFFFFF} : std::uint32_t {0})) {
        std::cout << "all-ones/all-zeros mask test passed ✔\n";
        detail::_limb_countl_zero(100000ull);
        test2_passed = true;
    } else {
        std::cout << "all-ones/all-zeros mask test FAILED\n";
        std::cout << "Expected: " << std::hex << (k < 16 ? std::uint32_t {0xFFFFFFFF} : std::uint32_t {0})
                  << ", got: \nHex: " << std::hex << test2 << "\nBinary: " << std::bitset<32>(test2) << "\n";
    }

    /*
     * Context for this, I was having ChatGPT review my code because I'm a solo dev and while it's not as good as
     * having a human peer review it, it's better than nothing.
     *
     * However, it kept insisting that the runtime masks were wrong because I was calling consteval functions
     * at runtime. I tried to explain that that's fine as long as I'm not using any values that are not
     * known at compile time. It kept insisting that I was wrong and that I needed to rewrite the code to use
     * inline constexpr functions instead. So I wrote this little test to prove that the code works as intended.
     *
     * By making the choice of which mask to print depend on the runtime value of k, I ensure that
     * the function call to kth_bit is not optimized away at compile time. If the code compiles and runs
     * and prints the expected output, then it proves that the code is correct and that ChatGPT is wrong.
     *
     * Censoring myself because I'm publishing to GitHub.
     */
    std::cout << "IF THIS PRINTS... F*** YOU CHAT GPT lol\n";

    return test1_passed && test2_passed;
}

int main() {
    using namespace bigint;

    bool all_tests_passed = true;

    const bool smdGPT_test_passed = smdGPT();
    all_tests_passed &= smdGPT_test_passed;
    std::cout << "smdGPT test " << (smdGPT_test_passed ? "passed ✔" : "FAILED") << "\n";

    // We can actually determine if a specific test passed
    if (all_tests_passed) {
        std::cout << "\nAll tests passed ✔\n";
        return 0;
    } else {
        std::cout << "\nSome tests FAILED\n";
        return 1;
    }
}

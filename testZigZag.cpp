//
// Created by schif on 9/4/2025.
//


#include <iostream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
/**
 * @brief Compute a signed zig-zag offset for interleaved left-right traversal.
 *
 * This function maps an unsigned index `idx` and a step count `offset` into
 * a signed ptrdiff_t that alternates sign on each successive `idx` value:
 *   - For even idx: returns +⌊(idx + 2*offset + 1)/2⌋
 *   - For odd  idx: returns -⌊(idx + 2*offset + 1)/2⌋
 *
 * It uses a signed type the same width as `size_t` to ensure two’s-complement
 * safety, preventing overflow in signed arithmetic.
 *
 * @param idx    Zero-based element index (unsigned).
 * @param offset Step count to incorporate into zigzag pattern.
 * @return Signed zig-zag offset, alternating sign each `idx`.
 */
static std::ptrdiff_t computeZigzagOffset(const std::size_t idx, const std::size_t offset) noexcept {
    using S     = int64_t;
    const S i   = static_cast<S>(idx);
    const S off = static_cast<S>(offset);

    // Compute the positive step count first, then apply alternating sign:
    // pos = ⌊(i + 2*off + 1) / 2⌋
    const S pos = (i + 2 * off + 1) / 2;
    // If idx is odd (lowest bit = 1), negate pos; else return +pos
    return (i & 1u) ? -pos : pos; // two’s-comp safe
}

/**
 * @brief Compute a pair of zig-zag offsets for left/right interleaving.
 *
 * For a given pair index `nth`, returns:
 *   {.first  = zigzagOffset(2*nth,   l_cnt),
 *    .second = zigzagOffset(2*nth+1, r_cnt)}
 *
 * This yields one even offset (first) and one odd offset (second),
 * enabling a balanced left/right selection in bulk algorithms.
 *
 * @param nth   Index of the desired offset pair.
 * @param l_cnt Number of left picks so far.
 * @param r_cnt Number of right picks so far.
 * @return {leftOffset, rightOffset} as ptrdiff_t values.
 */
static std::pair<std::ptrdiff_t, std::ptrdiff_t> computeZigzagOffsetPair(const std::size_t nth   = 0,
                                                                         const std::size_t l_cnt = 0,
                                                                         const std::size_t r_cnt = 0) noexcept {
    return {computeZigzagOffset(2 * nth, l_cnt), computeZigzagOffset(2 * nth + 1, r_cnt)};
}

// ---------- square-array printer ----------
constexpr bool is_square(const std::size_t n) {
    const auto r = static_cast<std::size_t>(std::sqrt(static_cast<long double>(n)));
    return r * r == n;
}

// --- labeled square-array printer ---
template <std::size_t N>
void print_pair_array(const std::pair<std::ptrdiff_t, std::ptrdiff_t> (&a)[N],
                      const char* title = nullptr) {
    static_assert(N > 0, "array must not be empty");
    const auto side =
        static_cast<std::size_t>(std::sqrt(static_cast<long double>(N)));
    if (side * side != N) {
        throw std::runtime_error("Array length must be a perfect square");
    }

    // column width per number so columns line up
    int w = 0;
    for (const auto &p : a) {
        w = std::max<int>(w, std::max(
            (int)std::to_string(p.first).size(),
            (int)std::to_string(p.second).size()));
    }
    const int cell_num_w = std::max(2, w);
    const int cell_w = 2 * cell_num_w + 4; // width of "(a,b)"

    auto digits = [](std::size_t x) -> int {
        int d = 1; while (x >= 10) { x /= 10; ++d; } return d;
    };
    const int idx_w = digits(side - 1);

    if (title) std::cout << title << '\n';

    // header row (column indices)
    std::cout << std::setw(idx_w) << ' ' << "  ";
    for (std::size_t j = 0; j < side; ++j) {
        std::cout << j << std::setw(cell_w + 1); // +1 for the trailing space after each cell
    }
    std::cout << '\n';

    // each row with left label
    for (std::size_t i = 0; i < side; ++i) {
        std::cout << std::setw(idx_w) << i << "  ";
        for (std::size_t j = 0; j < side; ++j) {
            const auto &p = a[i * side + j];
            std::cout << '(' << std::setw(cell_num_w) << p.first
                      << ',' << std::setw(cell_num_w) << p.second << ") ";
        }
        std::cout << '\n';
    }
}


int main() {
    std::cout << "Hello World\n";
    constexpr std::size_t m  = 9;
    constexpr std::size_t mm = m * m;

    for (std::size_t n = 0; n < 11; ++n) {
        std::pair<std::ptrdiff_t, std::ptrdiff_t> array_for_n[mm];

        for (std::size_t i = 0; i < m; ++i) {        // <-- FIX: < m
            for (std::size_t j = 0; j < m; ++j) {    // <-- FIX: < m
                const std::size_t idx = m * i + j;
                array_for_n[idx] = computeZigzagOffsetPair(n, i, j);
            }
        }

        print_pair_array(array_for_n, ("n = " + std::to_string(n) /* optional title */).c_str() /* optional title */);
    }
    return 0;
}

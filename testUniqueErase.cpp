#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <ranges>

void print(const std::vector<int>& v, std::string_view label)
{
    std::cout << label << " { ";
    for (int x : v) std::cout << x << ' ';
    std::cout << "}\n";
}

int main()
{
    // ----- 1. make a vector that contains duplicates -------------- //
    std::vector<int> data { 7, 2, 9, 2, 7, 7, 4, 9, 1, 4 };

    print(data, "raw      ");

    // ----- 2. sort (needed because unique removes *consecutive* dups) //
    std::ranges::sort(data);
    print(data, "sorted   ");

    // ----- 3. unique + erase idiom ---------------------------------- //
    auto newEnd = std::unique(data.begin(), data.end());   // step A
    data.erase(newEnd, data.end());                        // step B

    print(data, "deduped  ");

    // ----- 4. sanity check ------------------------------------------ //
    const std::vector<int> expected { 1, 2, 4, 7, 9 };
    if (data != expected)
        std::cerr << "Error: dedup failed!\n";
}

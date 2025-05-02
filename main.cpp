#include <iostream>
#include <cblas.h>
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <boost/regex.hpp>

#include <iostream>
#include <string>
#include "DOUBLYLINKEDCIRCULARHASHMAP.hpp"
#include <cassert>

int doublyLinkedCHMTest() {
    // 1) Build and fill the map
    DoublyLinkedCircularHashMap<int, std::string> map;
    for (int i = 0; i < 50; i++) {
        map.insert(i, std::to_string(i));
    }

    // 2) Print via operator[]
    std::cout << "=== Initial map ===\n";
    for (int i = 0; i < 50; i++) {
        std::cout << "Key: " << i
                  << ", Value: " << map[i] << "\n";
    }

    // 3) First orderedGet (stepping by 7, from head)
    std::cout << "\n=== orderedGet stepping by 7 ===\n";
    for (size_t i = 0; i < map.size(); i++) {
        auto p = map.orderedGet(7 * i, nullptr, true);
        std::cout << "Index " << i
                  << " -> Value: " << *p << "\n";
    }

    // 4) Second orderedGet (stepping by 7, starting from a custom node)
    std::cout << "\n=== orderedGet stepping by 7, from node at 2*i ===\n";
    for (size_t i = 0; i < map.size(); i++) {
        // pick a “from” position at index 2*i
        auto from = map.orderedGetNode(2 * i, nullptr, false);
        auto p    = map.orderedGet(7 * i, from, true);
        std::cout << "Index " << i
                  << " -> Value: " << *p << "\n";
    }

    // 5) Remove every 5th key
    std::cout << "\n=== Removing every 5th key ===\n";
    for (int i = 0; i < 50; i += 5) {
        map.remove(i);
    }
    std::cout << "Remaining keys:\n";
    for (int i = 0; i < 50; i++) {
        if (map.contains(i))
            std::cout << i << " ";
    }
    std::cout << "\n";

    // 6) Test find()
    std::cout << "\n=== find() results ===\n";
    for (int i = 0; i < 50; i++) {
        if (auto v = map.find_ptr(i))
            std::cout << "Key " << i << " -> " << *v << "\n";
        else
            std::cout << "Key " << i << " not found\n";
    }

    // 7) Mutate via iterator
    std::cout << "\n=== Mutating via iterator ===\n";
    for (auto &v: map | std::views::values) {
        v += "!";
    }

    // 8) Read-only iteration
    std::cout << "\n=== After mutation (const iteration) ===\n";
    for (auto const& [k, v] : map) {
        std::cout << k << " -> " << v << "\n";
    }

    // 9) Final size
    std::cout << "\nFinal size: " << map.size() << "\n";

    // 10) Testing shift and move operations on a small map
    std::cout << "\n=== Testing shift and move operations ===\n";
    DoublyLinkedCircularHashMap<int, int> smap;
    for (int i = 1; i <= 5; ++i) smap.insert(i, i);
    std::cout << "Original small map: ";
    for (const auto &k: smap | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    smap.shift_idx(2, 1);
    std::cout << "After shift_idx(2,1): ";
    for (const auto &k: smap | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    smap.shift_idx(3, -1);
    std::cout << "After shift_idx(3,-1): ";
    for (const auto &k: smap | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    smap.shift_n_key(4, -2);
    std::cout << "After shift_n_key(4,-2): ";
    for (const auto &k: smap | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    smap.pos_swap_k(1, 5);
    std::cout << "After pos_swap_k(1,5): ";
    for (const auto &k: smap | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    smap.pos_swap(0, 2);
    std::cout << "After pos_swap(0,2): ";
    for (const auto &k: smap | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    smap.move_n_key_to_n_key(3, 2);
    std::cout << "After move key 3 before key 2: ";
    for (const auto &k: smap | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    smap.move_idx_to_idx(4, 1);
    std::cout << "After move index 4 before index 1: ";
    for (const auto &k: smap | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    // Build two maps:
    DoublyLinkedCircularHashMap<int, int> a, b;
    for(int i = 1; i <= 5; ++i) a.insert(i, i*10);  // a: 1,2,3,4,5
    for(int i = 6; i <= 8; ++i) b.insert(i, i*10);  // b: 6,7,8
    std::cout << "Initial state of map a: ";
    for (const auto &k: a | std::views::keys) std::cout << k << " ";
    std::cout << "\n";
    std::cout << "Initial state of map b: ";
    for (const auto &k: b | std::views::keys) std::cout << k << " ";
    std::cout << "\n\n";

    // Splice out [2, 5) from 'a' (i.e. keys 2,3,4) into the front of 'b'
    auto first = a.find(2);
    auto last  = a.find(5);      // half-open → does not include key 5
    auto ret   = b.splice(b.begin(), a, first, last);
    std::cout << "First in splice: " << (*first).first << "\n";
    std::cout << "Last in splice: " << (*last).first << "\n\n";

    // After splice:
    //   a should be: [1,5]
    //   b should be: [2,3,4,6,7,8]
    //   ret should == iterator to the element '2' in b

    std::vector expectA = {1,5};
    std::vector expectB = {2,3,4,6,7,8};

    // verify a
    {
        size_t idx = 0;
        for (const auto &fst: a | std::views::keys) {
            assert(idx < expectA.size());
            assert(fst == expectA[idx]);
            ++idx;
        }
        assert(idx == expectA.size());
    }

    // verify b
    {
        size_t idx = 0;
        for (auto it = b.begin(); it != b.end(); ++it, ++idx) {
            assert(idx < expectB.size());
            assert((*it).first == expectB[idx]);
        }
        assert(idx == expectB.size());
    }

    // verify return iterator points at the first spliced element (key=2)
    assert(ret != b.end() && (*ret).first == 2);

    std::cout << "[OK] splice test passed\n";
    // print the final state of both maps
    std::cout << "Final state of map a: ";
    for (const auto &k: a | std::views::keys) std::cout << k << " ";
    std::cout << "\n";
    std::cout << "Final state of map b: ";
    for (const auto &k: b | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    // test split
    std::cout << "\n=== Testing split ===\n";
    DoublyLinkedCircularHashMap<int, int> map3;
    for (int i = 0; i < 10; i++) {
        map3.insert(i, i);
    }
    std::cout << "Original map:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "Key: " << i
                  << ", Value: " << map3[i] << "\n";
    }
    auto splitMap = map3.split(5);
    std::cout << "Split map (should be keys 5..9):\n";
    for (auto const& [k, v] : splitMap) {
        std::cout << "Key: " << k << ", Value: " << v << "\n";
    }
    std::cout << "Original map after split:\n";
    for (auto const& [k, v] : map3) {
        std::cout << "Key: " << k << ", Value: " << v << "\n";
    }

    // === Testing setHashFunction with bucket‐introspection ===
    std::cout << "\n=== Testing setHashFunction ===\n";

    // build a small map so collisions are easy
    DoublyLinkedCircularHashMap<int,int> hmap(8, /*maxLoadFactor=*/1.0);
    for(int i = 0; i < 16; ++i) {
        hmap.insert(i, i*10);
    }
    assert(hmap.size() == 16);

    // snapshot insertion order
    std::vector<std::pair<int,int>> before;
    before.reserve(16);
    for(auto const& kv : hmap)
        before.emplace_back(kv);

    // show bucket sizes before
    std::cout << "-- before rehash --\n";
    hmap.printBucketDistribution();

    // also inspect a few raw hashes
    std::cout << "raw hash info before:\n";
    hmap.debugKey(0);
    hmap.debugKey(7);
    hmap.debugKey(15);

    // install a “bad” hash: everything → bucket 0
    hmap.setHashFunction([](const int &key){
        return 0u;
    });

    // verify size unchanged
    assert(hmap.size() == 16);

    // bucket sizes after
    std::cout << "\n-- after rehash --\n";
    hmap.printBucketDistribution();

    // and raw‐hash info again
    std::cout << "raw hash info after:\n";
    hmap.debugKey(0);
    hmap.debugKey(7);
    hmap.debugKey(15);

    // verify lookups & insertion order
    for(int i = 0; i < 16; ++i) {
        auto p = hmap.find_ptr(i);
        assert(p && *p == i*10);
    }
    {
        size_t idx = 0;
        for(const auto&[fst, snd] : hmap) {
            assert(fst  == before[idx].first);
            assert(snd == before[idx].second);
            ++idx;
        }
        assert(idx == before.size());
    }

    std::cout << "[OK] setHashFunction + distribution test passed\n";


    return 0;
}

void testDoublyLinkedCircularHashMap() {
    using Map = DoublyLinkedCircularHashMap<int, std::string>;
    std::cout << "=== Starting DoublyLinkedCircularHashMap Tests ===\n";

    // Observers & Accessors
    Map m(4, 0.75);
    std::cout << "[1] empty(): " << (m.empty() ? "true" : "false") << "\n";
    std::cout << "[2] size(): " << m.size() << "\n";
    std::cout << "[3] bucketCount(): " << m.bucketCount() << "\n";
    std::cout << "[4] loadFactor(): " << m.loadFactor() << "\n";
    std::cout << "[5] maxLoadFactor(): " << m.maxLoadFactor() << "\n";

    // Insert & Remove
    std::cout << "\n=== Testing insert / insert_at / remove ===\n";
    m.insert(10, "ten");
    m.insert(20, "twenty");
    m.insert_at(5, "five", 0);
    std::cout << "After inserts, size = " << m.size() << ", empty() = " << (m.empty() ? "true" : "false") << "\n";

    std::cout << "Removing key 20: " << (m.remove(20) ? "success" : "fail") << "\n";
    std::cout << "Removing key 99 (non-existent): " << (m.remove(99) ? "success" : "fail") << "\n";

    // find_ptr, contains, operator[], at
    std::cout << "\n=== Testing find_ptr / contains / operator[] / at ===\n";
    auto ptr5 = m.find_ptr(5);
    std::cout << "find_ptr(5): " << (ptr5 ? *ptr5 : "<null>") << "\n";
    std::cout << "contains(10): " << (m.contains(10) ? "true" : "false") << "\n";
    std::cout << "operator[](30) = (default) " << m[30] << "\n";
    m[30] = "thirty";
    std::cout << "at(30) = " << m.at(30) << "\n";
    try {
        m.at(999);
    } catch (const std::out_of_range &e) {
        std::cout << "at(999) threw: " << e.what() << "\n";
    }

    // Ordered access
    std::cout << "\n=== Testing orderedGet / orderedGetNode ===\n";
    std::cout << "Element at index 0: " << *m.orderedGet(0) << "\n";
    std::cout << "Element at index -1 (last): " << *m.orderedGet(-1) << "\n";
    m.orderedGet(1, nullptr, true); // debug flag

    // Iterator traversal
    std::cout << "\n=== Iterator traversal ===\n";
    std::cout << "Keys in insertion order: ";
    for (const auto &key: m | std::views::keys) std::cout << key << " ";
    std::cout << "\n";

    // Iterate and mutate
    std::cout << "Mutating values via iterator:\n";
    for (auto &v: m | std::views::values) {
        v += "!";
    }
    std::cout << "After mutation, values are:\n";
    for (const auto &v: m | std::views::values) {
        std::cout << v << " ";
    }
    std::cout << "\n";


    // Queue / Stack operations
    std::cout << "\n=== Queue & Stack functions ===\n";
    m.push_back(40, "forty");
    m.push_front(50, "fifty");
    std::cout << "front() = " << *m.front() << ", back() = " << *m.back() << "\n";
    m.pop_front();
    m.pop_back();
    std::cout << "After pops, size = " << m.size() << "\n";

    // Swap & Move
    std::cout << "\n=== Swap & Move operations ===\n";
    m.insert(60, "sixty");
    m.insert(70, "seventy");
    std::cout << "Before swap positions of 5 and 10: ";
    m.pos_swap_k(5, 10);
    for (const auto &key: m | std::views::keys) std::cout << key << " "; std::cout << "\n";

    // Shift, rotate, reverse
    std::cout << "\n=== Shift / Rotate / Reverse ===\n";
    m.shift_n_key(10, 1);
    std::cout << "After shift_n_key(10,1): "; for (const auto &key: m | std::views::keys) std::cout << key << " "; std::cout << "\n";
    m.rotate(2);
    std::cout << "After rotate(2): "; for (const auto &key: m | std::views::keys) std::cout << key << " "; std::cout << "\n";
    m.reverse();
    std::cout << "After reverse(): "; for (const auto &key: m | std::views::keys) std::cout << key << " "; std::cout << "\n";

    // Splice & Split
    std::cout << "\n=== Splice & Split ===\n";

    // Build two maps:
    DoublyLinkedCircularHashMap<int, int> a, b;
    for(int i = 1; i <= 5; ++i) a.insert(i, i*10);  // a: 1,2,3,4,5
    for(int i = 6; i <= 8; ++i) b.insert(i, i*10);  // b: 6,7,8
    std::cout << "Initial state of map a: ";
    for (const auto &k: a | std::views::keys) std::cout << k << " ";
    std::cout << "\n";
    std::cout << "Initial state of map b: ";
    for (const auto &k: b | std::views::keys) std::cout << k << " ";
    std::cout << "\n\n";

    //Splice out [2, 5) from 'a' (i.e. keys 2,3,4) into the front of 'b'
    auto first = a.find(2);
    auto last  = a.find(5);      // half-open → does not include key 5
    auto ret   = b.splice(b.begin(), a, first, last);
    std::cout << "First in splice: " << (*first).first << "\n";
    std::cout << "Last in splice: " << (*last).first << "\n\n";

    // After splice:
    //   a should be: [1,5]
    //   b should be: [2,3,4,6,7,8]
    //   ret should == iterator to the element '2' in b

    std::vector expectA = {1,5};
    std::vector expectB = {2,3,4,6,7,8};

    // verify a
    {
        size_t idx = 0;
        for ([[maybe_unused]] const auto &fst: a | std::views::keys) {
            assert(idx < expectA.size());
            assert(fst == expectA[idx]);
            ++idx;
        }
        assert(idx == expectA.size());
    }

    // verify b
    {
        size_t idx = 0;
        for (auto it = b.begin(); it != b.end(); ++it, ++idx) {
            assert(idx < expectB.size());
            assert((*it).first == expectB[idx]);
        }
        assert(idx == expectB.size());
    }

    // verify return iterator points at the first spliced element (key=2)
    assert(ret != b.end() && (*ret).first == 2);

    std::cout << "[OK] splice test passed\n";
    // print the final state of both maps
    std::cout << "Final state of map a: ";
    for (const auto &k: a | std::views::keys) std::cout << k << " ";
    std::cout << "\n";
    std::cout << "Final state of map b: ";
    for (const auto &k: b | std::views::keys) std::cout << k << " ";
    std::cout << "\n";

    // test split
    std::cout << "\n=== Testing split ===\n";
    DoublyLinkedCircularHashMap<int, int> map3;
    for (int i = 0; i < 10; i++) {
        map3.insert(i, i);
    }
    std::cout << "Original map:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "Key: " << i
                  << ", Value: " << map3[i] << "\n";
    }
    auto splitMap = map3.split(5);
    std::cout << "Split map (should be keys 5..9):\n";
    for (auto const& [k, v] : splitMap) {
        std::cout << "Key: " << k << ", Value: " << v << "\n";
    }
    std::cout << "Original map after split:\n";
    for (auto const& [k, v] : map3) {
        std::cout << "Key: " << k << ", Value: " << v << "\n";
    }

    // Bucket distribution & debugKey
    std::cout << "\n=== Bucket Distribution & debugKey ===\n";
    m.printBucketDistribution();
    m.debugKey(10);

    // Validate & Minimize
    std::cout << "\n=== Validate & minimize_size ===\n";
    try {
        m.validate();
        std::cout << "validate() passed\n";
    } catch (const std::exception &e) {
        std::cout << "validate() failed: " << e.what() << "\n";
    }
    m.minimize_size();
    std::cout << "After minimize_size, bucketCount() = " << m.bucketCount() << "\n";

    // COmpare both print bucket distribution fuunctions
    std::cout << "\n=== Compare printBucketDistribution and printBucketDistribution2 ===\n";
    m.printBucketDistribution();
    m.printBucketDistribution2();
    std::cout << "Both functions should show the same distribution.\n";

    // === Testing erase_if ===
    {
        using Map = DoublyLinkedCircularHashMap<int, std::string>;
        Map mah;
        // populate 0..9
        for (int i = 0; i < 10; ++i) {
            mah.insert(i, std::to_string(i));
        }

        std::cout << "\nBefore erase_if, size = " << mah.size() << "\n";
        std::cout << "Contents: ";
        for (const auto &key: mah | std::views::keys) std::cout << key << ' ';
        std::cout << "\n";

        // remove all even keys
        size_t removed = mah.erase_if([](const int key, std::string & /*val*/) {
            return key % 2 == 0;
        });

        std::cout << "erase_if removed " << removed << " elements\n";
        std::cout << "After erase_if, size = " << mah.size() << "\n";
        std::cout << "Remaining keys: ";
        for (const auto &key: mah | std::views::keys) std::cout << key << ' ';
        std::cout << "\n";

        // simple check
        if (removed == 5 && mah.size() == 5) {
            std::cout << "[OK] erase_if test passed\n";
        } else {
            std::cout << "[FAIL] erase_if test failed\n";
        }
    }


    std::cout << "=== All tests completed ===\n";
}

int regex_test() {
    std::string text("abc abd");
    const boost::regex regex("ab.");

    boost::sregex_token_iterator iter(text.begin(), text.end(), regex, 0);

    for(const boost::sregex_token_iterator end; iter != end; ++iter ) {
        std::cout<<*iter<<'\n';
    }

    return 0;
}

int cblas_test() {
    // dimensions
    constexpr int N = 3;

    // row‑major matrices A and B
    constexpr double A[ N*N ] = {
        1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        7.0,  8.0,  9.0
   };
    constexpr double B[ N*N ] = {
        9.0,  8.0,  7.0,
        6.0,  5.0,  4.0,
        3.0,  2.0,  1.0
   };
    double C[ N*N ];

    // C := 1.0·A·B + 0.0·C
    cblas_dgemm(
        CblasRowMajor,    // our arrays are row-major
        CblasNoTrans,     // A not transposed
        CblasNoTrans,     // B not transposed
        N, N, N,          // dimensions M=N=K=3
        1.0,              // alpha
        A, N,             // A, leading dim = N
        B, N,             // B, leading dim = N
        0.0,              // beta
        C, N              // C, leading dim = N
    );

    // print C
    std::cout << "C = A * B" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[ i*N + j ] << " ";
            if (j == N - 1) {
                std::cout << std::endl;
            }
        }
    }

    return 0;
}

int main() {
    //call the tests
    doublyLinkedCHMTest();
    regex_test();
    cblas_test();
    testDoublyLinkedCircularHashMap();
    // have user enter a key to exit
    std::cout << "Press Enter to exit...\n";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "Exiting...\n";
    return 0;
}
// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.
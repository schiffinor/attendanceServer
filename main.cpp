#include <iostream>
#include <cblas.h>
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <boost/regex.hpp>
#include <algorithm>
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
        auto p = map.orderedGet(7 * i, from, true);
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
    for (auto const &[k, v]: map) {
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
    for (int i = 1; i <= 5; ++i) a.insert(i, i * 10); // a: 1,2,3,4,5
    for (int i = 6; i <= 8; ++i) b.insert(i, i * 10); // b: 6,7,8
    std::cout << "Initial state of map a: ";
    for (const auto &k: a | std::views::keys) std::cout << k << " ";
    std::cout << "\n";
    std::cout << "Initial state of map b: ";
    for (const auto &k: b | std::views::keys) std::cout << k << " ";
    std::cout << "\n\n";

    // Splice out [2, 5) from 'a' (i.e. keys 2,3,4) into the front of 'b'
    auto first = a.find(2);
    auto last = a.find(5); // half-open → does not include key 5
    auto ret = b.splice(b.begin(), a, first, last);
    std::cout << "First in splice: " << (*first).first << "\n";
    std::cout << "Last in splice: " << (*last).first << "\n\n";

    // After splice:
    //   a should be: [1,5]
    //   b should be: [2,3,4,6,7,8]
    //   ret should == iterator to the element '2' in b

    std::vector expectA = {1, 5};
    std::vector expectB = {2, 3, 4, 6, 7, 8};

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
    for (auto const &[k, v]: splitMap) {
        std::cout << "Key: " << k << ", Value: " << v << "\n";
    }
    std::cout << "Original map after split:\n";
    for (auto const &[k, v]: map3) {
        std::cout << "Key: " << k << ", Value: " << v << "\n";
    }

    // === Testing setHashFunction with bucket‐introspection ===
    std::cout << "\n=== Testing setHashFunction ===\n";

    // build a small map so collisions are easy
    DoublyLinkedCircularHashMap<int, int> hmap(8, /*maxLoadFactor=*/1.0);
    for (int i = 0; i < 16; ++i) {
        hmap.insert(i, i * 10);
    }
    assert(hmap.size() == 16);

    // snapshot insertion order
    std::vector<std::pair<int, int> > before;
    before.reserve(16);
    for (auto const &kv: hmap)
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
    hmap.setHashFunction([](const int &key) {
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
    for (int i = 0; i < 16; ++i) {
        auto p = hmap.find_ptr(i);
        assert(p && *p == i*10);
    } {
        size_t idx = 0;
        for (const auto &[fst, snd]: hmap) {
            assert(fst == before[idx].first);
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
    for (const auto &key: m | std::views::keys) std::cout << key << " ";
    std::cout << "\n";

    // Shift, rotate, reverse
    std::cout << "\n=== Shift / Rotate / Reverse ===\n";
    m.shift_n_key(10, 1);
    std::cout << "After shift_n_key(10,1): ";
    for (const auto &key: m | std::views::keys) std::cout << key << " ";
    std::cout << "\n";
    m.rotate(2);
    std::cout << "After rotate(2): ";
    for (const auto &key: m | std::views::keys) std::cout << key << " ";
    std::cout << "\n";
    m.reverse();
    std::cout << "After reverse(): ";
    for (const auto &key: m | std::views::keys) std::cout << key << " ";
    std::cout << "\n";

    // Splice & Split
    std::cout << "\n=== Splice & Split ===\n";

    // Build two maps:
    DoublyLinkedCircularHashMap<int, int> a, b;
    for (int i = 1; i <= 5; ++i) a.insert(i, i * 10); // a: 1,2,3,4,5
    for (int i = 6; i <= 8; ++i) b.insert(i, i * 10); // b: 6,7,8
    std::cout << "Initial state of map a: ";
    for (const auto &k: a | std::views::keys) std::cout << k << " ";
    std::cout << "\n";
    std::cout << "Initial state of map b: ";
    for (const auto &k: b | std::views::keys) std::cout << k << " ";
    std::cout << "\n\n";

    //Splice out [2, 5) from 'a' (i.e. keys 2,3,4) into the front of 'b'
    auto first = a.find(2);
    auto last = a.find(5); // half-open → does not include key 5
    auto ret = b.splice(b.begin(), a, first, last);
    std::cout << "First in splice: " << (*first).first << "\n";
    std::cout << "Last in splice: " << (*last).first << "\n\n";

    // After splice:
    //   a should be: [1,5]
    //   b should be: [2,3,4,6,7,8]
    //   ret should == iterator to the element '2' in b

    std::vector expectA = {1, 5};
    std::vector expectB = {2, 3, 4, 6, 7, 8};

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
    for (auto const &[k, v]: splitMap) {
        std::cout << "Key: " << k << ", Value: " << v << "\n";
    }
    std::cout << "Original map after split:\n";
    for (auto const &[k, v]: map3) {
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

    for (const boost::sregex_token_iterator end; iter != end; ++iter) {
        std::cout << *iter << '\n';
    }

    return 0;
}

int cblas_test() {
    // dimensions
    constexpr int N = 3;

    // row‑major matrices A and B
    constexpr double A[N * N] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    constexpr double B[N * N] = {
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0
    };
    double C[N * N];

    // C := 1.0·A·B + 0.0·C
    cblas_dgemm(
        CblasRowMajor, // our arrays are row-major
        CblasNoTrans, // A not transposed
        CblasNoTrans, // B not transposed
        N, N, N, // dimensions M=N=K=3
        1.0, // alpha
        A, N, // A, leading dim = N
        B, N, // B, leading dim = N
        0.0, // beta
        C, N // C, leading dim = N
    );

    // print C
    std::cout << "C = A * B" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
            if (j == N - 1) {
                std::cout << std::endl;
            }
        }
    }

    return 0;
}

void print(const std::vector<int> &v, const std::string_view label) {
    std::cout << label << " { ";
    for (const int x: v) std::cout << x << ' ';
    std::cout << "}\n";
}

void testUniqueErase() {
    // ----- 1. make a vector that contains duplicates -------------- //
    std::vector data{7, 2, 9, 2, 7, 7, 4, 9, 1, 4};

    print(data, "raw      ");

    // ----- 2. sort (needed because unique removes *consecutive* dups) //
    std::ranges::sort(data);
    print(data, "sorted   ");

    // ----- 3. unique + erase idiom ---------------------------------- //
    const auto newEnd = std::ranges::unique(data).begin(); // step A
    data.erase(newEnd, data.end()); // step B

    print(data, "deduped  ");

    // ----- 4. sanity check ------------------------------------------ //
    if (const std::vector expected{1, 2, 4, 7, 9}; data != expected)
        std::cerr << "Error: dedup failed!\n";
}

// ---------------------------------------------------------------------------
// Quick functional test of find_n_nodes()
// ---------------------------------------------------------------------------
void DLCHM_findNNodes_smokeTest() {
    using Map = DoublyLinkedCircularHashMap<int, std::string>;
    Map dll;

    // 1. build a list with 10 nodes (keys 0..9)
    for (int k = 0; k < 10; ++k)
        dll.insert(k, "v" + std::to_string(k));

    // 2. request a batch of indices in unsorted order, incl. duplicates
    const std::vector wants{7, 2, 2, 9, -1, 0}; // -1 should wrap to 9
    const std::list wants2 = {7, 2, 2, 9, -1, 0};
    const std::vector wants3{0, 2, 7, 9}; // unique keys

    // 3. call the template; C++17 CTAD deduces the container type
    auto ptrs = dll.find_n_nodes(wants3, true, true);
    auto ptrs2 = dll.find_n_nodes(wants, false, true);
    auto ptrs3 = dll.find_n_nodes(wants2, false, true);

    // 3.1 Print the results
    // print output container type:
    std::cout << "find_n_nodes output container type: "
            << typeid(ptrs).name() << '\n';
    std::cout << "find_n_nodes output container type: "
            << typeid(ptrs2).name() << '\n';
    std::cout << "find_n_nodes output container type: "
            << typeid(ptrs3).name() << '\n';

    // print container size:
    std::cout << "find_n_nodes size: " << ptrs.size() << '\n';
    std::cout << "find_n_nodes size (unique): " << ptrs2.size() << '\n';
    std::cout << "find_n_nodes size (unique, v_out = false): " << ptrs3.size() << '\n';

    // print container contents:
    std::cout << "find_n_nodes results:\n";
    for (const auto *n: ptrs)
        std::cout << "  " << n->key_ << ": " << n->value_ << "\n";

    std::cout << "find_n_nodes results (unique):\n";
    for (const auto *n: ptrs2)
        std::cout << "  " << n->key_ << ": " << n->value_ << "\n";

    std::cout << "find_n_nodes results (unique, v_out = false)\n";
    for (const auto *n: ptrs3)
        std::cout << "  " << n->key_ << ": " << n->value_ << "\n";


    // 4. verify we got unique nodes {0,2,7,9}
    std::vector<int> gotKeys;
    for (const auto *n: ptrs) gotKeys.push_back(n->key_);

    std::vector<int> gotKeys2;
    for (const auto *n: ptrs2) gotKeys2.push_back(n->key_);

    std::list<int> gotKeys3;
    for (const auto *n: ptrs3) gotKeys3.push_back(n->key_);


    if (std::vector expect{0, 2, 7, 9}; gotKeys != expect) {
        std::cerr << "find_n_nodes smoke‑test FAILED\n";
        std::cerr << " expected {0,2,7,9}, got { ";
        for (int k: gotKeys) std::cerr << k << ' ';
        std::cerr << "}\n";
    }
    std::cout << "find_n_nodes smoke‑test passed ✔\n";

    if (std::vector expect2{0, 2, 2, 7, 9, 9}; gotKeys2 != expect2) {
        std::cerr << "find_n_nodes 2 smoke‑test FAILED\n";
        std::cerr << " expected {0,2,2,7,9,9}, got { ";
        for (int k: gotKeys2) std::cerr << k << ' ';
        std::cerr << "}\n";
    }
    std::cout << "find_n_nodes 2 smoke‑test passed ✔\n";

    if (std::list expect3{0, 2, 2, 7, 9, 9}; gotKeys3 != expect3) {
        std::cerr << "find_n_nodes 3 smoke‑test FAILED\n";
        std::cerr << " expected {0,2,2,7,9,9}, got { ";
        for (int k: gotKeys3) std::cerr << k << ' ';
        std::cerr << "}\n";
    }
    std::cout << "find_n_nodes 3 smoke‑test passed ✔\n";
}

void test_zigzag() {
    using Map = DoublyLinkedCircularHashMap<int, std::string>;

    for (int i = 0; i < 10; ++i) {
        std::pair<int, int> p = Map::computeZigzagOffsetPair(i, 0, 0);
        auto [left, right] = p;
        std::cout << "computeZigzagOffsetPair(" << i << ") = "
                << "left: " << left << ", right: " << right << "\n";
    }
}

//------------------------------------------------------------------------------
// A minimal allocator that just counts allocate()/deallocate() calls
struct CountingAllocatorBase {
    static size_t allocCount;
    static size_t deallocCount;
};

size_t CountingAllocatorBase::allocCount = 0;
size_t CountingAllocatorBase::deallocCount = 0;

// now every specialization of CountingAllocator<T> will inherit the same counters:
template<typename T>
struct CountingAllocator : CountingAllocatorBase {
    using value_type = T;

    CountingAllocator() = default;

    template<typename U>
    explicit CountingAllocator(const CountingAllocator<U> &) noexcept {
    }

    static T *allocate(const size_t n) {
        allocCount += n; // this is CountingAllocatorBase::allocCount
        return static_cast<T *>(operator new(n * sizeof(T)));
    }

    static void deallocate(T *p, const size_t n) noexcept {
        deallocCount += n;
        ::operator delete(p);
    }

    template<typename U, typename... Args>
    void construct(U *p, Args &&... args) {
        ::new(static_cast<void *>(p)) U(std::forward<Args>(args)...);
    }

    template<typename U>
    static void destroy(U *p) {
        p->~U();
    }
};

//------------------------------------------------------------------------------
// Test function for allocator support
void testAllocatorSupport() {
    using MapT = DoublyLinkedCircularHashMap<
        int,
        std::string,
        std::hash<int>,
        std::equal_to<int>,
        CountingAllocator<std::pair<const int, std::string> >
    >;
    using Alloc = CountingAllocator<std::pair<const int, std::string> >;

    std::cout << "\n=== Testing Custom Allocator Support ===\n";

    // Reset counters
    Alloc::allocCount = 0;
    Alloc::deallocCount = 0;

    // Construct map with our counting allocator
    MapT m(
        /*initBuckets=*/ 8,
                         /*maxLoadFactor=*/ 1.0,
                         std::hash<int>{},
                         std::equal_to<int>{},
                         Alloc{}
    );

    constexpr int N = 10;
    // 1) Insert N elements
    for (int i = 0; i < N; ++i) {
        m.insert(i, std::to_string(i));
    }
    std::cout << "After insert, allocCount = " << Alloc::allocCount
            << " (expected " << N << ")\n";
    assert(Alloc::allocCount == N);

    // 2) Remove them one by one
    for (int i = 0; i < N; ++i) {
        const bool erased = m.remove(i);
        assert(erased);
    }
    std::cout << "After remove, deallocCount = " << Alloc::deallocCount
            << " (expected " << N << ")\n";
    assert(Alloc::deallocCount == N);

    std::cout << "[OK] Custom Allocator test passed\n";
}

#include <chrono>
#include <random>

//------------------------------------------------------------------------------
// Benchmark for find_n_nodes
void testFindNNodesPerformance() {
    using namespace std::chrono;

    std::cout << "\n=== Benchmark: find_n_nodes ===\n";

    // 1) build a map of size N
    constexpr size_t N = 100000;
    DoublyLinkedCircularHashMap<int, int> m(/*initBuckets=*/2 * N, /*maxLoadFactor=*/1.0);
    for (int i = 0; i < static_cast<int>(N); ++i) {
        m.insert(i, i);
    }

    // 2) prepare a reproducible RNG for picking random indices with no duplicates
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution dist(0, static_cast<int>(N) - 1);

    // 3) choose a variety of M values (number of requests)
    const std::vector<size_t> Mvals = {
        1,
        10,
        100,
        1000,
        N / 10,
        N / 2,
        N - 1
    };

    // 4) for each M, generate M random indices, time find_n_nodes, print
    for (const size_t M: Mvals) {
        // sample M random requests
        std::vector<int> req;
        req.reserve(M);
        // fill with random indices no repeats allowed and no index > N
        for (size_t i = 0; i < M; ++i) {
            req.push_back(dist(rng));
        }

        // warm-up
        volatile auto dummy = m.find_n_nodes(req, /*pre_sorted=*/false, /*verbose=*/false, true);

        // timed run
        auto t0 = steady_clock::now();
        auto out = m.find_n_nodes(req, /*pre_sorted=*/false, /*verbose=*/false, true);
        auto t1 = steady_clock::now();

        // sanity check: we got back M pointers
        assert(out.size() == M);

        const auto us = duration_cast<microseconds>(t1 - t0).count();
        const double bound = static_cast<double>(M) / static_cast<double>(M + 1) * static_cast<double>(N - 1);

        std::cout
                << " M=" << M
                << "  time=" << us << " µs"
                << "  bound≈" << static_cast<size_t>(bound) << " visits\n";
    }

    std::cout << "[DONE] find_n_nodes benchmark\n";
}

int main() {
    //call the tests
    std::cout << "=== Starting tests ===\n";
    testFindNNodesPerformance();
    testAllocatorSupport();
    test_zigzag();
    DLCHM_findNNodes_smokeTest();
    testUniqueErase();
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

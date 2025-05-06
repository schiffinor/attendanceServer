/**
 * @file    DoublyLinkedCircularHashMap.hpp
 * @author  Román Schiffino  (schiffinor)
 * @email   schiffinoroman@gmail.com
 * @version 1.0  ––– first *full* release of my first C++ project.
 * @brief   “Hash‑map meets doubly‑linked list, they move in together, adopt
 *          cache‑friendly nodes and live happily ever after.”
 *
 * ──────────────────────────────────────────────────────────────────────────────
 *  Hi!  Román here.  I’m still learning modern C++ (prior to this the most
 *  I had written was a projectile motion calculator and a helper function for
 *  calculating pairwise minkowski distances in python a la scipy cdist) – so I
 *  decided to throw myself into the deep-end by creating a STL-like mash up a
 *  hash‑table, a circular doubly‑linked list with as many bells and whistles as
 *  I could think of.  That Frankenstein is the **DoublyLinkedCircularHashMap**
 *  you are reading.
 *
 *  Why bother?
 *  ───────────
 *  • **O(1)** average insert / erase / contains thanks to the hash table.
 *  • **Stable insertion order** thanks to the circular list – perfect for LRU
 *    caches, task queues, “remember the order I gave you!” situations, etc.
 *  • **Bidirectional iterators**: walk it like a list, still find like a map.
 *  • **Allocator‑aware**: pass your own allocator (I even wrote a counting
 *    one in the tests).
 *  • **Cache‑aware**: every `Node` is aligned to a cache‑line boundary
 *    (64 bytes – or `std::hardware_destructive_interference_size` if the
 *    compiler knows it), so traversals prefetch nicely.
 *  • **Heterogeneous lookup** à la <unordered_map> – pass `std::string_view`
 *    to a `std::string` map, etc.
 *  • A metric ton of “extra” features: positional swaps, rotation, split/
 *    splice, stack/queue helpers, bucket histograms, integrity validator…
 *
 *  An ode to `find_n_nodes`
 *  ───────────────────────────────
 *  The proudest algorithm in here is `find_n_nodes`, a bulk‑lookup that digs
 *  up *M* arbitrary positions from our *N*‑element list using a greedy zig‑zag
 *  walk from both ends.  Instead of naively walking *O(M·N)* nodes, it touches
 *  at most
 *
 *      (M / (M + 1)) · (N – 1)
 *
 *  nodes – so if you only ask for a handful (M ≪ N) you pay almost nothing,
 *  and even the worst case (M≈N) degrades gracefully to a single full pass.
 *  There’s optional profiling output to prove it, and the benchmark in
 *  `main.cpp` (see `testFindNNodesPerformance`) times it so you can watch the
 *  bound in action.  Internally it:
 *
 *    1. Normalises every requested index modulo *N* (negatives welcome).
 *    2. Sorts & dedupes (unless you insist on duplicates).
 *    3. Maintains two cursors – one from `head_`, one from `tail_` – and always
 *       walks the cheaper side next, alternately stealing from the left and
 *       right demand queues (the “zig‑zag”).
 *    4. Records the total distance walked so we can shout “Told you so!” in
 *       the profiler.
 *
 *  The result comes back as the same container type you passed in, just
 *  re‑bound to `Node*` – a neat template trick so vectors stay vectors, sets
 *  stay sets, and so on.
 *
 *  Quick‑start cheat‑sheet
 *  ───────────────────────
 *  ```cpp
 *  DoublyLinkedCircularHashMap<std::string,int> m;
 *  m.insert("one", 1);          // append
 *  m.insert_at("zero", 0, 0);   // prepend
 *  m["two"] = 2;                // operator[] inserts default if absent
 *
 *  for (auto& [k,v] : m) { … }  // prints zero, one, two
 *
 *  // Bulk lookup: give me nodes 0 and ‑1 (front & back)
 *  auto nodes = m.find_n_nodes(std::vector<int>{0, -1});
 *  std::cout << nodes[0]->key_ << ", "<< nodes[1]->key_ << '\n';
 *  ```
 *
 *  Pro‑tips / gotchas
 *  ──────────────────
 *  • The container is **circular**: `head_->prev_ == tail_` and vice‑versa.
 *    Never compare to `nullptr` when you chase `next_`/`prev_` – use the list’s
 *    end iterator if you need a sentinel.
 *  • `erase()` is iterator‑friendly and returns the next iterator, but the
 *    raw `Node*` you get from other helpers becomes *invalid* after any
 *    structural change.  Be smart.
 *  • Hit `validate()` in debug builds if you suspect mis‑wiring; it walks the
 *    list & buckets and throws on anything fishy.
 *  • Custom allocators must satisfy the usual STL rules (rebind, etc.).  My
 *    counting allocator in the unit‑tests is a minimal example.
 *
 *  Roadmap / TODO
 *  ──────────────────────────────────────────────
 *  • Strong exception‑safety for *every* mutator (most already are).
 *  • Transparent node recycling pool for constant‑time erase/insert without
 *    visiting `operator new`.
 *
 *  License
 *  ───────
 *  Beer‑ware: If we ever meet and idrk, I don't imagine people really using this,
 *  but in the rare occasion where not only do you value this, but that you want
 *  to thank me in some way, and we meet, buy me a beer or something.
 *  Otherwise use it however you like.
 *
 *  Thanks for checking out my code!  –Román
 *
 *  ─────────────────────────────────────────────────────────────────────────────
 *  A small note on the code:
 *  I am quite passionate about optimization, performance, efficiency, and generality.
 *  I have a degree in mathematics and although I have been coding for a long time
 *  and have had my small share of CS classes, I am not a CS major. I never dealt with
 *  C or C++ in school, everything I learned was self-taught. I referenced Schildt's
 *  C++ A Beginner's Guide, Filipek's C++ 17 In Detail, and the C++ standard library
 *  references. I also referenced a lot of the STL code. In any case once again I
 *  am NOT a CS major, I don't know if everything I did is "correct" or "best practice"
 *  I know how computers work and I have a good head for code, but some of the things
 *  like the cache line size and the alignment stuff I just did because I thought I
 *  could eke out a bit more performance. I know what it does, but I don't know if
 *  I did it right or if it is the best way to do it. I also don't know if it is
 *  the best way to do it in C++. Also, I have two toolchains set-up, one for
 *  WSL and one for Windows. I used WSL for most of the development, but this file
 *  is part of a larger project that I am working on that will be for Windows, so I
 *  hope everything works well. I use a rather customized Windows 11 environment,
 *  on a very beefy workstation. All I know is that this works on my machine and
 *  that the algorithms are mathematically sound. Please let me know if you
 *  have any questions or if you find any bugs. I am always looking to improve
 *  and learn more about C++ and programming in general. I am also always looking
 *  for ways to improve my code and make it better. SO PLEASE, if you have any
 *  ideas, notes, or suggestions, please let me know.
 *  ─────────────────────────────────────────────────────────────────────────────
 *  P.S. <<
 *  One little thing I would like to mention is that while I did write up this
 *  code myself and I did come up with all these ideas and algorithms, myself,
 *  aside from the basic CS stuff, and STL conventions, I did use ChatGPT in
 *  _ main ways:
 *  1. I am tremendously bad at writing documentation, so I asked it to help me
 *  with a bunch of that stuff. I try to add my personal flair and I also
 *  provided a strict template for it to use, but at the end of the day most of
 *  it has been heavily written by AI.
 *  2. I asked it to help me with the unit tests. Again, that's just really boring.
 *  Pretty much all of the tests are written by AI, but I did write a few of them
 *  and I did have to edit most of them to make them work.
 *  3. I used it to have someone to talk my ideas through with. I would walk it
 *  through my ideas and it would help me clarify them, or point out things I
 *  was missing. Really it just helped me formalize my ideas a bit.
 *  4. I used it to reformat some of the code. I've coded in a lot of languages
 *  but Python is my bread and butter, I've been coding python since I was like 11.
 *  I had no clue what this file "should've" looked like, so I asked it to help me
 *  make it all nice and pretty, as well as to clean up some of the code. However,
 *  this was often more trouble than it was worth as I would implement what I
 *  thought was only a style change and then it would break the code. Like the
 *  countless times it butchered my find_n_nodes function. Like it just really
 *  couldn't understand the algorithm I came up with, because like while my algorithm
 *  isn't exactly novel (the basic algorithm and research has existed for a long time,
 *  I found out after developing it that its closely related to the
 *  Linear Tape Scheduling Problem), it is a pretty unique implementation and application.
 *  Thus it was really hard for it to understand what I was trying to do. So I had
 *  to go line by line and fix the code while maintaining the style, because yeah,
 *  it looked a lot prettier than what I had wrote. It is my algorithm though lol.
 *  5. I used it to suggest features to implement. Basically, whenever I had just
 *  finished a big feature, I would feed it my whole file and ask, "what features
 *  should I implement next?" and it would give me a list of features to implement.
 *  I would then do my best to implement each feature, doing my research and yeah.
 *  6. I used it to come up with names for stupid helper functions and variables. I
 *  had no idea what to call half of the stuff I was writing, in fact if you look at
 *  the deprecated functions, you'll see what I mean. For example, if you look at
 *  shift_idx_og you may lay eyes on some of my beautifully named variables.
 *  At one point i legitimately had a block of variables named:
 *   - ridx_mod_pnt_idx
 *   - lidx_mod_pnt_idx
 *   - _ridx_mod_pnt_idx
 *   - _lidx_mod_pnt_idx
 *   - etc...
 *  It was pretty bad and hard to make sense of for anyone who wasn't me, so I
 *  asked the AI to just make up some names.
 *  7. Lat but not least, I used it in debugging to help me root through the walls
 *  of errors and debug info I was getting. Realistically, it just helped to have
 *  something parse through the text and help me find the root problems so I could then
 *  research the problem and implement a fix.
 *
 *  Regardless, I am very proud of this code and I think it's great. At the end of the
 *  day, I wrote this code and I am the one who came up with the ideas and algorithms. I
 *  am pretty proud of this code, especially the find_n_nodes function.
 *
 *  So yeah, I hope you enjoy this code and I hope it helps you in your projects.
 * ──────────────────────────────────────────────────────────────────────────────
 */


#ifndef DOUBLYLINKEDCIRCULARHASHMAP_HPP
#define DOUBLYLINKEDCIRCULARHASHMAP_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <cmath>
#include <ranges>
#include <unordered_set>
#include <type_traits>
#include <concepts>

// Branch prediction hints for GCC/Clang (no-op on other compilers)
#if defined(__GNUC__) || defined(__clang__)
# define LIKELY(x)   (__builtin_expect(!!(x), 1))
# define UNLIKELY(x) (__builtin_expect(!!(x), 0))
# define NOEXCEPT noexcept
#else
# define LIKELY(x)   (x)
# define UNLIKELY(x) (x)
#endif

template<typename R>
concept IntRange =
        std::ranges::input_range<R> &&
        std::integral<std::ranges::range_value_t<R> >;

template<typename C>
concept Reservable =
        requires(C c, std::size_t n) { c.reserve(n); };


/**
 * @brief A combined hash‐map and doubly‐linked list container in one tidy package.
 *
 * DoublyLinkedCircularHashMap provides constant‐time lookups, insertions, and removals
 * via an internal hash table, while also preserving element insertion order through
 * a circular doubly‐linked list. Think of it as a hashmap and a linkedlist living in
 * perfect harmony—without the mess of separate data structures.
 *
 * Key features:
 *   - Average O(1) complexity for operator[], at(), insert(), erase(), and contains().
 *   - Iteration in insertion order, forwards or backwards, with bidirectional iterators.
 *   - Circular linkage ensures no “null” checks when traversing—just wraparound magic.
 *   - Customizable hashing and equality through template parameters.
 *
 * @tparam Key     Type of the keys. Must be hashable by Hash and comparable by KeyEq.
 * @tparam Value   Type of the mapped values.
 * @tparam Hash    Hash functor type; defaults to std::hash<Key>.
 * @tparam KeyEq   Equality predicate type; defaults to std::equal_to<Key>.
 *
 * Example usage:
 *   DoublyLinkedCircularHashMap<std::string, int> m;
 *   m.insert({"foo", 42});
 *   for (auto &p : m) {
 *       // Iterates in insertion order: ("foo", 42)
 *   }
 */
template<
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEq = std::equal_to<Key>,
    typename Alloc = std::allocator<std::pair<const Key, Value> > >
class DoublyLinkedCircularHashMap {
    //───────────────────────────────────────────────────────────────────────────//
    // ----- Node structure -----

    /**
     * @brief Node structure for DoublyLinkedCircularHashMap.
     *
     * Aligns each node to a 64-byte boundary to optimize cache usage and prevent
     * false sharing. Each Node participates in both a circular doubly-linked list
     * for maintaining insertion order and a separate doubly-linked chain for its
     * hash bucket, enabling O(1) average complexity for lookups, insertions, and
     * removals while preserving iteration order.
     *
     * @tparam Key   Type of the key stored in the map.
     * @tparam Value Type of the value associated with the key.
     */
    struct Node;
    using node_allocator_t = typename std::allocator_traits<Alloc>::template rebind_alloc<Node>;
    node_allocator_t alloc_;
#include <new>   // for hardware_*_interference_size
#if defined(__cpp_lib_hardware_interference_size)
    struct alignas(std::hardware_destructive_interference_size) Node {
#else
    constexpr std::size_t CacheLineSize = 64;
    struct alignas(CacheLineSize) Node {
#endif
        Key key_; /**< The key for this node. Immutable after construction. */
        Value value_; /**< The value mapped to key_. Move-constructed for efficiency. */
        //----- Circular doubly-linked list pointers (maintain insertion order) -----
        Node *next_; /**< Next node in insertion order; wraps around to head_ if at tail_. */
        Node *prev_; /**< Previous node in insertion order; wraps to tail_ if at head_. */
        //----- Hash bucket chaining pointers (for collision resolution) -----
        Node *hashNext_; /**< Next node in the same hash bucket chain. nullptr if last in bucket. */
        Node *hashPrev_; /**< Previous node in the same hash bucket chain. nullptr if first in bucket. */
        /**
         * @brief Construct a new Node with given key and value.
         *
         * Initializes both circular list pointers to point to itself, forming a singleton
         * circular list, and nullifies bucket pointers until insertion into a bucket.
         *
         * @param key    The key to store (copied).
         * @param value  The value to store (moved for efficiency).
         */
        Node(const Key &key, Value value)
            : key_(key),
              value_(std::move(value)),
              next_(this), // Self-referential: node is its own next in empty list
              prev_(this), // Self-referential: node is its own prev in empty list
              hashNext_(nullptr), // Not in a bucket yet
              hashPrev_(nullptr) {
            // No further initialization needed—ready to be linked in
        }
    };


    //────────────────────────────────────────────────────────────────────────//
    //----- Internal data members for DoublyLinkedCircularHashMap -----

    std::vector<Node *> htBaseVector_; /**< Heads of each hash bucket chain. Size = number of buckets. */
    std::vector<size_t> bucketSizes_; /**< Number of nodes currently in each bucket, for diagnostics. */

    Node *head_ = nullptr; /**< First node in insertion order; nullptr if map is empty. */
    Node *tail_ = nullptr; /**< Last node in insertion order; nullptr if map is empty. */
    size_t size_ = 0; /**< Total number of elements currently in the map. */

    double maxLoadFactor_ = 1.0; /**< Threshold (size_/bucket_count_) to trigger rehashing. */

    std::function<size_t(const Key &)> hashFunc_; /**< User-specified or default hash functor (std::hash). */
    std::function<bool(const Key &, const Key &)> keyEqFunc_; /**< Equality comparator for Key. */

    size_t rehashCount_ = 0; /**< Number of times the table has been rehashed. Useful for profiling. */
    size_t maxBucketSize_ = 0; /**< Largest bucket size observed since last rehash. */
    size_t largestBucketIdx_ = SIZE_MAX; /**< Index of the bucket that currently has maxBucketSize_. */

    //────────────────────────────────────────────────────────────────────────//
    //----- Private helpers -----

    /**
     * @brief Returns how many times the hash table has been resized (rehashes).
     * @return Total number of rehash operations performed.
     */
    [[nodiscard]] size_t rehashCount() const {
        return rehashCount_; ///< Direct accessor—no surprises here.
    }

    /**
     * @brief Gets the number of elements currently in a specific bucket.
     * @param idx Zero-based bucket index.
     * @throws std::out_of_range if idx is not a valid bucket.
     * @return Element count of the bucket at index idx.
     */
    [[nodiscard]] size_t bucketSize(size_t idx) const {
        if (idx >= htBaseVector_.size()) {
            // Guard against rogue indices—better to crash than corrupt memory.
            throw std::out_of_range("Index out of range");
        }
        return bucketSizes_[idx]; ///< Recorded size for this bucket.
    }

    /**
     * @brief Fetches the largest bucket size observed.
     * @return Maximum number of entries in any single bucket.
     */
    [[nodiscard]] size_t largestBucketSize() const {
        return maxBucketSize_; ///< Handy for diagnostics or tuning load factor.
    }

    /**
     * @brief Retrieves the index of the bucket that currently holds the most elements.
     * @return Index of the largest bucket, or SIZE_MAX if map is empty.
     */
    [[nodiscard]] size_t largestBucketIdx() const {
        return largestBucketIdx_; ///< SIZE_MAX signals “no buckets yet.”
    }

    /**
     * @brief Provides read-only access to all bucket sizes.
     * @return A const reference to the vector of per-bucket element counts.
     */
    [[nodiscard]] const std::vector<size_t> &bucketSizes() const {
        return bucketSizes_; ///< Useful for monitoring distribution.
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Hashing and bucket management functions -----

    /**
     * @brief Computes which bucket a given key belongs in.
     * @param key The key whose bucket we want.
     * @return Index in htBaseVector_ after hashing and modulo.
     */
    [[nodiscard]] size_t bucketIndex_(const Key &key) const {
        // Hash + modulo ensures uniform spread and wraparound.
        return hashFunc_(key) % htBaseVector_.size();
    }

    /**
     * @brief Heterogeneous bucket index computation.
     * @param k Key of a different type.
     * @return Index in htBaseVector_ after hashing and modulo.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
    size_t bucketIndex_(const K2 &k) const {
        return hashFunc_(k) % htBaseVector_.size();
    }

    /**
     * @brief Inserts a node into its proper bucket using separate chaining.
     * @param node Node to insert; we’ll splice it into the head of its chain.
     *
     * New nodes go to the front for O(1) insertion. Updates bucketSizes_, and
     * if this bucket grows beyond the old max, updates maxBucketSize_ and
     * largestBucketIdx_.
     */
    void bucketInsert_(Node *node) {
        auto idx = bucketIndex_(node->key_);
        Node *head = htBaseVector_[idx];
        if (!head) {
            // Empty bucket → node stands alone.
            htBaseVector_[idx] = node;
            node->hashNext_ = node->hashPrev_ = nullptr;
        } else {
            // Prepend node to existing chain.
            node->hashNext_ = head;
            head->hashPrev_ = node;
            node->hashPrev_ = nullptr;
            htBaseVector_[idx] = node;
        }
        ++bucketSizes_[idx]; // Keep the count honest.
        // Track maximum chain length.
        if (bucketSizes_[idx] > maxBucketSize_) {
            maxBucketSize_ = bucketSizes_[idx];
            largestBucketIdx_ = idx;
        }
    }

    /**
     * @brief Removes a node from its bucket chain, fixing up links.
     * @param node Node to remove; will be detached from its bucket.
     * @throws std::runtime_error if a size underflow is detected.
     *
     * Decrements bucketSizes_. If removing from the largest bucket,
     * rescans all buckets to find the new maximum.
     */
    void bucketRemove_(Node *node) {
        auto idx = bucketIndex_(node->key_);
        if (node->hashPrev_) {
            node->hashPrev_->hashNext_ = node->hashNext_;
        } else {
            // Node was head of chain → move head pointer.
            htBaseVector_[idx] = node->hashNext_;
        }
        if (node->hashNext_) {
            node->hashNext_->hashPrev_ = node->hashPrev_;
        }
        node->hashPrev_ = node->hashNext_ = nullptr; // fully detach

        if (bucketSizes_[idx] > 0) {
            --bucketSizes_[idx];
        } else {
            // I don't think this can happen, but just in case...
            throw std::runtime_error("Bucket size is already 0");
        }

        // If we just shrank the largest bucket, recompute the max.
        if (idx == largestBucketIdx_ && bucketSizes_[idx] < maxBucketSize_) {
            maxBucketSize_ = 0;
            largestBucketIdx_ = SIZE_MAX;
            for (size_t i = 0; i < htBaseVector_.size(); ++i) {
                if (bucketSizes_[i] > maxBucketSize_) {
                    maxBucketSize_ = bucketSizes_[i];
                    largestBucketIdx_ = i;
                }
            }
        }
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Circular list management functions -----

    /**
     * @brief Inserts a node into the circular list immediately after `where`.
     * @param where Existing node after which to insert.
     * @param node  New node to splice in.
     *
     * Updates prev/next pointers. If we link after the tail, tail_ is bumped.
     */
    void linkAfter_(Node *where, Node *node) {
        node->prev_ = where;
        node->next_ = where->next_;
        where->next_->prev_ = node;
        where->next_ = node;
        if (where == tail_) {
            tail_ = node; // new end of list
        }
    }

    /**
     * @brief Inserts a node into the circular list immediately before `where`.
     * @param where Existing node before which to insert.
     * @param node  New node to splice in.
     *
     * Updates prev/next pointers. If we link before the head, head_ is bumped.
     */
    void linkBefore_(Node *where, Node *node) {
        node->next_ = where;
        node->prev_ = where->prev_;
        where->prev_->next_ = node;
        where->prev_ = node;
        if (where == head_) {
            head_ = node; // new front of list
        }
    }

    /**
     * @brief Detaches a node from the circular list.
     * @param node Node to unlink. Its pointers remain valid but list skips it.
     *
     * Adjusts head_ and tail_ if you just removed one of them.
     */
    void unlink_(Node *node) {
        node->prev_->next_ = node->next_;
        node->next_->prev_ = node->prev_;
        if (node == head_) {
            head_ = node->next_;
        }
        if (node == tail_) {
            tail_ = node->prev_;
        }
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Rehashing and resizing functions -----

    /**
     * @brief Resizes the bucket array, rehashing every element.
     * @param newBucketCount Desired number of buckets after resize.
     *
     * Allocates a fresh table, clears bucket stats, then reinserts every
     * node in insertion order (circular list traversal). Increments rehashCount_.
     */
    void rehash_(size_t newBucketCount) {
        std::vector<Node *> newTable(newBucketCount, nullptr);
        htBaseVector_.swap(newTable);
        bucketSizes_.assign(newBucketCount, 0);
        maxBucketSize_ = 0;
        largestBucketIdx_ = SIZE_MAX;

        if (head_) {
            Node *cur = head_;
            do {
                cur->hashNext_ = cur->hashPrev_ = nullptr; // reset chain links
                bucketInsert_(cur);
                cur = cur->next_;
            } while (cur != head_);
        }
        ++rehashCount_; // document that we rehashed
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Static Utilities -----

    // ----- Pointer walking functions -----

    /**
     * @brief Advances a single pointer forward or backward, with cache hints.
     * @param start   Starting node.
     * @param steps   Number of hops to make.
     * @param forward true → use next_; false → use prev_.
     * @return Pointer after walking steps times.
     * @note Uses __builtin_prefetch to hint CPU caching.
     */
    static Node *walk(Node *__restrict__ start, int steps, bool forward) noexcept {
        while (steps--) {
            __builtin_prefetch(forward ? start->next_ : start->prev_, 0, 1);
            start = forward ? start->next_ : start->prev_;
        }
        return start;
    }

    /**
     * @brief Simultaneously walks multiple node pointers in bulk.
     * @tparam Container A container type holding Node* elements.
     * @param starts  Original pointers.
     * @param steps   Steps to advance each pointer.
     * @param forward Direction of walk.
     * @return New container with updated pointers.
     * @note Prefetch hints in the inner loop for each node.
     */
    template<typename Container>
    static Container multi_walk(const Container &starts, int steps, bool forward) noexcept {
        Container cur = starts; // copy to avoid mutating caller data
        while (--steps) {
            for (auto &n: cur) {
                __builtin_prefetch(forward ? n->next_ : n->prev_, 0, 1);
                n = forward ? n->next_ : n->prev_;
            }
        }
        return cur;
    }

public:
    //───────────────────────────────────────────────────────────────────────────//
    // ----- Constructors and destructors -----

    /**
     * @brief Constructs a new DoublyLinkedCircularHashMap with custom parameters.
     *
     * Initializes the internal bucket array, bucket size tracking, and list pointers.
     * Uses provided hash and key‐equality functors, moving them into place to avoid extra copies.
     *
     * @param initBuckets    Initial number of hash buckets (default: 16). Must be > 0.
     * @param maxLoadFactor  Maximum allowed load factor before automatic rehash (default: 1.0).
     * @param hashFunc       Hash functor to map keys to size_t (default: std::hash<Key>).
     * @param keyEqFunc      Equality comparator for keys (default: std::equal_to<Key>).
     * @param alloc          Allocator for Node objects (default: std::allocator<Node>).
     */
    explicit DoublyLinkedCircularHashMap(
        size_t initBuckets = 16,
        const double maxLoadFactor = 1.0,
        std::function<size_t(const Key &)> hashFunc = std::hash<Key>(),
        std::function<bool(const Key &, const Key &)> keyEqFunc = std::equal_to<Key>(),
        const Alloc &alloc = Alloc{}
    )
        : alloc_(alloc), // all buckets start empty
          htBaseVector_(initBuckets, nullptr), // track 0 elements per bucket
          bucketSizes_(initBuckets, 0), // empty list → no head
          head_(nullptr), // empty list → no tail
          tail_(nullptr), // store user’s max load factor
          maxLoadFactor_(maxLoadFactor), // move-in hash functor
          hashFunc_(std::move(hashFunc)), // move-in equality functor
          keyEqFunc_(std::move(keyEqFunc)) // store allocator
    {
        // Nothing else to do here—size_ and rehashCount_ default to 0.
    }

    /**
     * @brief Destructor: cleans up all dynamically allocated nodes.
     *
     * Calls clear(), which walks the circular list, deletes each Node,
     * and resets internal state.
     */
    ~DoublyLinkedCircularHashMap() {
        clear(); // delete every node, reset size_ to zero, clear buckets
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Copy constructors and assignment -----

    /**
     * @brief Copy constructor: deep-copies another map’s contents.
     *
     * Allocates a fresh bucket array of the same size, then iterates the other map
     * in insertion order and re-inserts each key/value pair. Load factors and
     * hash/equality functors are copied.
     *
     * @param other Map to copy from.
     */
    DoublyLinkedCircularHashMap(const DoublyLinkedCircularHashMap &other)
        : htBaseVector_(other.htBaseVector_.size(), nullptr), // new empty buckets
          head_(nullptr), // list will be rebuilt
          tail_(nullptr),
          maxLoadFactor_(other.maxLoadFactor_), // copy load factor
          hashFunc_(other.hashFunc_), // copy functor
          keyEqFunc_(other.keyEqFunc_) // copy comparator
    {
        if (other.head_) {
            Node *cur = other.head_;
            do {
                // insert() handles both hashing and list-linking
                insert(cur->key_, cur->value_);
                cur = cur->next_;
            } while (cur != other.head_);
        }
    }

    /**
     * @brief Copy-assignment operator: clears and then deep-copies from other.
     *
     * Provides strong exception safety by first clearing this map, then
     * assigning new buckets and load factor, finally reinserting every element.
     *
     * @param other Source map to copy.
     * @return Reference to *this.
     */
    DoublyLinkedCircularHashMap &operator=(const DoublyLinkedCircularHashMap &other) {
        if (this != &other) {
            clear(); // free existing nodes and reset state
            // Resize buckets to match ’other’
            htBaseVector_.assign(other.htBaseVector_.size(), nullptr);
            maxLoadFactor_ = other.maxLoadFactor_;
            hashFunc_ = other.hashFunc_;
            keyEqFunc_ = other.keyEqFunc_;
            // Rebuild list and hash table
            if (other.head_) {
                Node *cur = other.head_;
                do {
                    insert(cur->key_, cur->value_);
                    cur = cur->next_;
                } while (cur != other.head_);
            }
        }
        return *this;
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Move constructors and assignment -----

    /**
     * @brief Move constructor: takes ownership of resources from another map.
     *
     * Transfers bucket array, list pointers, size, and functors. Leaves other map
     * in an empty-but-valid state (head_ and tail_ set to nullptr, size_ zero).
     *
     * @param other Map to move from.
     */
    DoublyLinkedCircularHashMap(DoublyLinkedCircularHashMap &&other) noexcept
        : htBaseVector_(std::move(other.htBaseVector_)), // steal bucket vector
          head_(other.head_), // copy head pointer
          tail_(other.tail_), // copy tail pointer
          size_(other.size_), // copy size count
          maxLoadFactor_(other.maxLoadFactor_), // copy load factor
          hashFunc_(std::move(other.hashFunc_)), // move-in hash functor
          keyEqFunc_(std::move(other.keyEqFunc_)) // move-in equality functor
    {
        // Reset other to empty state so its destructor is safe.
        other.head_ = other.tail_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Move-assignment operator: swaps this map’s contents with other.
     *
     * Efficiently exchanges bucket arrays, list pointers, and functors.
     * Leaves other with prior contents of *this.
     *
     * @param other Map to move-assign from.
     * @return Reference to *this.
     */
    DoublyLinkedCircularHashMap &operator=(DoublyLinkedCircularHashMap &&other) noexcept {
        swap(other); // leverage strong swap for all internals
        return *this;
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Observers -----

    /**
     * @brief Check whether the map contains no elements.
     *
     * Uses the stored size_ counter for an O(1) check.
     *
     * @return true if size_ == 0, false otherwise.
     */
    [[nodiscard]] bool empty() const {
        return size_ == 0; // Fast path: compare against zero
    }

    /**
     * @brief Get the number of elements stored in the map.
     *
     * @return Current element count (size_).
     */
    [[nodiscard]] size_t size() const {
        return size_; // Directly return the counter
    }

    /**
     * @brief Compute the current load factor of the hash table.
     *
     * Load factor = number of elements / number of buckets.
     *
     * @return A double in [0, ∞); typically you aim to keep this ≤ maxLoadFactor_.
     */
    [[nodiscard]] double loadFactor() const {
        return static_cast<double>(size_) / htBaseVector_.size(); // dynamic measure of fullness
    }

    /**
     * @brief Retrieve the number of buckets in the hash table.
     *
     * @return Size of htBaseVector_ (bucket count).
     */
    [[nodiscard]] size_t bucketCount() const {
        return htBaseVector_.size(); // slots available for chaining
    }

    /**
     * @brief Get the maximum load factor threshold.
     *
     * When loadFactor() exceeds this, the table will rehash.
     *
     * @return The stored maxLoadFactor_.
     */
    [[nodiscard]] double maxLoadFactor() const noexcept {
        return maxLoadFactor_; // no side effects
    }

    /**
     * @brief Set a new maximum load factor and rehash if currently exceeded.
     *
     * @param newMaxLoadFactor Desired load factor threshold.
     */
    void maxLoadFactor(const double newMaxLoadFactor) {
        maxLoadFactor_ = newMaxLoadFactor; // update threshold
        // If we're already above the new threshold, redistribute now
        if (const auto need = static_cast<size_t>(std::ceil(static_cast<double>(size_) / newMaxLoadFactor)); need > bucketCount()) {
            rehash_(bucketCount());
        }
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- Capacity -----

    /**
     * @brief Reserve capacity for at least newSize elements without rehashing.
     *
     * Computes the minimal number of buckets required to keep loadFactor() ≤ maxLoadFactor_,
     * then calls rehash_ to adjust the table size.
     *
     * @param newSize Minimum number of elements to accommodate.
     */
    void reserve(const size_t newSize) {
        const size_t newBucketCount = std::max<size_t>(
            1,
            static_cast<size_t>(std::ceil(static_cast<double>(newSize) / maxLoadFactor_))
        );
        if (newBucketCount > bucketCount()) {
            // Only rehash if we need more buckets
            rehash_(newBucketCount);
        }
    }

    /**
     * @brief Resize the hash table to a new number of buckets.
     *
     * This will rehash all existing elements into the new bucket array.
     * If newSize is less than the current size, it will throw an exception.
     *
     * @param newSize New number of buckets.
     */
    void rehash(const size_t newSize) {
        if (newSize == bucketCount()) return;
        rehash_(newSize); // rehash to new size
    }

    /**
     * @brief Remove all elements from the map, deleting every Node.
     *
     * Traverses the circular list in insertion order, deletes each node,
     * then resets head_, tail_, size_, and clears all bucket heads.
     */
    void clear() {
        if (head_) {
            // Walk from head_->next_ until we circle back
            Node *cur = head_->next_;
            while (cur != head_) {
                Node *nxt = cur->next_;
                std::allocator_traits<node_allocator_t>::destroy(alloc_, cur); // free each node
                std::allocator_traits<node_allocator_t>::deallocate(alloc_, cur, 1); // deallocate
                cur = nxt;
            }
            delete head_; // finally delete the original head
        }
        // Reset internal state
        head_ = tail_ = nullptr;
        size_ = 0;
        std::fill(htBaseVector_.begin(), htBaseVector_.end(), nullptr);
        std::ranges::fill(bucketSizes_, 0);
        maxBucketSize_ = 0;
        largestBucketIdx_ = SIZE_MAX;
    }

    /**
     * @brief Shrink bucket count to minimum needed for current elements.
     *
     * Calculates the bucket count that satisfies loadFactor() ≤ maxLoadFactor_.
     * If this differs from current bucketCount(), triggers a rehash.
     */
    void minimize_size() {
        const size_t minSize = std::max<size_t>(
            1,
            static_cast<size_t>(std::ceil(static_cast<double>(size_) / maxLoadFactor_))
        );
        if (minSize != bucketCount()) {
            rehash_(minSize);
        }
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- Modifiers -----

    /**
     * @brief Insert or update a key/value pair at a specific position.
     *
     * This supports insertion:
     *  - at the end (where == -1 or where == size_),
     *  - at the front (where == 0),
     *  - before the element at index i (where > 0),
     *  - or after the element at index -i - 1 (where < 0).
     *
     * @param key   Key to insert or update.
     * @param value Value to associate with the key.
     * @param where Insertion index (see above). Defaults to -1 (append).
     * @throws std::out_of_range if where ∉ [-(size_+1) ... size_].
     */
    void insert_at(const Key &key, const Value &value, const int where = -1) {
        // 1) Validate insertion index
        const int intSize = static_cast<int>(size_);
        if (where > intSize || where < -(intSize + 1)) {
            throw std::out_of_range("Index out of range");
        }

        // 2) If key already exists in its bucket, overwrite and exit
        size_t idx = bucketIndex_(key);
        for (Node *cur = htBaseVector_[idx]; cur; cur = cur->hashNext_) {
            if (keyEqFunc_(cur->key_, key)) {
                cur->value_ = value; // update existing
                return;
            }
        }

        // 3) Create a fresh node (self-linked in list)
        Node *node = std::allocator_traits<node_allocator_t>::allocate(alloc_, 1);
        std::allocator_traits<node_allocator_t>::construct(
            alloc_,
            node,
            std::move(key),
            std::move(value)
        );

        // 4) Splice into the circular doubly-linked list
        if (where == -1 || where == intSize) {
            // Append at tail
            if (!head_) {
                head_ = tail_ = node; // first element
            } else {
                linkAfter_(tail_, node);
            }
        } else if (where == 0) {
            // Prepend at head
            if (!head_) {
                head_ = tail_ = node;
            } else {
                linkBefore_(head_, node);
            }
        } else {
            // Positional insertion relative to orderedGetNode()
            if (!head_) {
                head_ = tail_ = node;
            } else {
                Node *cur = orderedGetNode(where);
                if (where >= 0) {
                    linkBefore_(cur, node);
                } else {
                    linkAfter_(cur, node);
                }
            }
        }

        // 5) Link into hash bucket chain and bump size_
        bucketInsert_(node);
        ++size_;

        // 6) Auto-rehash if we're over the load factor
        if (loadFactor() > maxLoadFactor_) {
            rehash_(htBaseVector_.size() * 2);
        }
    }

    /**
     * @brief Append or update a key/value pair at the end (insertion order).
     *
     * @param key   Key to insert or update.
     * @param value Value to associate.
     */
    void insert(const Key &key, const Value &value) {
        insert_at(key, value, -1); // convenience overload
    }

    /**
     * @brief Remove the mapping for a given key, if present.
     *
     * Searches the appropriate bucket, unlinks from both the bucket chain
     * and the circular list, deletes the node, and decrements size_.
     *
     * @param key Key to remove.
     * @return true if an element was found and removed; false otherwise.
     */
    bool remove(const Key &key) {
        size_t idx = bucketIndex_(key);
        Node *cur = htBaseVector_[idx];
        while (cur) {
            if (keyEqFunc_(cur->key_, key)) {
                // Detach from hash chain
                bucketRemove_(cur);

                // Detach from list
                if (cur == head_ && cur == tail_) {
                    head_ = tail_ = nullptr; // only element
                } else {
                    unlink_(cur);
                }

                std::allocator_traits<node_allocator_t>::destroy(alloc_, cur); // Free the node
                std::allocator_traits<node_allocator_t>::deallocate(alloc_, cur, 1); // Deallocate memory
                --size_; // decrement count
                return true;
            }
            cur = cur->hashNext_;
        }
        return false; // key not found
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Element Access -----

    /**
     * @brief Provides direct access to the mapped value, inserting a default if missing.
     *
     * If the key exists, returns a reference to its associated value. Otherwise,
     * default-constructs a Value, inserts a new mapping, and returns a reference
     * to that freshly minted value.
     *
     * @param key The key to look up or insert.
     * @return Reference to the value associated with key.
     */
    Value &operator[](const Key &key) {
        if (Value *pv = find_ptr(key)) {
            return *pv; // Fast path: key already present
        }
        insert(key, Value{}); // Splice in a default-constructed Value
        // Note: default Value may not be thrilling, but it gets the job done
        return *find_ptr(key); // Guaranteed to succeed now
    }

    /**
     * @brief Safe element access that throws on missing key.
     *
     * Looks up the key; if found, returns a reference to the mapped value.
     * Otherwise throws std::out_of_range.
     *
     * @param key The key to access.
     * @throws std::out_of_range if key is not present.
     * @return Reference to the mapped value.
     */
    Value &at(const Key &key) {
        if (auto p = find_ptr(key)) {
            return *p; // Good: key found
        }
        throw std::out_of_range("Key not found"); // Bad: no such key
    }

    /**
     * @brief Const overload of at(), throws on missing key.
     *
     * @param key The key to access.
     * @throws std::out_of_range if key is not present.
     * @return Const reference to the mapped value.
     */
    const Value &at(const Key &key) const {
        if (auto p = find_ptr(key)) {
            return *p; // Returns const reference
        }
        throw std::out_of_range("Key not found");
    }

    /**
     * @brief Heterogeneous lookup: throws if key not present.
     *
     * Allows lookup by any type K2 that Hash and KeyEq can handle
     * (e.g., std::string_view), avoiding unnecessary Key construction.
     *
     * @tparam K2 A key-like type.
     * @param key The key-like object to access.
     * @throws std::out_of_range if key is not present.
     * @return Reference to the mapped value.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    Value &at(const K2 &key) {
        if (auto p = find_ptr(key)) {
            return *p;
        }
        throw std::out_of_range("Key not found");
    }

    /**
     * @brief Const heterogeneous lookup: throws if key not present.
     *
     * @tparam K2 A key-like type.
     * @param key The key-like object to access.
     * @throws std::out_of_range if key is not present.
     * @return Const reference to the mapped value.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    const Value &at(const K2 &key) const {
        if (auto p = find_ptr(key)) {
            return *p;
        }
        throw std::out_of_range("Key not found");
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- Lookup and find functions -----

    /**
     * @brief Core lookup: find the Node holding a given key, or nullptr.
     *
     * Scans the bucket chain for an exact key match.
     *
     * @param key The key to search for.
     * @return Pointer to the Node if found, nullptr otherwise.
     */
    Node *find_node(const Key &key) {
        size_t idx = bucketIndex_(key);
        for (Node *cur = htBaseVector_[idx]; cur; cur = cur->hashNext_) {
            if (keyEqFunc_(cur->key_, key)) {
                return cur; // Found the node
            }
        }
        return nullptr; // Miss
    }

    /**
     * @brief Const overload of find_node().
     */
    const Node *find_node(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_node(key);
    }

    /**
     * @brief Heterogeneous Node lookup by K2 key-like type.
     *
     * @tparam K2 A type that Hash and KeyEq can accept.
     * @param key The key-like object to search.
     * @return Pointer to the Node if found, nullptr otherwise.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    Node *find_node(const K2 &key) {
        size_t idx = bucketIndex_(key);
        for (Node *cur = htBaseVector_[idx]; cur; cur = cur->hashNext_) {
            if (keyEqFunc_(cur->key_, key)) {
                return cur;
            }
        }
        return nullptr;
    }

    /**
     * @brief Const heterogeneous lookup overload.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    const Node *find_node(const K2 &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_node(key);
    }

    /**
     * @brief Find and return pointer to the value for a given key, or nullptr.
     *
     * Convenience wrapper around find_node() that returns &node->value_.
     *
     * @param key The key to search.
     * @return Pointer to the mapped value if found, nullptr otherwise.
     */
    Value *find_ptr(const Key &key) {
        if (Node *n = find_node(key)) {
            return &n->value_; // Handy direct pointer
        }
        return nullptr;
    }

    /**
     * @brief Const overload of find_ptr().
     */
    const Value *find_ptr(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_ptr(key);
    }

    /**
     * @brief Heterogeneous find_ptr() for key-like types.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    Value *find_ptr(const K2 &key) {
        if (Node *n = find_node(key)) {
            return &n->value_;
        }
        return nullptr;
    }

    /**
     * @brief Const heterogeneous find_ptr() overload.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    const Value *find_ptr(const K2 &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_ptr(key);
    }

    /**
     * @brief Check if a key is present in the map.
     *
     * @param key The key to test.
     * @return true if find_ptr(key) != nullptr, false otherwise.
     */
    [[nodiscard]] bool contains(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_ptr(key) != nullptr;
    }

    /**
     * @brief Heterogeneous contains() for key-like types.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    [[nodiscard]] bool contains(const K2 &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_ptr(key) != nullptr;
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Functor configuration -----

    /**
     * @brief Replace the hash‐functor and immediately rehash all entries.
     *
     * You can swap in any callable `h` that takes `const Key&` and returns `size_t`.
     * After changing the hash, we rebuild every bucket to maintain correct distribution.
     *
     * @tparam HF  Hash‐functor type (e.g. lambda, function object, etc.).
     * @param h    New hash functor; forwarded into hashFunc_.
     */
    template<class HF>
    void setHashFunction(HF &&h) {
        hashFunc_ = std::forward<HF>(h); // swap in new hash
        rehash_(htBaseVector_.size()); // rebuild buckets under new hash
    }

    /**
     * @brief Replace the key‐equality functor.
     *
     * You can swap in any callable `k` that takes `(const Key&, const Key&)`
     * (or heterogeneous overloads) and returns `bool`. No rehash needed,
     * since equality only affects comparisons, not bucket indices.
     *
     * @tparam KEF  Key‐equality functor type.
     * @param k     New equality functor; forwarded into keyEqFunc_.
     */
    template<class KEF>
    void setKeyEqFunction(KEF &&k) {
        keyEqFunc_ = std::forward<KEF>(k); // swap in new comparator
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Iterator Support -----

    // Forward declare const_iterator
    struct const_iterator;

    /**
     * @brief Bidirectional iterator for traversing in insertion order.
     *
     * Presents each element as a std::pair<const Key&, Value&>, supporting
     * prefix/postfix ++ and --. Reaching past tail_ yields end() (cur == nullptr).
     */
    struct iterator {
        using iterator_category = std::bidirectional_iterator_tag; /**< STL category tag. */
        using value_type = std::pair<const Key &, Value &>; /**< Dereferenced type. */
        using reference = value_type; /**< Reference alias. */
        using difference_type = std::ptrdiff_t; /**< Signed distance. */
        using pointer = void; /**< Pointer unsupported. */

        Node *cur; /**< Current node; nullptr means end(). */
        const DoublyLinkedCircularHashMap *map; /**< Owning map, for head_/tail_. */

        /**
         * @brief Default-construct an end() iterator.
         */
        iterator() noexcept : cur(nullptr), map(nullptr) {
        }

        /**
         * @brief Construct an iterator at the given node in the given map.
         * @param node Starting node (nullptr for end()).
         * @param m    Pointer to the map instance.
         */
        iterator(Node *node, const auto *m) : cur(node), map(m) {
        }

        /**
         * @brief Prefix increment: advance to the next element.
         * @return Reference to this iterator, now pointing to the next element.
         */
        iterator &operator++() {
            if (cur) {
                cur = cur->next_;
                if (cur == map->head_) {
                    cur = nullptr; // wrapped past tail_ → end()
                }
            }
            return *this;
        }

        /**
         * @brief Postfix increment: advance to the next element, return old.
         * @return Iterator pointing to the element before increment.
         */
        iterator operator++(int) {
            iterator tmp = *this;
            ++*this;
            return tmp;
        }

        /**
         * @brief Prefix decrement: move to the previous element.
         * @return Reference to this iterator, now pointing to the previous element.
         */
        iterator &operator--() {
            if (!cur) {
                cur = map->tail_; // end() → last element
            } else {
                cur = cur->prev_;
            }
            return *this;
        }

        /**
         * @brief Postfix decrement: move to the previous element, return old.
         * @return Iterator pointing to the element before decrement.
         */
        iterator operator--(int) {
            iterator tmp = *this;
            --*this;
            return tmp;
        }

        /**
         * @brief Compare two iterators for equality.
         * @param other Other iterator.
         * @return true if both point to the same element (or both end()).
         */
        bool operator==(const iterator &other) const {
            return cur == other.cur;
        }

        /**
         * @brief Compare two iterators for inequality.
         * @param other Other iterator.
         * @return true if they point to different elements.
         */
        bool operator!=(const iterator &other) const {
            return cur != other.cur;
        }

        /**
         * @brief Dereference to access the key/value pair.
         * @return std::pair<const Key&, Value&> of the current element.
         */
        reference operator*() const {
            return {cur->key_, cur->value_};
        }

        explicit operator const_iterator() const noexcept {
            return const_iterator(cur, this);
        }
    };

    /**s
     * @brief Const bidirectional iterator for insertion-order traversal.
     *
     * Identical behavior to iterator, but yields const Value&.
     */
    struct const_iterator {
        using iterator_category = std::bidirectional_iterator_tag; /**< STL category tag. */
        using value_type = std::pair<const Key &, const Value &>; /**< Dereferenced type. */
        using reference = value_type; /**< Reference alias. */
        using difference_type = std::ptrdiff_t; /**< Signed distance. */
        using pointer = void; /**< Pointer unsupported. */

        const Node *cur; /**< Current node; nullptr for end(). */
        const DoublyLinkedCircularHashMap *map; /**< Owning map pointer. */

        /**
         * @brief Default-construct an end() iterator.
         */
        const_iterator() noexcept : cur(nullptr), map(nullptr) {
        }

        /**
         * @brief Construct a const_iterator at the given node in the given map.
         * @param node Starting node (nullptr for end()).
         * @param m    Pointer to the map instance.
         */
        const_iterator(const Node *node, const auto *m) : cur(node), map(m) {
        }

        /**
         * @brief Prefix increment: advance to the next element.
         * @return Reference to this iterator, now pointing to the next element.
         */
        const_iterator &operator++() {
            if (cur) {
                cur = cur->next_;
                if (cur == map->head_) {
                    cur = nullptr;
                }
            }
            return *this;
        }

        /**
         * @brief Postfix increment: advance to the next element, return old.
         * @return Iterator pointing to the element before increment.
         */
        const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++*this;
            return tmp;
        }

        /**
         * @brief Prefix decrement: move to the previous element.
         * @return Reference to this iterator, now pointing to the previous element.
         */
        const_iterator &operator--() {
            if (!cur) {
                cur = map->tail_;
            } else {
                cur = cur->prev_;
            }
            return *this;
        }

        /**
         * @brief Postfix decrement: move to the previous element, return old.
         * @return Iterator pointing to the element before decrement.
         */
        const_iterator operator--(int) {
            const_iterator tmp = *this;
            --*this;
            return tmp;
        }

        /**
         * @brief Compare two const_iterators for equality.
         * @param other Other iterator.
         * @return true if both point to the same element (or both end()).
         */
        bool operator==(const const_iterator &other) const {
            return cur == other.cur;
        }

        /**
         * @brief Compare two const_iterators for inequality.
         * @param other Other iterator.
         * @return true if they point to different elements.
         */
        bool operator!=(const const_iterator &other) const {
            return cur != other.cur;
        }

        /**
         * @brief Dereference to access the key/value pair.
         * @return std::pair<const Key&, const Value&> of the current element.
         */
        reference operator*() const {
            return {cur->key_, cur->value_};
        }
    };

    // Aliases for reverse iteration
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /**
     * @brief Begin iterator: points to the first element (head_).
     * @return iterator at head_, or end() if empty.
     */
    iterator begin() {
        return iterator(head_, this);
    }

    /**
     * @brief End iterator: sentinel one past the last element.
     * @return iterator(cur = nullptr).
     */
    iterator end() {
        return iterator(nullptr, this);
    }

    /**
     * @brief Const begin: points to first element in const map.
     */
    const_iterator begin() const {
        return const_iterator(head_, this);
    }

    /**
     * @brief Const end: sentinel one past the last element.
     */
    const_iterator end() const {
        return const_iterator(nullptr, this);
    }

    /**
     * @brief Const begin alias.
     */
    const_iterator cbegin() const {
        return begin();
    }

    /**
     * @brief Const end alias.
     */
    const_iterator cend() const {
        return end();
    }

    /**
     * @brief Reverse begin: wraps rbegin() to last element.
     */
    reverse_iterator rbegin() {
        return reverse_iterator(end());
    }

    /**
     * @brief Reverse end: past-the-first-element sentinel.
     */
    reverse_iterator rend() {
        return reverse_iterator(begin());
    }

    /**
     * @brief Const reverse begin.
     */
    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(end());
    }

    /**
     * @brief Const reverse end.
     */
    const_reverse_iterator rend() const {
        return const_reverse_iterator(begin());
    }

    /**
     * @brief Const reverse begin alias.
     */
    const_reverse_iterator crbegin() const {
        return rbegin();
    }

    /**
     * @brief Const reverse end alias.
     */
    const_reverse_iterator crend() const {
        return rend();
    }

    /**
     * @brief Erase element at iterator position and return next iterator.
     *
     * If it == end(), does nothing and returns end(). Otherwise, removes
     * the node from both hash and list, deletes it, and returns the iterator
     * that followed the erased element.
     *
     * @param it Iterator pointing to element to remove.
     * @return Iterator to the element after the removed one.
     */
    iterator erase(iterator it) {
        if (it == end()) {
            return it; // nothing to do
        }
        Node *node = it.cur;
        Node *nextNode = node->next_;
        Node *oldHead = head_;
        remove(node->key_);

        // If we just deleted the only element, or if nextNode wrapped back to head, we're at end():
        if (!head_ || nextNode == oldHead) {
            return end(); // no more elements
        }
        // Otherwise, return the iterator to the next element
        return iterator(nextNode, this); // wrap raw node in iterator
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- STL-like find functions -----

    /**
     * @brief STL-style lookup: returns an iterator to the element with the given key.
     *
     * Searches the appropriate bucket chain via find_node(). If found, constructs
     * an iterator pointing to that node; otherwise returns end().
     *
     * @param key Key to search for.
     * @return iterator to the element, or end() if not found.
     */
    iterator find(const Key &key) {
        if (Node *found = find_node(key)) {
            return iterator(found, this); // wrap raw node in iterator
        }
        return end(); // no such element
    }

    /**
     * @brief Const overload of STL-style find().
     *
     * @param key Key to search for.
     * @return const_iterator to the element, or end() if not found.
     */
    const_iterator find(const Key &key) const {
        if (const Node *found = find_node(key)) {
            return const_iterator(found, this);
        }
        return end();
    }

    /**
     * @brief Alias for const find(), mirroring cbegin()/cend() naming.
     *
     * @param key Key to search for.
     * @return const_iterator to the element, or end() if not found.
     */
    const_iterator cfind(const Key &key) const {
        return find(key);
    }

    /**
     * @brief Heterogeneous STL-style find() for key-like types.
     *
     * Enables lookup by types K2 (e.g., std::string_view) as long as Hash and
     * KeyEq can accept K2 arguments.
     *
     * @tparam K2 A type compatible with Hash and KeyEq.
     * @param key The key-like object to search.
     * @return iterator to the element, or end() if not found.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    iterator find(const K2 &key) {
        if (Node *found = find_node(key)) {
            return iterator(found, this);
        }
        return end();
    }

    /**
     * @brief Const heterogeneous STL-style find() for key-like types.
     *
     * @tparam K2 A type compatible with Hash and KeyEq.
     * @param key The key-like object to search.
     * @return const_iterator to the element, or end() if not found.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    const_iterator find(const K2 &key) const {
        if (const Node *found = find_node(key)) {
            return const_iterator(found, this);
        }
        return end();
    }

    /**
     * @brief Alias for const heterogeneous find(), matching cfind() naming.
     *
     * @tparam K2 A type compatible with Hash and KeyEq.
     * @param key The key-like object to search.
     * @return const_iterator to the element, or end() if not found.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    const_iterator cfind(const K2 &key) const {
        return find(key);
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- FIFO / Queue-like operations -----

    /**
     * @brief Enqueue: append a key/value pair at the back of the sequence.
     *
     * Equivalent to insert() at the end. If the key exists, its value is updated
     * and its position remains unchanged.
     *
     * @param key   Key of the element to enqueue.
     * @param value Value to associate with key.
     */
    void push_back(const Key &key, const Value &value) {
        insert(key, value); // Reuse insert for tail insertion
    }

    /**
     * @brief Access the value at the front of the queue.
     *
     * @note Calling front() on an empty map is undefined behavior.
     * @return Pointer to the value stored in head_.
     */
    Value *front() {
        return &head_->value_; // head_ points to the first node
    }

    /**
     * @brief Access the value at the back of the queue.
     *
     * @note Calling back() on an empty map is undefined behavior.
     * @return Pointer to the value stored in tail_.
     */
    Value *back() {
        return &tail_->value_; // tail_ points to the last node
    }

    /**
     * @brief Emplace an element at the front of the queue.
     *
     * Directly forwards key/value into a new Node at position 0.
     *
     * @param key   Key of the element to emplace.
     * @param value Value to associate with key.
     */
    void emplace(const Key &key, const Value &value) {
        insert_at(key, value, 0); // position 0 → new head
    }

    /**
     * @brief Dequeue: remove and return the front element’s value pointer.
     *
     * WARNING: The returned pointer refers to memory freed by remove().
     * Use or copy the value immediately before any further modifications.
     *
     * @return Pointer to the old front value (now dangling after removal).
     */
    Value *pop_front() {
        Value *pv = front(); // grab address before removal
        remove(orderedGetNode(0)->key_); // remove head element
        return pv; // pointer now invalid
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- LIFO / Stack-like operations -----

    /**
     * @brief Push element onto stack front (same as enqueue front).
     *
     * @param key   Key of the element.
     * @param value Value to associate.
     */
    void push_front(const Key &key, const Value &value) {
        insert_at(key, value, 0); // new head = stack top
    }

    /**
     * @brief Peek at the top of the stack.
     *
     * @note Undefined if map is empty.
     * @return Pointer to the value at head_ (stack top).
     */
    Value *top() {
        return front(); // head_ is the stack top
    }

    /**
     * @brief Peek at the bottom of the stack (oldest element).
     *
     * @note Undefined if map is empty.
     * @return Pointer to the value at tail_ (stack bottom).
     */
    Value *bottom() {
        return back(); // tail_ is the stack bottom
    }

    /**
     * @brief Pop the top element off the stack and return its value pointer.
     *
     * WARNING: The returned pointer refers to memory freed by remove().
     * Use or copy the value immediately before any further modifications.
     *
     * @return Pointer to the old top value (now dangling after removal).
     */
    Value *pop_back() {
        Value *pv = bottom(); // grab bottom (since top==front)
        remove(orderedGetNode(-1)->key_); // remove last element
        return pv; // pointer now invalid
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- Advanced Operations -----

    // ----- Ordered getters -----
    /**
     * @brief Retrieve the Node at a given insertion-order index.
     *
     * Because the list is circular, we first reduce idx modulo size_,
     * then decide whether it’s shorter to walk forward from head_ or
     * backward from tail_. Optionally emits debug info.
     *
     * @param idx   Arbitrary integer index (can be negative or out-of-range).
     * @param from  Optional starting Node; if non-null, we compute steps from here.
     * @param debug If true, prints diagnostic info to std::cout.
     * @return Pointer to the Node at that logical position, or nullptr if empty.
     */
    Node *orderedGetNode(const int idx, Node *from = nullptr, const bool debug = false) {
        if (size_ == 0)
            return nullptr; // nothing to return in empty map

        const int intSize = static_cast<int>(size_);
        int mod_idx = idx % intSize; // step 1: wrap index into [−size_..size_)

        if (mod_idx < 0) // step 2: normalize negative
            mod_idx += intSize;

        // step 3: pick best start point
        const bool start_at_tail = (mod_idx > intSize / 2);
        size_t steps;
        Node *cur;

        if (from) {
            // If user specified a start node, walk relative to that
            cur = from;
            steps = start_at_tail ? intSize - mod_idx : mod_idx;
        } else if (!start_at_tail) {
            // closer to head_: walk forward mod_idx steps
            cur = head_;
            steps = mod_idx;
        } else {
            // closer to tail_: walk backward (size_ − mod_idx − 1) steps
            cur = tail_;
            steps = intSize - mod_idx - 1;
        }

        if (debug) {
            const char *origin = from ? "from" : (start_at_tail ? "tail" : "head");
            std::cout
                    << "orderedGetNode(" << idx << "): reduced idx=" << mod_idx
                    << ", starting at " << origin << ", steps=" << steps << "\n";
        }

        // step 4: do the walk
        for (size_t i = 0; i < steps; ++i) {
            cur = start_at_tail
                      ? cur->prev_ // backward
                      : cur->next_; // forward
        }

        if (debug) {
            std::cout << "  landed on key=" << cur->key_
                    << ", value=" << cur->value_ << "\n";
        }
        return cur; // step 5: return result
    }

    /**
     * @brief Retrieve the value at a given insertion-order index.
     *
     * Wrapper around orderedGetNode() to return &node->value_.
     *
     * @param idx   Insertion-order index.
     * @param from  Optional start Node.
     * @param debug Debug flag.
     * @return Pointer to the mapped Value.
     */
    Value *orderedGet(const int idx, Node *from = nullptr, const bool debug = false) {
        return &orderedGetNode(idx, from, debug)->value_;
    }


    // ----- Positional swapping -----
    /**
     * @brief Swap two Nodes in the circular list by rewiring their neighbors.
     *
     * Handles three cases:
     *  • n1 immediately before n2
     *  • n2 immediately before n1 (including wraparound)
     *  • non-adjacent nodes
     *
     * After rewiring, patches head_ and tail_ if they were swapped.
     *
     * @param n1 First node.
     * @param n2 Second node.
     */
    void pos_swap_node(Node *n1, Node *n2) {
        if (n1 == n2)
            return; // nothing to do

        // Case A: n1 → n2 adjacency
        if (n1->next_ == n2) {
            Node *p = n1->prev_, *q = n2->next_;
            // relink to: p → n2 → n1 → q
            p->next_ = n2;
            n2->prev_ = p;
            n2->next_ = n1;
            n1->prev_ = n2;
            n1->next_ = q;
            q->prev_ = n1;
        }
        // Case B: n2 → n1 adjacency
        else if (n2->next_ == n1) {
            Node *p = n2->prev_, *q = n1->next_;
            // relink to: p → n1 → n2 → q
            p->next_ = n1;
            n1->prev_ = p;
            n1->next_ = n2;
            n2->prev_ = n1;
            n2->next_ = q;
            q->prev_ = n2;
        }
        // Case C: non-adjacent
        else {
            // swap incoming neighbor pointers
            std::swap(n1->prev_->next_, n2->prev_->next_);
            std::swap(n1->next_->prev_, n2->next_->prev_);
            // swap their own prev_/next_
            std::swap(n1->prev_, n2->prev_);
            std::swap(n1->next_, n2->next_);
        }

        // fix head_/tail_ if we moved them
        if (head_ == n1) head_ = n2;
        else if (head_ == n2) head_ = n1;
        if (tail_ == n1) tail_ = n2;
        else if (tail_ == n2) tail_ = n1;
    }

    /**
     * @brief Swap positions of two elements identified by keys.
     *
     * Looks up each key’s Node, throws if either is missing, then calls pos_swap_node().
     *
     * @param key1 First key.
     * @param key2 Second key.
     * @throws std::invalid_argument if either key is not present.
     */
    void pos_swap_k(const Key &key1, const Key &key2) {
        Node *n1 = find_node(key1), *n2 = find_node(key2);
        if (!n1 || !n2)
            throw std::invalid_argument("One of the keys does not exist");
        pos_swap_node(n1, n2);
    }

    /**
     * @brief Heterogeneous key-based swap.
     *
     * Allows swapping by any key-like type K2 that Hash and KeyEq accept.
     */
    template<typename K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void pos_swap_k(const K2 &key1, const K2 &key2) {
        Node *n1 = find_node(key1), *n2 = find_node(key2);
        if (!n1 || !n2)
            throw std::invalid_argument("One of the keys does not exist");
        pos_swap_node(n1, n2);
    }

    /**
     * @brief Swap positions of two elements by insertion-order indices.
     *
     * Fetches each Node via orderedGetNode(), throws if either is null, then swaps.
     *
     * @param idx1 First insertion-order index.
     * @param idx2 Second insertion-order index.
     * @throws std::invalid_argument if either index is invalid.
     */
    void pos_swap(const int idx1, const int idx2) {
        Node *n1 = orderedGetNode(idx1), *n2 = orderedGetNode(idx2);
        if (!n1 || !n2)
            throw std::invalid_argument("One of the indices does not exist");
        pos_swap_node(n1, n2);
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- Move operations -----

    /**
     * @brief Move a node to immediately before another node.
     *
     * Detaches `node` from its current spot, then inserts it right before `target`.
     * No-op if `node == target`.
     *
     * @param node   Node to relocate.
     * @param target Node before which to insert `node`.
     */
    void move_node_before(Node *node, Node *target) {
        if (node == target) {
            return; // Can't move before itself
        }
        unlink_(node); // Remove from current position
        linkBefore_(target, node); // Splice into new position
    }

    /**
     * @brief Move a node to immediately after another node.
     *
     * Detaches `node`, then inserts it right after `target`.
     * No-op if `node == target`.
     *
     * @param node   Node to relocate.
     * @param target Node after which to insert `node`.
     */
    void move_node_after(Node *node, Node *target) {
        if (node == target) {
            return; // Can't move after itself
        }
        unlink_(node); // Remove from list
        linkAfter_(target, node); // Insert after target
    }

    /**
     * @brief Move a node before the node identified by a key.
     *
     * Finds the node for `targetKey`; throws if missing.
     *
     * @param node      Node to move.
     * @param targetKey Key of the node before which to insert.
     * @throws std::invalid_argument if `targetKey` not found.
     */
    void move_node_before_n_key(Node *node, const Key &targetKey) {
        Node *target = find_node(targetKey);
        if (!target) {
            throw std::invalid_argument("Target key not found");
        }
        move_node_before(node, target);
    }

    /**
     * @brief Templated heterogeneous version of move_node_before_n_key.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void move_node_before_n_key(Node *node, const K2 &targetKey) {
        Node *target = find_node(targetKey);
        if (!target) {
            throw std::invalid_argument("Target key not found");
        }
        move_node_before(node, target);
    }

    /**
     * @brief Move a node after the node identified by a key.
     *
     * @param node      Node to move.
     * @param targetKey Key of the node after which to insert.
     * @throws std::invalid_argument if `targetKey` not found.
     */
    void move_node_after_n_key(Node *node, const Key &targetKey) {
        Node *target = find_node(targetKey);
        if (!target) {
            throw std::invalid_argument("Target key not found");
        }
        move_node_after(node, target);
    }

    /**
     * @brief Templated heterogeneous version of move_node_after_n_key.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void move_node_after_n_key(Node *node, const K2 &targetKey) {
        Node *target = find_node(targetKey);
        if (!target) {
            throw std::invalid_argument("Target key not found");
        }
        move_node_after(node, target);
    }

    /**
     * @brief Alias: move a node to the position of a target key (before).
     *
     * @param node      Node to move.
     * @param targetKey Key whose node will precede `node`.
     */
    void move_node_to_key(Node *node, const Key &targetKey) {
        move_node_before_n_key(node, targetKey);
    }

    /**
     * @brief Templated heterogeneous alias for move_node_to_key.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void move_node_to_key(Node *node, const K2 &targetKey) {
        move_node_before_n_key(node, targetKey);
    }

    /**
     * @brief Move a node before the node at the given insertion-order index.
     *
     * @param node      Node to move.
     * @param targetIdx Insertion-order index of target.
     * @throws std::invalid_argument if index invalid.
     */
    void move_node_before_n_idx(Node *node, const int targetIdx) {
        Node *target = orderedGetNode(targetIdx);
        if (!target) {
            throw std::invalid_argument("Target index not found");
        }
        move_node_before(node, target);
    }

    /**
     * @brief Move a node after the node at the given insertion-order index.
     *
     * @param node      Node to move.
     * @param targetIdx Insertion-order index of target.
     * @throws std::invalid_argument if index invalid.
     */
    void move_node_after_n_idx(Node *node, const int targetIdx) {
        Node *target = orderedGetNode(targetIdx);
        if (!target) {
            throw std::invalid_argument("Target index not found");
        }
        move_node_after(node, target);
    }

    /**
     * @brief Alias: move a node to the position at the given index.
     */
    void move_node_to_idx(Node *node, const int targetIdx) {
        move_node_before_n_idx(node, targetIdx);
    }

    /**
     * @brief Move the node for a given key before a specific node.
     *
     * @param key    Key of the node to move.
     * @param target Node before which to insert.
     * @throws std::invalid_argument if key not found.
     */
    void move_n_key_to_node(const Key &key, Node *target) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_before(node, target);
    }

    /**
     * @brief Templated heterogeneous version of move_n_key_to_node.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void move_n_key_to_node(const K2 &key, Node *target) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_before(node, target);
    }

    /**
     * @brief Move element for key1 before element for key2.
     *
     * @param key1 Source key.
     * @param key2 Target key.
     * @throws std::invalid_argument if either key missing.
     */
    void move_n_key_to_n_key(const Key &key1, const Key &key2) {
        Node *node = find_node(key1);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_to_key(node, key2);
    }

    /**
     * @brief Templated heterogeneous version of move_n_key_to_n_key.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void move_n_key_to_n_key(const K2 &key1, const K2 &key2) {
        Node *node = find_node(key1);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_to_key(node, key2);
    }

    /**
     * @brief Move element for a key to a specific index.
     *
     * @param key       Key of element to move.
     * @param targetIdx Insertion-order index to move to.
     * @throws std::invalid_argument if key missing.
     */
    void move_n_key_to_idx(const Key &key, const int targetIdx) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_to_idx(node, targetIdx);
    }

    /**
     * @brief Templated heterogeneous version of move_n_key_to_idx.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void move_n_key_to_idx(const K2 &key, const int targetIdx) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_to_idx(node, targetIdx);
    }

    /**
     * @brief Move the element at idx to before a given node.
     *
     * @param idx    Insertion-order index of the source.
     * @param target Node before which to insert.
     * @throws std::invalid_argument if idx invalid.
     */
    void move_idx_to_node(const int idx, Node *target) {
        Node *node = orderedGetNode(idx);
        if (!node) {
            throw std::invalid_argument("Index not found");
        }
        move_node_before(node, target);
    }

    /**
     * @brief Move the element at idx to before the node for a given key.
     */
    void move_idx_to_n_key(const int idx, const Key &targetKey) {
        Node *node = orderedGetNode(idx);
        if (!node) {
            throw std::invalid_argument("Index not found");
        }
        move_node_to_key(node, targetKey);
    }

    /**
     * @brief Templated heterogeneous version of move_idx_to_n_key.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void move_idx_to_n_key(const int idx, const K2 &targetKey) {
        Node *node = orderedGetNode(idx);
        if (!node) {
            throw std::invalid_argument("Index not found");
        }
        move_node_to_key(node, targetKey);
    }

    /**
     * @brief Move the element at idx to before the element at targetIdx.
     *
     * @param idx       Source insertion-order index.
     * @param targetIdx Destination insertion-order index.
     * @throws std::invalid_argument if either index invalid.
     */
    void move_idx_to_idx(const int idx, const int targetIdx) {
        Node *node = orderedGetNode(idx);
        if (!node) {
            throw std::invalid_argument("Index not found");
        }
        move_node_to_idx(node, targetIdx);
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- Node shifting functions -----

    /**
     * @brief Shift a specific node by a given offset in the circular list.
     *
     * Moves `node` forward (positive shift) or backward (negative shift) by
     * `shift` steps, preserving insertion order relative to other elements.
     * Internally uses orderedGetNode to compute the destination node in O(k)
     * time (where k = min(walk distances)), then splices via move_node_after/
     * move_node_before in O(1).
     *
     * @param node  Pointer to the Node to shift; must be part of this map.
     * @param shift Number of positions to move:
     *              • >0 → forward,
     *              • <0 → backward,
     *              • 0 → no-op.
     * @throws std::invalid_argument if orderedGetNode cannot locate the dest.
     */
    void shift_node(Node *node, const int shift) {
        // Locate the node that should precede/follow `node` after shifting
        Node *found = orderedGetNode(shift, node);
        if (!found) {
            throw std::invalid_argument("Shift index not found");
        }

        // Fast-exit if shift is zero
        if (shift == 0) {
            return;
        }

        // Splice node into its new position in O(1)
        if (shift > 0) {
            move_node_after(node, found); // place after the found node
        } else {
            move_node_before(node, found); // place before the found node
        }
    }

    /**
     * @brief Shift the element associated with `key` by `shift` positions.
     *
     * Finds the Node for `key` and delegates to shift_node().
     *
     * @param key   Key of the element to shift.
     * @param shift Offset as in shift_node().
     * @throws std::invalid_argument if key is not present.
     */
    void shift_n_key(const Key &key, const int shift) {
        Node *node = find_node(key); // locate node by key
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        shift_node(node, shift);
    }

    /**
     * @brief Heterogeneous-version of shift_n_key for key-like types.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
                 && std::invocable<KeyEq, const Key &, const K2 &>
    void shift_n_key(const K2 &key, const int shift) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        shift_node(node, shift);
    }

    /**
     * @brief Shift the element at insertion-order index `idx` by `shift`.
     *
     * This optimized algorithm avoids traversing all N elements by:
     *  1. Reducing source and destination indices modulo N.
     *  2. Computing distances from head_ and tail_ for both indices.
     *  3. Choosing the shorter of (src-walk + dst-walk) via a single or two
     *     partial walks.
     *  4. Performing at most two walks of length ≤ N/2 each.
     *  5. Splicing the node in O(1).
     *
     * @param idx   Insertion-order index (can be negative or out-of-range).
     * @param shift Offset in insertion-order positions.
     * @throws std::out_of_range if the container is empty.
     */
    void shift_idx(const int idx, const int shift)
    __attribute__((always_inline, hot, optimize("O3"))) {
        // 0) Fast path for zero shift
        if (LIKELY(shift == 0)) return;

        // 0.5) Empty container check
        if (UNLIKELY(size_ == 0))
            throw std::out_of_range("Index out of range");

        const int N = static_cast<int>(size_);
        const int tail_i = N - 1;

        // 1) Compute modularized source/destination indices in [0..N-1]
        int src_mod = idx % N;
        int dst_mod = (idx + shift) % N;
        if (src_mod < 0) src_mod += N; // normalize negative
        if (dst_mod < 0) dst_mod += N;

        // 2) If normalized indices coincide, no move needed
        if (LIKELY(src_mod == dst_mod)) return;

        // 3) Distances from head_ and tail_ to src and dst
        const int src_from_head = src_mod;
        const int dst_from_head = dst_mod;
        const int src_from_tail = tail_i - src_mod;
        const int dst_from_tail = tail_i - dst_mod;

        // 4) Choose whether to reference from source or destination first
        const int src_walk = std::min(src_from_head, src_from_tail);
        const int dst_walk = std::min(dst_from_head, dst_from_tail);
        const bool from_src = (src_walk <= dst_walk);

        // 5) Decide direction and starting reference for first walk
        const bool first_dir = from_src
                                   ? (src_from_head <= src_from_tail)
                                   : (dst_from_head <= dst_from_tail);
        Node *first_ref = first_dir ? head_ : tail_;
        const int first_walk = from_src ? src_walk : dst_walk;

        // 6) Compute remaining walk for the other index
        const int second_walk = from_src ? dst_walk : src_walk;

        // 7) Perform first walk
        Node *cur = walk(first_ref, first_walk, first_dir);
        Node *src_node = from_src ? cur : nullptr;
        Node *dst_node = from_src ? nullptr : cur;

        // 8) Determine optimal path for second walk
        if (const int raw_dist = std::abs(src_from_head - dst_from_head); raw_dist <= second_walk) {
            // walking directly from current position is cheaper
            cur = walk(cur, raw_dist, first_dir);
        } else {
            // rebase from the closer end and walk
            const bool second_dir = from_src
                                        ? (dst_from_head <= dst_from_tail)
                                        : (src_from_head <= src_from_tail);
            Node *second_ref = second_dir ? head_ : tail_;
            cur = walk(second_ref, second_walk, second_dir);
        }
        if (from_src) dst_node = cur;
        else src_node = cur;

        // 9) Perform the final splice in O(1)
        if (shift > 0) {
            move_node_after(src_node, dst_node);
        } else {
            move_node_before(src_node, dst_node);
        }
    }

    // ---------- 1. Safe zig-zag helpers ---------- //

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
    static std::ptrdiff_t computeZigzagOffset(const std::size_t idx,
                                              const std::size_t offset) noexcept {
        using S = std::make_signed_t<std::size_t>; // signed type wide enough for size_t
        const S i = static_cast<S>(idx);
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
    static std::pair<std::ptrdiff_t, std::ptrdiff_t>
    computeZigzagOffsetPair(const std::size_t nth = 0,
                            const std::size_t l_cnt = 0,
                            const std::size_t r_cnt = 0) noexcept {
        return {
            computeZigzagOffset(2 * nth, l_cnt),
            computeZigzagOffset(2 * nth + 1, r_cnt)
        };
    }


    // ---------- 2. Bulk find operations ---------- //

    /**
     * @brief Find multiple nodes by insertion-order indices using a greedy zig-zag walk.
     *
     * This template function:
     *  1. Normalizes all raw indices modulo `size_`, handling negatives.
     *  2. Sorts (if `!pre_sorted`) and optionally deduplicates (`!allow_dupes`).
     *  3. Performs a greedy zig-zag traversal from both ends of the list, at each
     *     step choosing the closer next request to minimize total node hops.
     *  4. Populates an output container of `Node*`, preserving input order or including
     *     duplicates as requested.
     *  5. Runs traversing at max (M/(M+1))(N-1) nodes, where M is the number of
     *     unique requests and N is the total number of nodes. This function is most beneficial
     *     when M << N, as it avoids traversing the entire list for each request.
     *
     * @tparam Index      Integer type of input indices (signed OK).
     * @tparam Container  Container template (e.g. std::vector) for output.
     * @tparam AllocIndex Allocator type for input container C.
     * @tparam Rest       Additional template parameters of Container.
     * @tparam C          Type of input container (must satisfy IntRange).
     * @param in          Input container of raw indices.
     * @param pre_sorted  If true, skip sorting step.
     * @param verbose     If true, emit step-by-step debug prints.
     * @param allow_dupes If true, retain duplicate indices; else skip duplicates.
     * @param profiling_info If true, emit profiling info.
     * @return Container of Node* corresponding to each requested index.
     */
    template<typename Index,
        template<typename, typename...> class Container,
        typename AllocIndex,
        typename... Rest,
        IntRange C = Container<Index, AllocIndex, Rest...> >
    auto find_n_nodes(const C &in,
                      const bool pre_sorted = false,
                      const bool verbose = false,
                      const bool allow_dupes = false,
                      const bool profiling_info = false) {
        using S = std::make_signed_t<Index>;
        using AllocNodePtr = typename std::allocator_traits<AllocIndex>
                ::template rebind_alloc<Node *>;
        using OutContainer = Container<Node *, AllocNodePtr, Rest...>;
        using Vec = std::vector<Node *, AllocNodePtr>;

        const std::size_t N = size_; // total nodes in list
        const S tail_i = static_cast<S>(N) - 1; // last index

        // Early exit for empty map
        if (UNLIKELY(N == 0)) {
            return OutContainer{};
        }

        // 2a. Normalize & collect indices modulo N
        std::vector<S> mod_idx;
        mod_idx.reserve(in.size());
        for (auto raw: in) {
            S m = static_cast<S>(raw) % static_cast<S>(N);
            if (m < 0) m += static_cast<S>(N);
            mod_idx.push_back(m);
        }
        if (verbose) {
            std::cout << "find_n_nodes: normalized = { ";
            for (auto m: mod_idx) std::cout << m << ' ';
            std::cout << "}\n";
        }

        // 2b. Sort & dedupe
        if (!pre_sorted) {
            std::ranges::sort(mod_idx);
        }
#ifndef NDEBUG
        if (!std::is_sorted(mod_idx.begin(), mod_idx.end()))
            throw std::logic_error("find_n_nodes: indices not sorted!");
#endif

        // Copy for duplicate handling
        std::vector<S> copy_mod_idx = mod_idx;
        std::vector<std::vector<S> > pos_mod_idx(in.size(), std::vector<S>(1));
        S prev = static_cast<S>(-1);
        size_t prev_idx = static_cast<size_t>(-1), dupe_adj = 0;

        for (size_t i = 0; i < copy_mod_idx.size(); i++) {
            S cur = copy_mod_idx[i];
            if (cur == prev) {
                if (allow_dupes) {
                    pos_mod_idx[prev_idx].push_back(i);
                } else {
                    ++dupe_adj;
                    if (verbose) {
                        std::cout << "find_n_nodes: skipping duplicate " << cur << "\n";
                    }
                }
                mod_idx.erase(mod_idx.begin() + i);
            } else {
                ++prev_idx;
                pos_mod_idx[prev_idx].clear();
                S fill_idx = i - (allow_dupes ? 0 : dupe_adj);
                pos_mod_idx[prev_idx].push_back(fill_idx);
            }
            prev = cur;
        }
        pos_mod_idx.resize(prev_idx + 1);

        if (verbose) {
            std::cout << "find_n_nodes: unique mods = { ";
            for (auto m: mod_idx) std::cout << m << ' ';
            std::cout << "}\n";
        }

        const std::size_t M = mod_idx.size();
        if (M == 0) {
            return OutContainer{}; // no requests remain
        }

        // 2c. Greedy zig-zag walk
        Vec out(allow_dupes ? copy_mod_idx.size() : M);
        std::size_t l_cnt = 0, r_cnt = 0;
        S bnd_low = 0, bnd_high = tail_i;
        size_t left = 0, right = M - 1;
        Node *left_ref = head_, *right_ref = tail_;

        size_t total_walk = 0;
        while (left <= right) {
            // 2d. Compute next left/right pick offsets
            auto [l_off, r_off] = computeZigzagOffsetPair(0, l_cnt, r_cnt);
            S l_mod_idx = mod_idx[0 + l_off];
            S r_mod_idx = mod_idx[M + r_off];

            // 2e. Distances from current bounds
            const long l_dist = static_cast<long>(l_mod_idx) - bnd_low;
            const long r_dist = bnd_high - static_cast<long>(r_mod_idx);
            bool pick_left = (l_dist <= r_dist);
            total_walk += pick_left ? l_dist : r_dist;

            // 2f. Walk to the chosen node
            Node *cur = pick_left
                            ? walk(left_ref, static_cast<size_t>(l_dist), true)
                            : walk(right_ref, static_cast<size_t>(r_dist), false);

            // 2g. Update bounds & refs
            if (pick_left) {
                bnd_low = l_mod_idx;
                left_ref = cur;
                ++l_cnt;
                ++left;
            } else {
                bnd_high = r_mod_idx;
                right_ref = cur;
                ++r_cnt;
                --right;
            }

            // 2h. Fill output slots for this pick
            for (auto pos: pos_mod_idx[pick_left ? left - 1 : right + 1]) {
                out[pos] = cur;
            }
        }

        if (profiling_info) {
            std::cout << "find_n_nodes: profiling info: \n";
            std::cout << "Expected walk bound ((M/(M+1))(N-1)) = "
                    << (static_cast<double>(M) / (M + 1)) * (N - 1) << "\n";
            std::cout << "Actual walk bound = " << total_walk << "\n";
        }

        // 3. Convert to requested container type
        if constexpr (std::is_same_v<OutContainer, Vec>) {
            return out;
        } else {
            return OutContainer(out.begin(), out.end());
        }
    }

    // -------------- helper in the same namespace -------------- //

    /**
     * @brief Rebind a container template to a new value type.
     *
     * Example:
     *   rebind_to<std::vector, std::list<int>>::apply<float>
     *   yields std::vector<float>.
     */
    template<template<typename, typename...> class Dest,
        typename Concrete>
    struct rebind_to; // primary left undefined

    template<template<typename, typename...> class Dest,
        template<typename, typename...> class Src,
        typename U, typename... As>
    struct rebind_to<Dest, Src<U, As...> > {
        template<typename X, typename... Ys>
        using apply = Dest<X, Ys...>;
    };

    // ───────────────────────── helper: get_template ───────────────────────── //

    /**
     * @brief Extract the template-template parameter from a concrete container.
     *
     * For Concrete = std::vector<int,Alloc>, get_template<Concrete>::apply<T,Alloc2...>
     * produces std::vector<T,Alloc2...>.
     */
    template<typename Concrete>
    struct get_template; // primary undefined

    template<template<typename, typename...> class TT,
        typename U, typename... Args>
    struct get_template<TT<U, Args...> > {
        template<typename X, typename... Ys>
        using apply = TT<X, Ys...>;
    };

    // 2. wrapper that forwards a real template

    /**
     * @brief Convenience overload for find_n_nodes on arbitrary ranges.
     *
     * Deduces Index and allocator from `C` and forwards to the main
     * template implementation above.
     */
    template<IntRange C>
    [[nodiscard]]
    auto find_n_nodes(const C &in,
                      bool pre_sorted = false,
                      bool verbose = false,
                      bool allow_dupes = false,
                      bool profiling_info = false) {
        using Index = std::ranges::range_value_t<C>;
        using AllocIndex = typename C::allocator_type;

        return find_n_nodes<
            Index,
            get_template<std::remove_cvref_t<C> >::template apply,
            AllocIndex
        >(in, pre_sorted, verbose, allow_dupes, profiling_info);
    }


    // ---------- 5. Rotating and Reversal ---------- //

    /**
     * @brief Rotate the circular list by k steps (head moves forward).
     *
     * - If size_ ≤ 1 or k % size_ == 0: no-op.
     * - Normalizes k into [0..size_-1], walks k steps in O(min(k, N-k)),
     *   then updates tail_ to head_->prev_.
     *
     * @param k Rotation offset; positive → forward, negative handled by modulo.
     */
    void rotate(int k) noexcept {
        if (size_ <= 1 || (k %= static_cast<int>(size_)) == 0)
            return;

        if (k < 0) k += static_cast<int>(size_);
        head_ = walk(head_, k, /*forward=*/true);
        tail_ = head_->prev_;
    }

    /**
     * @brief Reverse the entire circular list in place.
     *
     * - Swaps head_ and tail_.
     * - Iterates once through the list, swapping each node’s next_ and prev_ pointers.
     * - O(N) time, O(1) extra space.
     */
    void reverse() noexcept {
        if (size_ <= 1) return;

        std::swap(head_, tail_);
        Node *cur = head_;
        do {
            std::swap(cur->next_, cur->prev_);
            cur = cur->prev_; // prev_ is original next_
        } while (cur != head_);
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- Splicing and Splitting -----

    /**
     * @brief Splice a range of elements from another map into this map.
     *
     * Moves the sequence [first, last) from `other` into *this*, inserting it
     * immediately before `pos`. Maintains both circular list and hash bucket
     * invariants.
     *
     * @param pos    Iterator to the insertion point in this map.
     * @param other  Source map from which to splice elements.
     * @param first  Iterator to the first element to splice in `other`.
     * @param last   Iterator one past the last element to splice.
     * @return Iterator pointing to the first spliced element in this map.
     */
    iterator splice(iterator pos,
                    DoublyLinkedCircularHashMap &other,
                    iterator first,
                    iterator last) {
        // No elements or empty range → nothing to do
        if (other.empty() || first == last) {
            return pos;
        }

        // Identify start and end of subchain to move
        Node *f = first.cur; // first node in range
        Node *l = last.cur; // node after last to move
        Node *lPrev = l->prev_; // last node in range

        // Count nodes to move
        size_t count = 0;
        for (Node *cur = f; cur != l; cur = cur->next_) {
            ++count;
        }

        // Detach subchain [f ... lPrev] from other map
        Node *fPrev = f->prev_;
        fPrev->next_ = l; // bypass subchain
        l->prev_ = fPrev;

        // Update other map's head, tail, and size
        other.size_ -= count;
        if (other.size_ == 0) {
            other.head_ = other.tail_ = nullptr;
        } else {
            if (other.head_ == f) other.head_ = l;
            if (other.tail_ == lPrev) other.tail_ = fPrev;
        }

        // Splice into this map at pos
        if (empty()) {
            // This map was empty → subchain becomes entire list
            head_ = f;
            tail_ = lPrev;
            head_->prev_ = tail_;
            tail_->next_ = head_;
        } else {
            Node *posNode = pos.cur; // insertion point
            Node *posPrev = posNode
                                ? posNode->prev_ // node before insertion
                                : tail_;

            // Link posPrev → f
            posPrev->next_ = f;
            f->prev_ = posPrev;

            // Link lPrev → posNode (or wrap to head_)
            if (posNode) {
                lPrev->next_ = posNode;
                posNode->prev_ = lPrev;
            } else {
                // pos == end() → wrap to head_
                lPrev->next_ = head_;
                head_->prev_ = lPrev;
            }

            // Update head_ if inserting at front
            if (posNode == head_) {
                head_ = f;
            }
            // Tail_ is always head_->prev_
            tail_ = head_->prev_;
        }

        // Rehome hash bucket pointers for moved nodes
        Node *cur = f;
        for (size_t i = 0; i < count; ++i) {
            Node *nxt = cur->next_;
            other.bucketRemove_(cur);
            bucketInsert_(cur);
            cur = nxt;
        }

        size_ += count; // update size of this map

        return iterator(f, this);
    }

    /**
     * @brief Split the map into two at a given index.
     *
     * Moves elements starting at `idx` into a new map, preserving relative order.
     *
     * @param idx  Insertion-order index at which to split. Negative and out-of-range
     *             values are normalized modulo size().
     * @return A new DoublyLinkedCircularHashMap containing the tail segment.
     */
    DoublyLinkedCircularHashMap split(const int idx) {
        // Prepare result map with same configuration
        DoublyLinkedCircularHashMap tailMap(
            htBaseVector_.size(), maxLoadFactor_,
            hashFunc_, keyEqFunc_);

        const int N = static_cast<int>(size_);
        if (N == 0) return tailMap; // empty → nothing to move

        // Normalize split index
        int k = idx % N;
        if (k < 0) k += N;

        // Trivial: split at 0 → all elements go to new map
        if (k == 0) {
            tailMap = std::move(*this);
            clear();
            return tailMap;
        }
        // Beyond end → nothing to split
        if (k >= N) {
            return tailMap;
        }

        // Locate cut point
        Node *cut = orderedGetNode(k);
        Node *cutPrev = cut->prev_;

        // Sever circular links in this map
        cutPrev->next_ = head_;
        head_->prev_ = cutPrev;

        // Initialize tailMap's list
        tailMap.head_ = cut;
        tailMap.tail_ = tail_;
        tailMap.head_->prev_ = tailMap.tail_;
        tailMap.tail_->next_ = tailMap.head_;

        // Update this map's tail
        tail_ = cutPrev;

        // Compute sizes
        const size_t oldSize = size_;
        size_ = static_cast<size_t>(k);
        tailMap.size_ = oldSize - size_;

        // Rehome bucket pointers for moved nodes
        Node *curNode = cut;
        for (size_t i = 0; i < tailMap.size_; ++i) {
            Node *nxt = curNode->next_;
            bucketRemove_(curNode);
            tailMap.bucketInsert_(curNode);
            curNode = nxt;
        }

        return tailMap;
    }

    /**
     * @brief Erase elements satisfying a predicate.
     *
     * Iterates through the map in insertion order, removing each element
     * for which `pred(key, value)` returns true.
     *
     * @tparam Pred A callable predicate taking `(const Key&, Value&)` and
     *              returning bool.
     * @param pred    Predicate to test each element.
     * @param verbose If true, prints diagnostic messages to std::cout.
     * @return Number of elements erased.
     */
    template<class Pred>
        requires std::predicate<Pred, const Key &, Value &>
    size_t erase_if(Pred &&pred, const bool verbose = false) {
        size_t erased = 0;
        for (auto it = begin(); it != end(); /* no increment */) {
            const Key &k = (*it).first;
            Value &v = (*it).second;
            if (verbose) {
                std::cout << "[erase_if] visiting key=" << k << "\n";
            }
            if (pred(k, v)) {
                if (verbose) {
                    std::cout << "[erase_if] erasing key=" << k << "\n";
                }
                it = erase(it);
                ++erased;
            } else {
                ++it;
            }
        }
        if (verbose) {
            std::cout << "[erase_if] done, erased=" << erased << "\n";
        }
        return erased;
    }


    //───────────────────────────────────────────────────────────────────────────//
    // ----- Debug, Validation, and Utility -----

    /**
     * @brief Compute the size of each hash bucket.
     *
     * Traverses every bucket chain and counts its nodes.
     *
     * @return A vector of bucket chain lengths, indexed by bucket.
     */
    [[nodiscard]] std::vector<size_t> bucketSizesCalc() const {
        std::vector<size_t> sizes(htBaseVector_.size(), 0);
        // Count nodes in each bucket chain
        for (size_t b = 0; b < htBaseVector_.size(); ++b) {
            for (Node *n = htBaseVector_[b]; n; n = n->hashNext_) {
                ++sizes[b];
            }
        }
        return sizes;
    }

    /**
     * @brief Print a histogram of bucket loads to the given stream.
     *
     * Uses bucketSizesCalc() to fetch chain lengths.
     *
     * @param os  Output stream (defaults to std::cout).
     */
    void printBucketDistribution(std::ostream &os = std::cout) const {
        const auto sizes = bucketSizesCalc();
        os << "Bucket distribution (" << sizes.size() << " buckets):\n";
        // Print each bucket’s count
        for (size_t i = 0; i < sizes.size(); ++i) {
            os << "  [" << i << "] = " << sizes[i] << "\n";
        }
    }

    /**
     * @brief Print a histogram of bucket loads using stored bucketSizes().
     *
     * Avoids recalculating by using the cached bucketSizes_ array.
     *
     * @param os  Output stream (defaults to std::cout).
     */
    void printBucketDistribution2(std::ostream &os = std::cout) const {
        os << "Bucket distribution (" << htBaseVector_.size() << " buckets):\n";
        // Print each bucket’s cached count
        for (size_t i = 0; i < htBaseVector_.size(); ++i) {
            os << "  [" << i << "] = " << bucketSizes()[i] << "\n";
        }
    }

    /**
     * @brief Debug helper: print a key’s raw hash and bucket index.
     *
     * @param k   Key to hash.
     * @param os  Output stream (defaults to std::cout).
     */
    void debugKey(const Key &k, std::ostream &os = std::cout) const {
        size_t h = hashFunc_(k);
        size_t b = h % htBaseVector_.size();
        os << "key=" << k
                << "  hash=" << h
                << "  bucket=" << b << "\n";
    }

    /**
     * @brief Validate internal consistency of the map.
     *
     * Checks circular list integrity, bucket-chain coverage, and head/tail pointers.
     * Throws std::runtime_error on any inconsistency.
     */
    void validate() const {
        // 1) Empty map must have null head/tail and no buckets
        if (size_ == 0) {
            if (head_ || tail_)
                throw std::runtime_error("Empty map must have null head/tail");
            for (auto b: htBaseVector_) {
                if (b) throw std::runtime_error("Empty map must have no bucket entries");
            }
            return;
        }

        // 2) Walk the circular list once, checking links and collecting nodes
        std::unordered_set<const Node *> seen;
        seen.reserve(size_);
        const Node *cur = head_;
        for (size_t i = 0; i < size_; ++i) {
            if (!cur) {
                throw std::runtime_error("List terminated early");
            }
            // Check next/prev consistency
            if (cur->next_->prev_ != cur || cur->prev_->next_ != cur) {
                throw std::runtime_error("List is not properly doubly‐linked");
            }
            // Ensure each node is unique
            if (!seen.insert(cur).second) {
                throw std::runtime_error("Node appears twice in list");
            }
            cur = cur->next_;
        }
        // Must wrap back to head
        if (cur != head_) {
            throw std::runtime_error("List does not wrap back to head");
        }
        // Check head/tail circular pointers
        if (head_->prev_ != tail_ || tail_->next_ != head_) {
            throw std::runtime_error("Head/Tail pointers not circularly consistent");
        }

        // 3) Walk each bucket chain, ensuring every list node appears exactly once
        size_t bucketCounted = 0;
        for (auto bucketHead: htBaseVector_) {
            for (const Node *n = bucketHead; n; n = n->hashNext_) {
                ++bucketCounted;
                if (!seen.count(n)) {
                    throw std::runtime_error("Bucket node not in list");
                }
                seen.erase(n);
            }
        }
        // All nodes should be accounted for
        if (bucketCounted != size_) {
            throw std::runtime_error("Bucket node‐count != size_");
        }
        if (!seen.empty()) {
            throw std::runtime_error("Some list nodes never appeared in any bucket");
        }
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Key and Value Views -----

    /**
     * @brief View of all keys in insertion order.
     *
     * @return A range of const Key&.
     */
    auto keys_view() const {
        return std::views::all(*this)
               | std::views::transform([](auto const &kv) -> const Key & { return kv.first; });
    }

    /**
     * @brief View of all values in insertion order.
     *
     * @return A range of const Value&.
     */
    auto values_view() const {
        return std::views::all(*this)
               | std::views::transform([](auto const &kv) -> const Value & { return kv.second; });
    }

    /** @brief Shorthand for keys_view(). */
    auto keys() const {
        return keys_view();
    }

    /** @brief Shorthand for values_view(). */
    auto values() const {
        return values_view();
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Swap Functions -----

    /**
     * @brief Swap contents with another map.
     *
     * Exchanges all internals: buckets, list pointers, size, and functors.
     *
     * @param other  Map to swap with.
     */
    void swap(DoublyLinkedCircularHashMap &other) noexcept {
        std::swap(htBaseVector_, other.htBaseVector_);
        std::swap(head_, other.head_);
        std::swap(tail_, other.tail_);
        std::swap(size_, other.size_);
        std::swap(maxLoadFactor_, other.maxLoadFactor_);
        std::swap(hashFunc_, other.hashFunc_);
        std::swap(keyEqFunc_, other.keyEqFunc_);
    }

    /**
     * @brief Friend swap for ADL.
     *
     * @param a  First map.
     * @param b  Second map.
     */
    friend void swap(DoublyLinkedCircularHashMap &a,
                     DoublyLinkedCircularHashMap &b) noexcept {
        a.swap(b);
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Deprecated functions -----

    // original version of shift_idx
    /**
     * @brief Original, highly‐optimized but hard‐to‐read version of shift_idx.
     *
     * Computes source and destination positions with minimal hops by:
     *  1. Normalizing indices modulo size.
     *  2. Calculating distances from head and tail.
     *  3. Walking the shorter path to each target.
     *  4. Splicing the node in place.
     *
     * @param idx    Insertion‐order index of the element to move.
     * @param shift  Number of positions to move (positive → forward; negative → backward).
     * @throws std::out_of_range if the map is empty or index out of range.
     */
    [[deprecated]] void shift_idx_og(const int idx, const int shift) {
        // make more efficient bny avoiding double lookup and minimizing distance:
        // make sure we actually need to do something
        if (shift == 0) {
            return;
        }
        // make sure list actually exists
        if (size_ == 0) {
            throw std::out_of_range("Index out of range");
        }
        const int int_size = static_cast<int>(size_);
        const int result_idx = idx + shift;
        const int tail_idx = size_ - 1;
        int mod_idx = idx % int_size;
        int mod_r_idx = result_idx % int_size;
        // handle negative indices
        if (mod_idx < 0) {
            mod_idx += int_size;
        }
        if (mod_r_idx < 0) {
            mod_r_idx += int_size;
        }
        // Again check to make sure we actually need to do something, because if we don't we can just return
        if (mod_idx == mod_r_idx) {
            return;
        }
        // check which of mod_idx and mod_r_idx is closest to a reference node. Ie head_ or tail_
        // we will first fetch the closest node to a reference node
        const int &idx_frm_head_idx = mod_idx;
        const int &r_idx_frm_head_idx = mod_r_idx;
        const int idx_frm_tail_idx = tail_idx - mod_idx;
        const int r_idx_frm_tail_idx = tail_idx - mod_r_idx;
        const int idx_shortest_idx = std::min(idx_frm_head_idx, idx_frm_tail_idx);
        const int r_idx_shortest_idx = std::min(r_idx_frm_head_idx, r_idx_frm_tail_idx);
        int closest_idx;
        int closest_idx_raw;
        int furthest_idx;
        int furthest_idx_raw;
        bool from_idx;
        bool from_head;
        bool far_from_head;
        Node *idx_node;
        Node *r_idx_node;
        if (idx_shortest_idx <= r_idx_shortest_idx) {
            closest_idx = idx_shortest_idx;
            closest_idx_raw = idx_frm_head_idx;
            furthest_idx = r_idx_shortest_idx;
            furthest_idx_raw = r_idx_frm_head_idx;
            from_idx = true;
            from_head = idx_frm_head_idx <= idx_frm_tail_idx;
            far_from_head = r_idx_frm_head_idx <= r_idx_frm_tail_idx;
        } else {
            closest_idx = r_idx_shortest_idx;
            closest_idx_raw = r_idx_frm_head_idx;
            furthest_idx = idx_shortest_idx;
            furthest_idx_raw = idx_frm_head_idx;
            from_idx = false;
            from_head = r_idx_frm_head_idx <= r_idx_frm_tail_idx;
            far_from_head = idx_frm_head_idx <= idx_frm_tail_idx;
        }
        Node *first_ref = from_head ? head_ : tail_;
        // iterate through the list to find the closest node
        Node *cur = first_ref;
        for (int i = 0; i < closest_idx; ++i) {
            cur = from_head ? cur->next_ : cur->prev_;
        }
        if (from_idx) {
            idx_node = cur;
        } else {
            r_idx_node = cur;
        }
        // now we will get the shortest distance of this node to the furthest node
        // now we must compare it against the distances of the furthest node to the head or tail
        if (const int raw_dist = std::abs(closest_idx_raw - furthest_idx_raw); raw_dist <= furthest_idx) {
            for (int i = 0; i < raw_dist; ++i) {
                cur = from_head ? cur->next_ : cur->prev_;
            }
            if (from_idx) {
                r_idx_node = cur;
            } else {
                idx_node = cur;
            }
        } else {
            Node *second_ref = far_from_head ? head_ : tail_;
            // iterate through the list to find the furthest node
            cur = second_ref;
            for (int i = 0; i < furthest_idx; ++i) {
                cur = far_from_head ? cur->next_ : cur->prev_;
            }
            if (from_idx) {
                r_idx_node = cur;
            } else {
                idx_node = cur;
            }
        }
        // now we have the two nodes we need for the shift we will shift idx_node to r_idx_node
        if (shift > 0) {
            move_node_after(idx_node, r_idx_node);
        } else {
            move_node_before(idx_node, r_idx_node);
        }
    }
};


#endif //DOUBLYLINKEDCIRCULARHASHMAP_HPP

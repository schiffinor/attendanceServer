//
// Created by schif on 4/21/2025.
//

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
            std::integral<std::ranges::range_value_t<R>>;

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
    typename KeyEq = std::equal_to<Key> >
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
    struct alignas(64) Node {
        Key key_;               /**< The key for this node. Immutable after construction. */
        Value value_;           /**< The value mapped to key_. Move-constructed for efficiency. */
        //----- Circular doubly-linked list pointers (maintain insertion order) -----
        Node *next_;            /**< Next node in insertion order; wraps around to head_ if at tail_. */
        Node *prev_;            /**< Previous node in insertion order; wraps to tail_ if at head_. */
        //----- Hash bucket chaining pointers (for collision resolution) -----
        Node *hashNext_;        /**< Next node in the same hash bucket chain. nullptr if last in bucket. */
        Node *hashPrev_;        /**< Previous node in the same hash bucket chain. nullptr if first in bucket. */
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
          next_(this),           // Self-referential: node is its own next in empty list
          prev_(this),           // Self-referential: node is its own prev in empty list
          hashNext_(nullptr),    // Not in a bucket yet
          hashPrev_(nullptr) {
                // No further initialization needed—ready to be linked in
        }
    };

    //────────────────────────────────────────────────────────────────────────//
    //----- Internal data members for DoublyLinkedCircularHashMap -----

    std::vector<Node *> htBaseVector_;    /**< Heads of each hash bucket chain. Size = number of buckets. */
    std::vector<size_t> bucketSizes_;     /**< Number of nodes currently in each bucket, for diagnostics. */

    Node *head_ = nullptr;                /**< First node in insertion order; nullptr if map is empty. */
    Node *tail_ = nullptr;                /**< Last node in insertion order; nullptr if map is empty. */
    size_t size_ = 0;                     /**< Total number of elements currently in the map. */

    double maxLoadFactor_ = 1.0;          /**< Threshold (size_/bucket_count_) to trigger rehashing. */

    std::function<size_t(const Key &)> hashFunc_; /**< User-specified or default hash functor (std::hash). */
    std::function<bool(const Key &, const Key &)> keyEqFunc_; /**< Equality comparator for Key. */

    size_t rehashCount_ = 0;              /**< Number of times the table has been rehashed. Useful for profiling. */
    size_t maxBucketSize_ = 0;            /**< Largest bucket size observed since last rehash. */
    size_t largestBucketIdx_ = SIZE_MAX;  /**< Index of the bucket that currently has maxBucketSize_. */

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
     * @brief Inserts a node into its proper bucket using separate chaining.
     * @param node Node to insert; we’ll splice it into the head of its chain.
     *
     * New nodes go to the front for O(1) insertion. Updates bucketSizes_, and
     * if this bucket grows beyond the old max, updates maxBucketSize_ and
     * largestBucketIdx_.
     */
    void bucketInsert_(Node *node) {
        auto idx   = bucketIndex_(node->key_);
        Node *head = htBaseVector_[idx];
        if (!head) {
            // Empty bucket → node stands alone.
            htBaseVector_[idx] = node;
            node->hashNext_    = node->hashPrev_ = nullptr;
        } else {
            // Prepend node to existing chain.
            node->hashNext_       = head;
            head->hashPrev_       = node;
            node->hashPrev_       = nullptr;
            htBaseVector_[idx]    = node;
        }
        ++bucketSizes_[idx]; // Keep the count honest.
        // Track maximum chain length.
        if (bucketSizes_[idx] > maxBucketSize_) {
            maxBucketSize_   = bucketSizes_[idx];
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
            maxBucketSize_   = 0;
            largestBucketIdx_ = SIZE_MAX;
            for (size_t i = 0; i < htBaseVector_.size(); ++i) {
                if (bucketSizes_[i] > maxBucketSize_) {
                    maxBucketSize_    = bucketSizes_[i];
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
        node->prev_         = where;
        node->next_         = where->next_;
        where->next_->prev_ = node;
        where->next_        = node;
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
        node->next_         = where;
        node->prev_         = where->prev_;
        where->prev_->next_ = node;
        where->prev_        = node;
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
        maxBucketSize_   = 0;
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
            for (auto &n : cur) {
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
     */
    explicit DoublyLinkedCircularHashMap(
        size_t initBuckets = 16,
        const double maxLoadFactor = 1.0,
        std::function<size_t(const Key &)> hashFunc = std::hash<Key>(),
        std::function<bool(const Key &, const Key &)> keyEqFunc = std::equal_to<Key>()
    )
        : htBaseVector_(initBuckets, nullptr),  // all buckets start empty
          bucketSizes_(initBuckets, 0),         // track 0 elements per bucket
          head_(nullptr),                       // empty list → no head
          tail_(nullptr),                       // empty list → no tail
          maxLoadFactor_(maxLoadFactor),        // store user’s max load factor
          hashFunc_(std::move(hashFunc)),       // move-in hash functor
          keyEqFunc_(std::move(keyEqFunc))      // move-in equality functor
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
        clear();  // delete every node, reset size_ to zero, clear buckets
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
        : htBaseVector_(other.htBaseVector_.size(), nullptr),  // new empty buckets
          head_(nullptr),                                     // list will be rebuilt
          tail_(nullptr),
          maxLoadFactor_(other.maxLoadFactor_),               // copy load factor
          hashFunc_(other.hashFunc_),                         // copy functor
          keyEqFunc_(other.keyEqFunc_)                        // copy comparator
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
            clear();  // free existing nodes and reset state
            // Resize buckets to match ’other’
            htBaseVector_.assign(other.htBaseVector_.size(), nullptr);
            maxLoadFactor_ = other.maxLoadFactor_;
            hashFunc_      = other.hashFunc_;
            keyEqFunc_     = other.keyEqFunc_;
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
        : htBaseVector_(std::move(other.htBaseVector_)),  // steal bucket vector
          head_(other.head_),                            // copy head pointer
          tail_(other.tail_),                            // copy tail pointer
          size_(other.size_),                            // copy size count
          maxLoadFactor_(other.maxLoadFactor_),          // copy load factor
          hashFunc_(std::move(other.hashFunc_)),         // move-in hash functor
          keyEqFunc_(std::move(other.keyEqFunc_))        // move-in equality functor
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
        swap(other);  // leverage strong swap for all internals
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
        return size_ == 0;      // Fast path: compare against zero
    }

    /**
     * @brief Get the number of elements stored in the map.
     *
     * @return Current element count (size_).
     */
    [[nodiscard]] size_t size() const {
        return size_;           // Directly return the counter
    }

    /**
     * @brief Compute the current load factor of the hash table.
     *
     * Load factor = number of elements / number of buckets.
     *
     * @return A double in [0, ∞); typically you aim to keep this ≤ maxLoadFactor_.
     */
    [[nodiscard]] double loadFactor() const {
        return static_cast<double>(size_) / htBaseVector_.size();  // dynamic measure of fullness
    }

    /**
     * @brief Retrieve the number of buckets in the hash table.
     *
     * @return Size of htBaseVector_ (bucket count).
     */
    [[nodiscard]] size_t bucketCount() const {
        return htBaseVector_.size();  // slots available for chaining
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
        maxLoadFactor_ = newMaxLoadFactor;      // update threshold
        // If we're already above the new threshold, redistribute now
        if (loadFactor() > newMaxLoadFactor) {
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
        rehash_(newBucketCount);              // O(N) rehash cost
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
                delete cur;                // free each node
                cur = nxt;
            }
            delete head_;                  // finally delete the original head
        }
        // Reset internal state
        head_ = tail_ = nullptr;
        size_ = 0;
        std::fill(htBaseVector_.begin(), htBaseVector_.end(), nullptr);
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
                cur->value_ = value;       // update existing
                return;
            }
        }

        // 3) Create a fresh node (self-linked in list)
        Node *node = new Node(key, value);

        // 4) Splice into the circular doubly-linked list
        if (where == -1 || where == intSize) {
            // Append at tail
            if (!head_) {
                head_ = tail_ = node;    // first element
            } else {
                linkAfter_(tail_, node);
            }
        }
        else if (where == 0) {
            // Prepend at head
            if (!head_) {
                head_ = tail_ = node;
            } else {
                linkBefore_(head_, node);
            }
        }
        else {
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
        insert_at(key, value, -1);      // convenience overload
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

                delete cur;              // free memory
                --size_;                 // decrement count
                return true;
            }
            cur = cur->hashNext_;
        }
        return false;                 // key not found
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
            return *pv;                   // Fast path: key already present
        }
        insert(key, Value{});            // Splice in a default-constructed Value
        // Note: default Value may not be thrilling, but it gets the job done
        return *find_ptr(key);           // Guaranteed to succeed now
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
            return *p;                    // Good: key found
        }
        throw std::out_of_range("Key not found");  // Bad: no such key
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
            return *p;                    // Returns const reference
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
                return cur;               // Found the node
            }
        }
        return nullptr;                  // Miss
    }

    /**
     * @brief Const overload of find_node().
     */
    const Node *find_node(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap*>(this)->find_node(key);
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
        return const_cast<DoublyLinkedCircularHashMap*>(this)->find_node(key);
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
            return &n->value_;         // Handy direct pointer
        }
        return nullptr;
    }

    /**
     * @brief Const overload of find_ptr().
     */
    const Value *find_ptr(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap*>(this)->find_ptr(key);
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
        return const_cast<DoublyLinkedCircularHashMap*>(this)->find_ptr(key);
    }

    /**
     * @brief Check if a key is present in the map.
     *
     * @param key The key to test.
     * @return true if find_ptr(key) != nullptr, false otherwise.
     */
    [[nodiscard]] bool contains(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap*>(this)->find_ptr(key) != nullptr;
    }

    /**
     * @brief Heterogeneous contains() for key-like types.
     */
    template<class K2>
        requires std::invocable<Hash, const K2 &>
              && std::invocable<KeyEq, const Key &, const K2 &>
    [[nodiscard]] bool contains(const K2 &key) const {
        return const_cast<DoublyLinkedCircularHashMap*>(this)->find_ptr(key) != nullptr;
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
        hashFunc_ = std::forward<HF>(h);     // swap in new hash
        rehash_(htBaseVector_.size());       // rebuild buckets under new hash
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
        keyEqFunc_ = std::forward<KEF>(k);   // swap in new comparator
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Iterator Support -----

    /**
     * @brief Bidirectional iterator for traversing in insertion order.
     *
     * Presents each element as a std::pair<const Key&, Value&>, supporting
     * prefix/postfix ++ and --. Reaching past tail_ yields end() (cur == nullptr).
     */
    struct iterator {
        using iterator_category = std::bidirectional_iterator_tag;   /**< STL category tag. */
        using value_type        = std::pair<const Key&, Value&>;     /**< Dereferenced type. */
        using reference         = value_type;                        /**< Reference alias. */
        using difference_type   = std::ptrdiff_t;                    /**< Signed distance. */
        using pointer           = void;                              /**< Pointer unsupported. */

        Node *cur;                                                 /**< Current node; nullptr means end(). */
        const DoublyLinkedCircularHashMap *map;                   /**< Owning map, for head_/tail_. */

        /**
         * @brief Default-construct an end() iterator.
         */
        iterator() noexcept : cur(nullptr), map(nullptr) {}

        /**
         * @brief Construct an iterator at the given node in the given map.
         * @param node Starting node (nullptr for end()).
         * @param m    Pointer to the map instance.
         */
        iterator(Node *node, const auto *m) : cur(node), map(m) {}

        /**
         * @brief Prefix increment: advance to the next element.
         * @return Reference to this iterator, now pointing to the next element.
         */
        iterator &operator++() {
            if (cur) {
                cur = cur->next_;
                if (cur == map->head_) {
                    cur = nullptr;  // wrapped past tail_ → end()
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
            ++(*this);
            return tmp;
        }

        /**
         * @brief Prefix decrement: move to the previous element.
         * @return Reference to this iterator, now pointing to the previous element.
         */
        iterator &operator--() {
            if (!cur) {
                cur = map->tail_;  // end() → last element
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
            --(*this);
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
    };

    /**
     * @brief Const bidirectional iterator for insertion-order traversal.
     *
     * Identical behavior to iterator, but yields const Value&.
     */
    struct const_iterator {
        using iterator_category = std::bidirectional_iterator_tag;     /**< STL category tag. */
        using value_type        = std::pair<const Key&, const Value&>; /**< Dereferenced type. */
        using reference         = value_type;                          /**< Reference alias. */
        using difference_type   = std::ptrdiff_t;                      /**< Signed distance. */
        using pointer           = void;                                /**< Pointer unsupported. */

        const Node *cur;                                               /**< Current node; nullptr for end(). */
        const DoublyLinkedCircularHashMap *map;                        /**< Owning map pointer. */

        /**
         * @brief Default-construct an end() iterator.
         */
        const_iterator() noexcept : cur(nullptr), map(nullptr) {}

        /**
         * @brief Construct a const_iterator at the given node in the given map.
         * @param node Starting node (nullptr for end()).
         * @param m    Pointer to the map instance.
         */
        const_iterator(const Node *node, const auto *m) : cur(node), map(m) {}

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
            ++(*this);
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
            --(*this);
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
    using reverse_iterator       = std::reverse_iterator<iterator>;
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
            return it;  // nothing to do
        }
        Node *node     = it.cur;
        iterator nextIt = ++it;  // advance before deletion
        remove(node->key_);
        return nextIt;
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
            return iterator(found, this);  // wrap raw node in iterator
        }
        return end();                       // no such element
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
        insert(key, value);                      // Reuse insert for tail insertion
    }

    /**
     * @brief Access the value at the front of the queue.
     *
     * @note Calling front() on an empty map is undefined behavior.
     * @return Pointer to the value stored in head_.
     */
    Value *front() {
        return &head_->value_;                   // head_ points to the first node
    }

    /**
     * @brief Access the value at the back of the queue.
     *
     * @note Calling back() on an empty map is undefined behavior.
     * @return Pointer to the value stored in tail_.
     */
    Value *back() {
        return &tail_->value_;                   // tail_ points to the last node
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
        insert_at(key, value, 0);               // position 0 → new head
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
        Value *pv = front();                     // grab address before removal
        remove(orderedGetNode(0)->key_);         // remove head element
        return pv;                               // pointer now invalid
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
        insert_at(key, value, 0);               // new head = stack top
    }

    /**
     * @brief Peek at the top of the stack.
     *
     * @note Undefined if map is empty.
     * @return Pointer to the value at head_ (stack top).
     */
    Value *top() {
        return front();                          // head_ is the stack top
    }

    /**
     * @brief Peek at the bottom of the stack (oldest element).
     *
     * @note Undefined if map is empty.
     * @return Pointer to the value at tail_ (stack bottom).
     */
    Value *bottom() {
        return back();                           // tail_ is the stack bottom
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
        Value *pv = bottom();                    // grab bottom (since top==front)
        remove(orderedGetNode(-1)->key_);        // remove last element
        return pv;                               // pointer now invalid
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
            return nullptr;                       // nothing to return in empty map

        const int intSize = static_cast<int>(size_);
        int mod_idx = idx % intSize;              // step 1: wrap index into [−size_..size_)

        if (mod_idx < 0)                           // step 2: normalize negative
            mod_idx += intSize;

        // step 3: pick best start point
        const bool start_at_tail = (mod_idx > intSize/2);
        size_t steps;
        Node *cur;

        if (from) {
            // If user specified a start node, walk relative to that
            cur    = from;
            steps  = start_at_tail ? intSize - mod_idx : mod_idx;
        }
        else if (!start_at_tail) {
            // closer to head_: walk forward mod_idx steps
            cur    = head_;
            steps  = mod_idx;
        }
        else {
            // closer to tail_: walk backward (size_ − mod_idx − 1) steps
            cur    = tail_;
            steps  = intSize - mod_idx - 1;
        }

        if (debug) {
            const char *origin = from ? "from" :
                                  (start_at_tail ? "tail" : "head");
            std::cout
                << "orderedGetNode(" << idx << "): reduced idx=" << mod_idx
                << ", starting at " << origin << ", steps=" << steps << "\n";
        }

        // step 4: do the walk
        for (size_t i = 0; i < steps; ++i) {
            cur = start_at_tail ? cur->prev_    // backward
                                : cur->next_;   // forward
        }

        if (debug) {
            std::cout << "  landed on key=" << cur->key_
                      << ", value=" << cur->value_ << "\n";
        }
        return cur;                               // step 5: return result
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
            return;                              // nothing to do

        // Case A: n1 → n2 adjacency
        if (n1->next_ == n2) {
            Node *p = n1->prev_, *q = n2->next_;
            // relink to: p → n2 → n1 → q
            p->next_    = n2;
            n2->prev_   = p;
            n2->next_   = n1;
            n1->prev_   = n2;
            n1->next_   = q;
            q->prev_    = n1;
        }
        // Case B: n2 → n1 adjacency
        else if (n2->next_ == n1) {
            Node *p = n2->prev_, *q = n1->next_;
            // relink to: p → n1 → n2 → q
            p->next_    = n1;
            n1->prev_   = p;
            n1->next_   = n2;
            n2->prev_   = n1;
            n2->next_   = q;
            q->prev_    = n2;
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
            return;                   // Can't move before itself
        }
        unlink_(node);                // Remove from current position
        linkBefore_(target, node);    // Splice into new position
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
            return;                   // Can't move after itself
        }
        unlink_(node);                // Remove from list
        linkAfter_(target, node);     // Insert after target
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
            move_node_after(node, found);   // place after the found node
        } else {
            move_node_before(node, found);  // place before the found node
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
        Node *node = find_node(key);       // locate node by key
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

        const int N      = static_cast<int>(size_);
        const int tail_i = N - 1;

        // 1) Compute modularized source/destination indices in [0..N-1]
        int src_mod = idx % N;
        int dst_mod = (idx + shift) % N;
        if (src_mod < 0) src_mod += N;    // normalize negative
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
        const int raw_dist = std::abs(src_from_head - dst_from_head);
        if (raw_dist <= second_walk) {
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

    static int computeZigzagOffset(size_t idx, size_t offset) {
        int i = static_cast<int>(idx);
        return (i & 1 ? -1 : 1) * (i + 2 * static_cast<int>(offset) + 1) / 2;
    }

    static std::pair<int,int>
    computeZigzagOffsetPair(size_t idx = 0,
                            size_t left_cnt  = 0,
                            size_t right_cnt = 0) {
        return {
            computeZigzagOffset(idx, left_cnt),
            computeZigzagOffset(idx, right_cnt)
        };
    }

    template< typename Index,
    template<typename, typename...> class Container,
    typename AllocIndex,
    typename... Rest,
    IntRange C = Container<Index, AllocIndex,  Rest...> >
    auto find_n_nodes(const C &in, const bool pre_sorted = false) {
        using AllocNodePtr =
            typename std::allocator_traits<AllocIndex>::template rebind_alloc<Node *>;

        using OutContainer = Container<Node *, AllocNodePtr, Rest...>;

        std::vector<int> mod_indices;
        // std::vector<int> dists_from_front;
        // std::vector<int> dists_from_back;
        mod_indices.reserve(in.size());
        // dists_from_front.reserve(in.size());
        // dists_from_back.reserve(in.size());

        OutContainer out;
        if constexpr (requires(OutContainer c) { c.reserve(0); })
            out.reserve(std::size(in));

        const int N      = static_cast<int>(size_);
        const int tail_i = N - 1;

        // 1. Compute modularized indices in [0...N-1]. If negative add size_.
        for (const auto &idx : in) {
            int mod_idx = idx % N;
            if (mod_idx < 0) mod_idx += N;    // normalize negative
            mod_indices.push_back(mod_idx);
            // dists_from_front.push_back(mod_idx);
            // dists_from_back.push_back(tail_i - mod_idx);
        }
        // 2. Sort the indices
        if (!pre_sorted) std::ranges::sort(mod_indices);



        int bnd_low = 0;
        int bnd_high = tail_i;






        return out;
    }


    // Rotating and Reversal

    void rotate(int k) noexcept {
        if (size_ <= 1 || (k %= static_cast<int>(size_)) == 0)
            return;

        // make k positive in [0..size_-1]
        if (k < 0) k += static_cast<int>(size_);

        // walk forward k steps from the current head
        head_ = walk(head_, k, /*forward=*/true);

        // tail is always head->prev_
        tail_ = head_->prev_;
    }

    void reverse() noexcept {
        if (size_ <= 1) return;

        // swap head and tail
        std::swap(head_, tail_);

        // walk through the list and swap next/prev pointers
        Node *cur = head_;
        do {
            std::swap(cur->next_, cur->prev_);
            cur = cur->prev_; // now points to the next node in the original order
        } while (cur != head_);
    }

    // Splicing and Splitting

    iterator splice(iterator pos, DoublyLinkedCircularHashMap &other, iterator first, iterator last) {
        /**
         * This function moves the range [first, last) from other to this, inserting at (before) pos.
         * @param pos: iterator to the position where the range will be inserted
         * @param other: the source container to splice from
         * @param first: iterator to the first element to splice in other
         * @param last: iterator to the last element to splice in other (not included in splice)
         * @return: iterator to the first element after the spliced range
         */
        // nothing to do?
        if (other.empty() || first == last) {
            return pos;
        }

        // identify the subchain [f, ..., L, pos]
        Node *f = first.cur;
        Node *l = last.cur;
        Node *lPrev = l->prev_;

        // count how many nodes we'll move
        size_t count = 0;
        for (Node *cur = f; cur != l; cur = cur->next_) {
            ++count;
        }

        // detach [f, ..., lPrev] from other
        Node *fPrev = f->prev_;
        fPrev->next_ = l;
        l->prev_ = fPrev;

        // fix other.head_, other.tail_, and size_
        other.size_ -= count;
        if (other.size() == 0) {
            other.head_ = other.tail_ = nullptr;
        } else {
            if (other.head_ == f) other.head_ = l;
            if (other.tail_ == lPrev) other.tail_ = fPrev;
        }

        // splice into this before pos
        if (empty()) {
            // this map is empty so the slice becomes the whole map
            head_ = f;
            tail_ = lPrev;
            head_->prev_ = tail_;
            tail_->next_ = head_;
        } else {
            Node *posNode = pos.cur;
            Node *posPrev = posNode ? posNode->prev_ : tail_;

            // link posPrev -> f
            posPrev->next_ = f;
            f->prev_ = posPrev;

            // link lPrev -> posNode (or head_)
            if (posNode) {
                lPrev->next_ = posNode;
                posNode->prev_ = lPrev;
            } else {
                // inserting at end() means wrapping back to le head_
                lPrev->next_ = head_;
                head_->prev_ = lPrev;
            }

            // if we inserted at the front, update head_
            if (posNode == head_) {
                head_ = f;
            }
            // plus always make surer tail_ = head_->prev_
            tail_ = head_->prev_;
        }
        // move each nodes bucket pointers from other to this
        Node *cur = f;
        for (size_t i = 0; i < count; ++i) {
            Node *nxt = cur->next_;
            other.bucketRemove_(cur);
            bucketInsert_(cur);
            cur = nxt;
        }

        // update size
        size_ += count;

        // return pos;
        return iterator(f, this);
    }

    DoublyLinkedCircularHashMap split(const int idx) {
        DoublyLinkedCircularHashMap tailMap(
            htBaseVector_.size(),
            maxLoadFactor_,
            hashFunc_,
            keyEqFunc_
        );
        // first take modulus of idx and handle negatives
        const int N = static_cast<int>(size_);
        if (N == 0) return tailMap; // nothing to split
        int k = idx % N;
        if (k < 0) k += N;

        // check if we need to do anything
        // trivial cases
        if (k == 0) {
            // everything goes into tailMap
            tailMap = std::move(*this);
            clear(); // now *this* is empty
            return tailMap;
        }

        if (k >= N) {
            // nothing to move
            return tailMap;
        }

        // Locate cut point
        Node *cut = orderedGetNode(idx);
        Node *cut_prev = cut->prev_;

        // sever circle in this map
        cut_prev->next_ = head_;
        head_->prev_ = cut_prev;

        // initialize tailMap's list pointers
        tailMap.head_ = cut;
        tailMap.tail_ = tail_;

        // close the circle in tailMap
        tailMap.head_->prev_ = tailMap.tail_;
        tailMap.tail_->next_ = tailMap.head_;

        // fix this map's tail
        tail_ = cut_prev;

        // compute sizes
        const size_t oldSize = size_;
        size_t tail_size = oldSize - static_cast<size_t>(k);
        size_ = static_cast<size_t>(k);
        tailMap.size_ = tail_size;

        // rehome bucket chains for moved segment
        {
            // start at cut node
            Node *cur = cut;
            for (size_t i = 0; i < tail_size; ++i) {
                Node *nxt = cur->next_;
                bucketRemove_(cur);
                tailMap.bucketInsert_(cur);
                cur = nxt;
            }
        }

        return tailMap;
    }

    // Conditional Erasing

    template<class Pred>
        requires std::predicate<Pred, const Key &, Value &>
    size_t erase_if(Pred &&pred, bool verbose = false) {
        size_t erased = 0;

        for (auto it = begin(); it != end();) {
            auto key = (*it).first;
            if (verbose) std::cout << "[erase_if] visiting key=" << key << "\n";

            if (pred(key, (*it).second)) {
                if (verbose) std::cout << "[erase_if] erasing key=" << key << "\n";
                it = erase(it); // your iterator-erase
                ++erased;
            } else {
                ++it;
            }
        }

        if (verbose) std::cout << "[erase_if] done, erased=" << erased << "\n";
        return erased;
    }

    // Debug, Validation, and Utility
    /// Return a vector giving the chain‐length of each bucket.
    [[nodiscard]] std::vector<size_t> bucketSizesCalc() const {
        std::vector<size_t> sizes(htBaseVector_.size(), 0);
        for (size_t b = 0; b < htBaseVector_.size(); ++b) {
            for (Node *n = htBaseVector_[b]; n; n = n->hashNext_) {
                ++sizes[b];
            }
        }
        return sizes;
    }

    /// Print a histogram of bucket loads.
    void printBucketDistribution(std::ostream &os = std::cout) const {
        const auto sizes = bucketSizesCalc();
        os << "Bucket distribution (" << sizes.size() << " buckets):\n";
        for (size_t i = 0; i < sizes.size(); ++i) {
            os << "  [" << i << "] = " << sizes[i] << "\n";
        }
    }

    // Print a histogram of bucket loads using bucketSizes() instead of bucketSizesCalc().
    void printBucketDistribution2(std::ostream &os = std::cout) const {
        os << "Bucket distribution (" << htBaseVector_.size() << " buckets):\n";
        for (size_t i = 0; i < htBaseVector_.size(); ++i) {
            os << "  [" << i << "] = " << bucketSizes()[i] << "\n";
        }
    }

    /// For a given key, show its raw hash() and the bucket index.
    void debugKey(const Key &k, std::ostream &os = std::cout) const {
        size_t h = hashFunc_(k);
        size_t b = h % htBaseVector_.size();
        os << "key=" << k
                << "  hash=" << h
                << "  bucket=" << b << "\n";
    }

    void validate() const {
        // 1) Empty‐map corner
        if (size_ == 0) {
            if (head_ || tail_)
                throw std::runtime_error("Empty map must have null head/tail");
            for (auto b: htBaseVector_) {
                if (b) throw std::runtime_error("Empty map must have no bucket entries");
            }
            return;
        }

        // 2) Validate circular doubly‐linked list in one walk, count nodes
        std::unordered_set<const Node *> seen;
        seen.reserve(size_);

        const Node *cur = head_;
        for (size_t i = 0; i < size_; ++i) {
            if (!cur)
                throw std::runtime_error("List terminated early");
            // next/prev consistency
            if (cur->next_->prev_ != cur || cur->prev_->next_ != cur)
                throw std::runtime_error("List is not properly doubly‐linked");
            // record we saw this node
            if (!seen.insert(cur).second)
                throw std::runtime_error("Node appears twice in list");
            cur = cur->next_;
        }
        // should wrap exactly back to head
        if (cur != head_)
            throw std::runtime_error("List does not wrap back to head");
        // check head/tail pointers
        if (head_->prev_ != tail_ || tail_->next_ != head_)
            throw std::runtime_error("Head/Tail pointers not circularly consistent");

        // 3) Validate bucket chains in one walk, reusing the same 'seen' set
        //    (so we catch duplicates across buckets and ensure every list node is in a bucket)
        size_t bucketCounted = 0;
        for (auto bucketHead: htBaseVector_) {
            for (const Node *n = bucketHead; n; n = n->hashNext_) {
                ++bucketCounted;
                // must have appeared in list
                if (!seen.count(n))
                    throw std::runtime_error("Bucket node not in list");
                // remove from set so at end we can detect missing nodes
                seen.erase(n);
            }
        }
        // total buckets visited must equal size_
        if (bucketCounted != size_)
            throw std::runtime_error("Bucket node‐count != size_");

        // 4) Finally, every node we saw in the list must have shown up in exactly one bucket
        if (!seen.empty())
            throw std::runtime_error("Some list nodes never appeared in any bucket");
    }

    // Key and Value views
    auto keys_view() const {
        return std::views::all(*this)
               | std::views::transform([](auto const &kv) -> const Key & { return kv.first; });
    }

    auto values_view() const {
        return std::views::all(*this)
               | std::views::transform([](auto const &kv) -> const Value & { return kv.second; });
    }

    auto keys() const {
        return keys_view();
    }

    auto values() const {
        return values_view();
    }

    // Swap functions

    void swap(DoublyLinkedCircularHashMap &other) noexcept {
        std::swap(htBaseVector_, other.htBaseVector_);
        std::swap(head_, other.head_);
        std::swap(tail_, other.tail_);
        std::swap(size_, other.size_);
        std::swap(maxLoadFactor_, other.maxLoadFactor_);
        std::swap(hashFunc_, other.hashFunc_);
        std::swap(keyEqFunc_, other.keyEqFunc_);
    }

    friend void swap(DoublyLinkedCircularHashMap &a, DoublyLinkedCircularHashMap &b) noexcept {
        a.swap(b);
    }

    // Deprecated functions

    // original version of shift_idx
    [[deprecated]] void shift_idx_og(const int idx, const int shift) {
        /**
         * Just use the other one, this was my original function i wrote off rip and it works but I asked chatgpt to
         * give me tips on how to make this more readable. My initial implementation is maximally efficient but, its
         * pretty dense and difficult to read. I will leave it here for reference.
         */
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

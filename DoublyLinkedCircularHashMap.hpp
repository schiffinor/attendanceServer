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
    // Node structure
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


    // Member variables
    std::vector<Node *> htBaseVector_; // bucket heads (one per bucket)
    std::vector<size_t> bucketSizes_; // sizes of each bucket
    Node *head_ = nullptr; // head of the doubly linked list
    Node *tail_ = nullptr; // tail of the doubly linked list
    size_t size_ = 0; // number of elements in the map
    double maxLoadFactor_ = 1.0; // maximum load factor before rehashing
    std::function<size_t(const Key &)> hashFunc_; // hash function
    std::function<bool(const Key &, const Key &)> keyEqFunc_; // key equality function
    size_t rehashCount_ = 0; // number of rehashes performed
    size_t maxBucketSize_ = 0; // current max bucket size
    size_t largestBucketIdx_ = SIZE_MAX;

    // Private helpers
    [[nodiscard]] size_t rehashCount() const {
        return rehashCount_;
    }

    [[nodiscard]] size_t bucketSize(size_t idx) const {
        if (idx >= htBaseVector_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return bucketSizes_[idx];
    }

    [[nodiscard]] size_t largestBucketSize() const {
        return maxBucketSize_;
    }

    [[nodiscard]] size_t largestBucketIdx() const {
        return largestBucketIdx_;
    }

    [[nodiscard]] const std::vector<size_t> &bucketSizes() const {
        return bucketSizes_;
    }

    // separate chaining helper funcs
    [[nodiscard]] size_t bucketIndex_(const Key &key) const {
        return hashFunc_(key) % htBaseVector_.size();
    }

    void bucketInsert_(Node *node) {
        auto idx = bucketIndex_(node->key_);
        Node *bhead = htBaseVector_[idx];
        if (!bhead) {
            htBaseVector_[idx] = node;
            node->hashNext_ = node->hashPrev_ = nullptr;
        } else {
            node->hashNext_ = bhead;
            bhead->hashPrev_ = node;
            node->hashPrev_ = nullptr;
            htBaseVector_[idx] = node;
        }
        // update bucket size and max bucket size if necessary
        ++bucketSizes_[idx];
        if (bucketSizes_[idx] > maxBucketSize_) {
            maxBucketSize_ = bucketSizes_[idx];
            largestBucketIdx_ = idx;
        }
    }

    void bucketRemove_(Node *node) {
        auto idx = bucketIndex_(node->key_);
        if (node->hashPrev_) {
            node->hashPrev_->hashNext_ = node->hashNext_;
        } else {
            // this is the first node in the bucket
            htBaseVector_[idx] = node->hashNext_;
        }
        if (node->hashNext_) {
            node->hashNext_->hashPrev_ = node->hashPrev_;
        }
        node->hashPrev_ = node->hashNext_ = nullptr;
        // update bucket size and max bucket size if necessary
        if (bucketSizes_[idx] > 0) {
            --bucketSizes_[idx];
        } else {
            // this shouldn't be possible, so lets throw an error
            throw std::runtime_error("Bucket size is already 0");
        }
        // handle max bucket size
        if (idx == largestBucketIdx_) {
            if (bucketSizes_[idx] < maxBucketSize_) {
                for (size_t i = 0; i < htBaseVector_.size(); ++i) {
                    if (bucketSizes_[i] == maxBucketSize_) {
                        largestBucketIdx_ = i;
                        break;
                    }
                }
                maxBucketSize_ = bucketSizes_[largestBucketIdx_];
                if (maxBucketSize_ == 0) {
                    largestBucketIdx_ = SIZE_MAX;
                }
            }
        }
    }

    // doubly linked list helper funcs
    void linkAfter_(Node *where, Node *node) {
        node->prev_ = where;
        node->next_ = where->next_;
        where->next_->prev_ = node;
        where->next_ = node;
        if (where == tail_) {
            tail_ = node;
        }
    }

    void linkBefore_(Node *where, Node *node) {
        node->next_ = where;
        node->prev_ = where->prev_;
        where->prev_->next_ = node;
        where->prev_ = node;
        if (where == head_) {
            head_ = node;
        }
    }

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

    // rehashing for hashList
    void rehash_(size_t newBucketCount) {
        std::vector<Node *> newTable(newBucketCount, nullptr);
        htBaseVector_.swap(newTable);
        bucketSizes_.assign(newBucketCount, 0);
        maxBucketSize_ = 0;
        largestBucketIdx_ = SIZE_MAX;

        // reinsert every node into new buckets
        if (head_) {
            Node *cur = head_;
            do {
                cur->hashNext_ = cur->hashPrev_ = nullptr;
                bucketInsert_(cur);
                cur = cur->next_;
            } while (cur != head_);
        }
        // update rehash count
        rehashCount_++;
    }

    // Static Utilities

    static Node *walk(Node *__restrict__ start, int steps, bool forward) noexcept {
        while (steps--) {
            // prefetch next pointer
            __builtin_prefetch(forward ? start->next_ : start->prev_, 0, 1);
            start = forward ? start->next_ : start->prev_;
        }
        return start;
    }

    template<typename Container>
    static Container multi_walk(const Container &starts, int steps, bool forward) noexcept {
        // make a mutable copy of the starting pointers
        Container cur = starts;

        // for each step, advance every node in curr[]
        while (--steps) {
            for (auto &n: cur) {
                // prefetch the next or previous pointer for cache-hinting
                __builtin_prefetch(forward ? n->next_ : n->prev_, 0, 1);
                // advance the node
                n = forward ? n->next_ : n->prev_;
            }
        }
        return cur;
    }

public:
    // Constructors and destructors
    explicit DoublyLinkedCircularHashMap(
        size_t initBuckets = 16,
        const double maxLoadFactor = 1.0,
        std::function<size_t(const Key &)> hashFunc = std::hash<Key>(),
        std::function<bool(const Key &, const Key &)> keyEqFunc = std::equal_to<Key>()
    )
        : htBaseVector_(initBuckets, nullptr),
          bucketSizes_(initBuckets, 0),
          head_(nullptr),
          tail_(nullptr),
          maxLoadFactor_(maxLoadFactor),
          hashFunc_(std::move(hashFunc)),
          keyEqFunc_(std::move(keyEqFunc)) {
    }

    ~DoublyLinkedCircularHashMap() {
        clear();
    }

    // Copy and move constructors and assignment operators

    // Copy constructor
    DoublyLinkedCircularHashMap(const DoublyLinkedCircularHashMap &other)
        : htBaseVector_(other.htBaseVector_.size(), nullptr),
          head_(nullptr),
          tail_(nullptr),
          maxLoadFactor_(other.maxLoadFactor_),
          hashFunc_(other.hashFunc_),
          keyEqFunc_(other.keyEqFunc_) {
        if (other.head_) {
            Node *cur = other.head_;
            do {
                insert(cur->key_, cur->value_);
                cur = cur->next_;
            } while (cur != other.head_);
        }
    }

    // Copy assignment operator
    DoublyLinkedCircularHashMap &operator=(const DoublyLinkedCircularHashMap &other) {
        if (this != &other) {
            clear();
            htBaseVector_.assign(other.htBaseVector_.size(), nullptr);
            maxLoadFactor_ = other.maxLoadFactor_;
            hashFunc_ = other.hashFunc_;
            keyEqFunc_ = other.keyEqFunc_;
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

    // Move constructor
    DoublyLinkedCircularHashMap(DoublyLinkedCircularHashMap &&other) noexcept
        : htBaseVector_(std::move(other.htBaseVector_)),
          head_(other.head_),
          tail_(other.tail_),
          size_(other.size_),
          maxLoadFactor_(other.maxLoadFactor_),
          hashFunc_(std::move(other.hashFunc_)),
          keyEqFunc_(std::move(other.keyEqFunc_)) {
        other.head_ = other.tail_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment operator
    DoublyLinkedCircularHashMap &operator=(DoublyLinkedCircularHashMap &&other) noexcept {
        swap(other);
        return *this;
    }

    // Observers
    [[nodiscard]] bool empty() const {
        return size_ == 0;
    }

    [[nodiscard]] size_t size() const {
        return size_;
    }

    [[nodiscard]] double loadFactor() const {
        return static_cast<double>(size_) / htBaseVector_.size();
    }

    [[nodiscard]] size_t bucketCount() const {
        return htBaseVector_.size();
    }

    [[nodiscard]] double maxLoadFactor() const noexcept {
        return maxLoadFactor_;
    }

    void maxLoadFactor(const double newMaxLoadFactor) {
        maxLoadFactor_ = newMaxLoadFactor;
        if (loadFactor() > newMaxLoadFactor) {
            rehash_(bucketCount());
        }
    }

    // Capacity
    void reserve(const size_t newSize) {
        const size_t newBucketCount = std::max<size_t>(
            1, static_cast<int>(std::ceil(static_cast<double>(newSize) / maxLoadFactor_)));
        rehash_(newBucketCount);
    }

    void clear() {
        // delete all nodes in insertion order
        if (head_) {
            Node *cur = head_->next_;
            while (cur != head_) {
                Node *nxt = cur->next_;
                delete cur;
                cur = nxt;
            }
            delete head_;
        }
        head_ = tail_ = nullptr;
        size_ = 0;
        std::fill(htBaseVector_.begin(), htBaseVector_.end(), nullptr);
    }

    void minimize_size() {
        if (const size_t minSize = std::max<size_t>(1, std::ceil(size_ / maxLoadFactor_)); minSize != bucketCount())
            rehash_(minSize);
    }

    // Modifiers
    void insert_at(const Key &key, const Value &value, const int where = -1) {
        // 1. Check if where is valid, where must be index where the node will be inserted. Between 0 and size_.
        const int intSize = static_cast<int>(size_);
        if (where > intSize || where < -(intSize + 1)) {
            throw std::out_of_range("Index out of range");
        }

        // 2. Check if key already exists
        size_t idx = bucketIndex_(key);
        for (Node *cur = htBaseVector_[idx]; cur; cur = cur->hashNext_) {
            if (keyEqFunc_(cur->key_, key)) {
                cur->value_ = value;
                return;
            }
        }

        // 3. Make new node.
        Node *node = new Node(key, value);

        // 4. Insert into the linked list.

        // 4.a. If where is -1 or equal to size_ insert at the end.
        if (where == -1 || where == intSize) {
            if (!head_) {
                head_ = tail_ = node;
            } else {
                linkAfter_(tail_, node);
            }
        } else if (where == 0) {
            if (!head_) {
                head_ = tail_ = node;
            } else {
                linkBefore_(head_, node);
            }
        } else {
            // 4.b. If where is any other accepted int use orderedGetNode to find the node to insert before.
            // 4.b.i. If list is empty, insert at the end and set head and tail.
            if (!head_) {
                head_ = tail_ = node;
            } else {
                Node *cur = orderedGetNode(where);
                // 4.b.ii. If where is non-negative insert before the node, else insert after the node.
                if (where >= 0)
                    linkBefore_(cur, node);
                else
                    linkAfter_(cur, node);
            }
        }

        // 5. Insert into the hashmap bucket.
        bucketInsert_(node);
        size_++;

        // 6. Check load factor and rehash if necessary.
        if (loadFactor() > maxLoadFactor_) {
            rehash_(htBaseVector_.size() * 2);
        }
    }

    void insert(const Key &key, const Value &value) {
        insert_at(key, value, -1);
    }

    bool remove(const Key &key) {
        size_t idx = bucketIndex_(key);
        Node *cur = htBaseVector_[idx];
        while (cur) {
            if (keyEqFunc_(cur->key_, key)) {
                bucketRemove_(cur);

                // unlink from the doubly linked list
                if (cur == head_ && cur == tail_) {
                    head_ = tail_ = nullptr;
                } else {
                    unlink_(cur);
                }

                delete cur;
                size_--;
                return true;
            }
            cur = cur->hashNext_;
        }
        return false;
    }

    // Element Access
    // Accessor for the value associated with the index from linked list.
    Value &operator[](const Key &key) {
        if (Value *pv = find_ptr(key)) {
            return *pv;
        }
        insert(key, Value{});
        // Note: this will create a new node with default value
        return *find_ptr(key);
    }

    // Safer element access
    // throw error if key not present
    Value &at(const Key &key) {
        if (auto p = find_ptr(key)) {
            return *p;
        }
        throw std::out_of_range("Key not found");
    }

    const Value &at(const Key &key) const {
        if (auto p = find_ptr(key)) {
            return *p;
        }
        throw std::out_of_range("Key not found");
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    Value &at(const K2 &key) {
        if (auto p = find_ptr(key)) {
            return *p;
        }
        throw std::out_of_range("Key not found");
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    const Value &at(const K2 &key) const {
        if (auto p = find_ptr(key)) {
            return *p;
        }
        throw std::out_of_range("Key not found");
    }

    // Lookup and find functions
    // The base find function behind pretty much every other find function.
    Node *find_node(const Key &key) {
        size_t idx = bucketIndex_(key);
        for (Node *cur = htBaseVector_[idx]; cur; cur = cur->hashNext_) {
            if (keyEqFunc_(cur->key_, key)) {
                return cur;
            }
        }
        return nullptr;
    }

    //find_ptr but const
    const Node *find_node(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_node(key);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    Node *find_node(const K2 &key) {
        auto idx = bucketIndex_(key);
        for (Node *cur = htBaseVector_[idx]; cur; cur = cur->hashNext_) {
            if (keyEqFunc_(cur->key_, key)) {
                return cur;
            }
        }
        return nullptr;
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    const Node *find_node(const K2 &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_node(key);
    }

    Value *find_ptr(const Key &key) {
        if (Node *n = find_node(key)) {
            return &n->value_; // return address of the value member
        }
        return nullptr;
    }

    const Value *find_ptr(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_ptr(key);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    Value *find_ptr(const K2 &key) {
        if (Node *n = find_node(key)) {
            return &n->value_; // return address of the value member
        }
        return nullptr;
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    const Value *find_ptr(const K2 &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_ptr(key);
    }

    [[nodiscard]] bool contains(const Key &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_ptr(key) != nullptr;
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    [[nodiscard]] bool contains(const K2 &key) const {
        return const_cast<DoublyLinkedCircularHashMap *>(this)->find_ptr(key) != nullptr;
    }

    // Iterator Support
    struct iterator {
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = std::pair<const Key &, Value &>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = void; // not supported

        Node *cur; // current node (nullptr means end())
        const DoublyLinkedCircularHashMap *map; // to access head and tail

        iterator() noexcept : cur(nullptr), map(nullptr) {
        }

        iterator(Node *node, const auto *m)
            : cur(node), map(m) {
        }

        // prefix ++
        iterator &operator++() {
            if (!cur) {
                return *this;
            }
            cur = cur->next_;
            if (cur == map->head_) {
                cur = nullptr;
            }
            return *this;
        }

        // postfix ++
        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        // prefix --
        iterator &operator--() {
            if (!cur) {
                cur = map->tail_;
            } else {
                cur = cur->prev_;
            }
            return *this;
        }

        // postfix --
        iterator operator--(int) {
            iterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(const iterator &other) const {
            return cur == other.cur;
        }

        bool operator!=(const iterator &other) const {
            return cur != other.cur;
        }

        reference operator*() const {
            return {cur->key_, cur->value_};
        }
    };

    struct const_iterator {
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = std::pair<const Key &, const Value &>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = void; // not supported

        const Node *cur; // current node (nullptr means end())
        const DoublyLinkedCircularHashMap *map; // to access head and tail

        const_iterator() noexcept : cur(nullptr), map(nullptr) {
        }

        const_iterator(const Node *node, const auto *m)
            : cur(node), map(m) {
        }

        // prefix ++
        const_iterator &operator++() {
            if (!cur) {
                return *this;
            }
            cur = cur->next_;
            if (cur == map->head_) {
                cur = nullptr;
            }
            return *this;
        }

        // postfix ++
        const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        // prefix --
        const_iterator &operator--() {
            if (!cur) {
                cur = map->tail_;
            } else {
                cur = cur->prev_;
            }
            return *this;
        }

        // postfix --
        const_iterator operator--(int) {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(const const_iterator &other) const {
            return cur == other.cur;
        }

        bool operator!=(const const_iterator &other) const {
            return cur != other.cur;
        }

        reference operator*() const {
            return {cur->key_, cur->value_};
        }
    };

    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() {
        return iterator(head_, this);
    }

    iterator end() {
        return iterator(nullptr, this);
    }

    const_iterator begin() const {
        return const_iterator(head_, this);
    }

    const_iterator end() const {
        return const_iterator(nullptr, this);
    }

    const_iterator cbegin() const {
        return begin();
    }

    const_iterator cend() const {
        return end();
    }

    reverse_iterator rbegin() {
        return reverse_iterator(end());
    }

    reverse_iterator rend() {
        return reverse_iterator(begin());
    }

    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator rend() const {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crbegin() const {
        return rbegin();
    }

    const_reverse_iterator crend() const {
        return rend();
    }

    //erasing iterator
    iterator erase(iterator it) {
        if (it == end()) {
            return it;
        }
        Node *node = it.cur;
        iterator next_it = ++it;
        remove(node->key_);
        return next_it;
    }

    // STL-like find
    // stl find iterator
    iterator find(const Key &key) {
        if (Node *found = find_node(key)) {
            return iterator(found, this);
        }
        return end();
    }

    const_iterator find(const Key &key) const {
        if (const Node *found = find_node(key)) {
            return const_iterator(found, this);
        }
        return end();
    }

    const_iterator cfind(const Key &key) const {
        return find(key);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    iterator find(const K2 &key) {
        if (Node *found = find_node(key)) {
            return iterator(found, this);
        }
        return end();
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    const_iterator find(const K2 &key) const {
        if (const Node *found = find_node(key)) {
            return const_iterator(found, this);
        }
        return end();
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    const_iterator cfind(const K2 &key) const {
        return find(key);
    }

    template<class HF>
    void setHashFunction(HF &&h) {
        hashFunc_ = std::forward<HF>(h);
        rehash_(htBaseVector_.size());
    }

    template<class KEF>
    void setKeyEqFunction(KEF &&k) {
        keyEqFunc_ = std::forward<KEF>(k);
    }

    // FIFO / QUEUE like operations

    void push_back(const Key &key, const Value &value) {
        insert(key, value);
    }

    Value *front() {
        return &head_->value_;
    }

    Value *back() {
        return &tail_->value_;
    }

    void emplace(const Key &key, const Value &value) {
        insert_at(key, value, 0);
    }

    Value *pop_front() {
        Value *pv = front();
        remove(orderedGetNode(0)->key_);
        return pv;
    }

    // LIFO / STACK like operations
    void push_front(const Key &key, const Value &value) {
        insert_at(key, value, 0);
    }

    Value *top() {
        return front();
    }

    Value *bottom() {
        return back();
    }

    Value *pop_back() {
        Value *pv = bottom();
        remove(orderedGetNode(-1)->key_);
        return pv;
    }

    // Advanced Operations

    // Ordered getters
    Node *orderedGetNode(const int idx, Node *from = nullptr, const bool debug = false) {
        if (size_ == 0) return nullptr; // empty container

        // Because our list is circular we will take the modulus of our index relative size, this will allow us to
        // minimize the number of iterations. we will also start from the shorter end of the list.
        // 1. Reduce index to size
        const int intSize = static_cast<int>(size_);
        int mod_idx = idx % intSize;

        //2. Handle negative indices
        if (mod_idx < 0) {
            mod_idx += intSize;
        }

        // 3. Decide whether to start at head or tail
        bool start_at_tail = mod_idx > intSize / 2;
        size_t steps;
        Node *cur;


        if (from) {
            cur = from;
            steps = start_at_tail ? intSize - mod_idx : mod_idx;
        } else if (!start_at_tail) {
            cur = head_;
            steps = mod_idx;
        } else {
            cur = tail_;
            // if mod_idx = size_-k, that's k steps backwards from tail
            steps = intSize - mod_idx - 1;
        }

        if (debug) {
            const char *origin = from ? "from" : start_at_tail ? "tail" : "head";
            std::cout
                    << "Debugging orderedGet:\n"
                    << "When index is large, we reduce to size by taking mod.\n"
                    << "orderedGet(" << idx << "):\n"
                    << "  reduced idx = " << mod_idx << "\n"
                    << "  starting from " << origin
                    << ", steps = " << steps << "\n";
        }

        // 4. Walk forwards or backwards
        for (size_t i = 0; i < steps; ++i) {
            cur = start_at_tail
                      ? cur->prev_ // backwards
                      : cur->next_; // forwards
        }

        // 5. Return pointer to current node
        if (debug) {
            std::cout << "  found node with key = " << cur->key_ << ", value = " << cur->value_ << "\n";
        }
        return cur;
    }

    Value *orderedGet(const int idx, Node *from = nullptr, const bool debug = false) {
        return &orderedGetNode(idx, from, debug)->value_;
    }

    // Positional swapping

    // swap two nodes in the linked list, basically just swaps the pointers
    void pos_swap_node(Node *n1, Node *n2) {
        if (n1 == n2) return;

        // Case A: n1 immediately precedes n2
        if (n1->next_ == n2) {
            Node *p = n1->prev_; // node before n1
            Node *q = n2->next_; // node after n2

            // relink: p → n2 → n1 → q
            p->next_ = n2;
            n2->prev_ = p;
            n2->next_ = n1;
            n1->prev_ = n2;
            n1->next_ = q;
            q->prev_ = n1;
        }
        // Case B: n2 immediately precedes n1 (covers wrap-around too)
        else if (n2->next_ == n1) {
            Node *p = n2->prev_;
            Node *q = n1->next_;

            // relink: p → n1 → n2 → q
            p->next_ = n1;
            n1->prev_ = p;
            n1->next_ = n2;
            n2->prev_ = n1;
            n2->next_ = q;
            q->prev_ = n2;
        }
        // Case C: nodes are non-adjacent
        else {
            // swap the “incoming” pointers of their neighbors
            std::swap(n1->prev_->next_, n2->prev_->next_);
            std::swap(n1->next_->prev_, n2->next_->prev_);
            // swap their own prev_/next_ pointers
            std::swap(n1->prev_, n2->prev_);
            std::swap(n1->next_, n2->next_);
        }

        // finally, if we swapped head_ or tail_, patch those
        if (head_ == n1) head_ = n2;
        else if (head_ == n2) head_ = n1;

        if (tail_ == n1) tail_ = n2;
        else if (tail_ == n2) tail_ = n1;
    }

    void pos_swap_k(const Key &key1, const Key &key2) {
        Node *node1 = find_node(key1);
        Node *node2 = find_node(key2);
        if (!node1 || !node2) {
            throw std::invalid_argument("One of the keys does not exist");
        }
        pos_swap_node(node1, node2);
    }

    template<typename K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void pos_swap_k(const K2 &key1, const K2 &key2) {
        Node *node1 = find_node(key1);
        Node *node2 = find_node(key2);
        if (!node1 || !node2) {
            throw std::invalid_argument("One of the keys does not exist");
        }
        pos_swap_node(node1, node2);
    }


    void pos_swap(const int idx1, const int idx2) {
        // get the nodes at the given indices
        Node *node1 = orderedGetNode(idx1);
        Node *node2 = orderedGetNode(idx2);
        // swap the nodes positions by getting the keys from the nodes and calling above function
        if (!node1 || !node2) {
            throw std::invalid_argument("One of the indices does not exist");
        }
        pos_swap_node(node1, node2);
    }

    // All our move functions

    void move_node_before(Node *node, Node *target) {
        if (node == target) {
            return;
        }
        unlink_(node);
        linkBefore_(target, node);
    }

    void move_node_after(Node *node, Node *target) {
        if (node == target) {
            return;
        }
        unlink_(node);
        linkAfter_(target, node);
    }

    void move_node_before_n_key(Node *node, const Key &targetKey) {
        Node *targetNode = find_node(targetKey);
        if (!targetNode) {
            throw std::invalid_argument("Target key not found");
        }
        move_node_before(node, targetNode);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void move_node_before_n_key(Node *node, const K2 &targetKey) {
        Node *targetNode = find_node(targetKey);
        if (!targetNode) {
            throw std::invalid_argument("Target key not found");
        }
        move_node_before(node, targetNode);
    }

    void move_node_after_n_key(Node *node, const Key &targetKey) {
        Node *targetNode = find_node(targetKey);
        if (!targetNode) {
            throw std::invalid_argument("Target key not found");
        }
        move_node_after(node, targetNode);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void move_node_after_n_key(Node *node, const K2 &targetKey) {
        Node *targetNode = find_node(targetKey);
        if (!targetNode) {
            throw std::invalid_argument("Target key not found");
        }
        move_node_after(node, targetNode);
    }

    void move_node_to_key(Node *node, const Key &targetKey) {
        move_node_before_n_key(node, targetKey);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void move_node_to_key(Node *node, const K2 &targetKey) {
        move_node_before_n_key(node, targetKey);
    }

    void move_node_before_n_idx(Node *node, const int targetIdx) {
        Node *targetNode = orderedGetNode(targetIdx);
        if (!targetNode) {
            throw std::invalid_argument("Target index not found");
        }
        move_node_before(node, targetNode);
    }

    void move_node_after_n_idx(Node *node, const int targetIdx) {
        Node *targetNode = orderedGetNode(targetIdx);
        if (!targetNode) {
            throw std::invalid_argument("Target index not found");
        }
        move_node_after(node, targetNode);
    }

    void move_node_to_idx(Node *node, const int targetIdx) {
        move_node_before_n_idx(node, targetIdx);
    }

    void move_n_key_to_node(const Key &key, Node *target) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_before(node, target);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void move_n_key_to_node(const K2 &key, Node *target) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_before(node, target);
    }

    void move_n_key_to_n_key(const Key &key, const Key &targetKey) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_to_key(node, targetKey);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void move_n_key_to_n_key(const K2 &key, const K2 &targetKey) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_to_key(node, targetKey);
    }

    void move_n_key_to_idx(const Key &key, const int targetIdx) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_to_idx(node, targetIdx);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void move_n_key_to_idx(const K2 &key, const int targetIdx) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        move_node_to_idx(node, targetIdx);
    }

    void move_idx_to_node(const int idx, Node *target) {
        Node *node = orderedGetNode(idx);
        if (!node) {
            throw std::invalid_argument("Index not found");
        }
        move_node_to_node(node, target);
    }

    void move_idx_to_n_key(const int idx, const Key &targetKey) {
        Node *node = orderedGetNode(idx);
        if (!node) {
            throw std::invalid_argument("Index not found");
        }
        move_node_to_n_key(node, targetKey);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void move_idx_to_n_key(const int idx, const K2 &targetKey) {
        Node *node = orderedGetNode(idx);
        if (!node) {
            throw std::invalid_argument("Index not found");
        }
        move_node_to_n_key(node, targetKey);
    }

    void move_idx_to_idx(const int idx, const int targetIdx) {
        Node *node = orderedGetNode(idx);
        if (!node) {
            throw std::invalid_argument("Index not found");
        }
        move_node_to_idx(node, targetIdx);
    }

    // Node shifting functions
    void shift_node(Node *node, const int shift) {
        Node *found = orderedGetNode(shift, node);
        if (!found) {
            throw std::invalid_argument("Shift index not found");
        }
        if (shift == 0) {
            return;
        }
        if (shift > 0) {
            move_node_after(node, found);
        } else {
            move_node_before(node, found);
        }
    }

    void shift_n_key(const Key &key, const int shift) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        shift_node(node, shift);
    }

    template<class K2>
        requires std::invocable<Hash, const K2 &> && std::invocable<KeyEq, const Key &, const K2 &>
    void shift_n_key(const K2 &key, const int shift) {
        Node *node = find_node(key);
        if (!node) {
            throw std::invalid_argument("Key not found");
        }
        shift_node(node, shift);
    }

    void shift_idx(const int idx, const int shift)
    __attribute__((always_inline, hot, optimize("O3"))) {
        if (LIKELY(shift == 0)) return;
        if (UNLIKELY(size_ == 0)) throw std::out_of_range("Index out of range");

        // if (shift == 0) return;
        // if (size_ == 0) throw std::out_of_range("Index out of range");

        const int N = static_cast<int>(size_);
        const int tail_i = N - 1;
        int src_mod = idx % N;
        int dst_mod = (idx + shift) % N;
        if (src_mod < 0) src_mod += N;
        if (dst_mod < 0) dst_mod += N;
        if (LIKELY(src_mod == dst_mod)) return;
        // if (src_mod == dst_mod) return;

        const int src_from_head = src_mod;
        const int dst_from_head = dst_mod;
        const int src_from_tail = tail_i - src_mod;
        const int dst_from_tail = tail_i - dst_mod;
        const int src_walk = std::min(src_from_head, src_from_tail);
        const int dst_walk = std::min(dst_from_head, dst_from_tail);
        const bool from_src = src_walk <= dst_walk;

        // Determine the starting node
        const bool first_dir = from_src ? (src_from_head <= src_from_tail) : (dst_from_head <= dst_from_tail);
        Node *first_ref = first_dir ? head_ : tail_;
        const int first_walk = from_src ? src_walk : dst_walk;
        const int second_walk = !from_src ? src_walk : dst_walk;

        // 1. Walk to the first node
        Node *cur = walk(first_ref, first_walk, first_dir);
        Node *src_node = from_src ? cur : nullptr;
        Node *dst_node = from_src ? nullptr : cur;

        // 2.  Decide on path to second reference, from cur or from head or tail.
        if (const int raw_dist = std::abs(src_from_head - dst_from_head); raw_dist <= dst_walk) {
            cur = walk(cur, raw_dist, first_dir);
        } else {
            const bool second_dir = from_src ? (dst_from_head <= dst_from_tail) : (src_from_head <= src_from_tail);
            Node *second_ref = second_dir ? head_ : tail_;
            cur = walk(second_ref, second_walk, second_dir);
        }
        if (from_src) dst_node = cur;
        else src_node = cur;

        // 3. Move the nodes
        if (shift > 0) move_node_after(src_node, dst_node);
        else move_node_before(src_node, dst_node);
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

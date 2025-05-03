//
// Created by schif on 5/2/2025.
//

#ifndef CENTRALDOUBLYLINKEDLIST_HPP
#define CENTRALDOUBLYLINKEDLIST_HPP
#include <functional>
#include <stdexcept>

template<typename Key, typename Value, typename KeyEq = std::equal_to<Key>>
class CentralDoublyLinkedList {
    struct Node {
        Key key_;
        Value value_;
        Node *left_;
        Node *right_;

        Node(const Key &key, const Value &value)
            : key_(key), value_(value), left_(nullptr), right_(nullptr) {}
    };

    Node *center_ = nullptr;
    Node *leftest_ = nullptr;
    Node *rightest_ = nullptr;
    size_t size_ = 0;
    bool linked_left_ = false;
    int link_balance_ = 0;
    KeyEq keyEqFunc_;


    void link_right_of_(Node *node, Node *where) {
        if (!where) {
            if (!center_) {
                center_ = node;
                rightest_ = node;
                leftest_ = node;
                return;
            }
            // this should not be allowed
            throw std::runtime_error("Cannot link right of null node");
        }
        where->right_ = node;
        node->left_ = where;
        node->right_ = where->right_;
        if (where->right_) {
            where->right_->left_ = node;
        }
        if (!node->right_) {
            rightest_ = node;
        }
    }

    void link_left_of_(Node *node, Node *where) {
        if (!where) {
            if (!center_) {
                center_ = node;
                rightest_ = node;
                leftest_ = node;
                return;
            }
            // this should not be allowed
            throw std::runtime_error("Cannot link left of null node");
        }
        where->left_ = node;
        node->right_ = where;
        node->left_ = where->left_;
        if (where->left_) {
            where->left_->right_ = node;
        }
        if (!node->left_) {
            leftest_ = node;
        }
    }

    void link_right_(Node *node) {
        lik_right_of_(node, center_);
        ++link_balance_;
        linked_left_ = false;
    }

    void link_left_(Node *node) {
        link_left_of_(node, center_);
        --link_balance_;
        linked_left_ = true;
    }

    void link_rightmost_(Node *node) {
        link_right_of_(node, rightest_);
        ++link_balance_;
        linked_left_ = false;
    }

    void link_leftmost_(Node *node) {
        link_left_of_(node, leftest_);
        --link_balance_;
        linked_left_ = true;
    }

    void unlink_(Node *node) {
        if  (node == center_) {
            if (link_balance_ > 0) {
                for (int i = 0; i < link_balance_; ++i) {
                    center_ = center_->right_;
                }
            } else if (link_balance_ < 0) {
                for (int i = 0; i < -link_balance_; ++i) {
                    center_ = center_->left_;
                }
            } else {
                if (linked_left_) {
                    center_ = center_->left_;
                } else {
                    center_ = center_->right_;
                }
            }
            link_balance_ = 0;
            linked_left_ = !linked_left_;
        }
        // link the node's left and right to each other
        if (node->left_) {
            node->left_->right_ = node->right_;
        }
        if (node->right_) {
            node->right_->left_ = node->left_;
        }
        if (node == leftest_) {
            leftest_ = node->right_;
        }
        if (node == rightest_) {
            rightest_ = node->left_;
        }
        // unlink the node from the list
        node->left_ = nullptr;
        node->right_ = nullptr;
    }

    //───────────────────────────────────────────────────────────────────────────//
    // ----- Static Utilities -----

    // ----- Pointer walking functions -----

    /**
     * @brief Advances a single pointer forward or backward, with cache hints.
     * @param start   Starting node.
     * @param steps   Number of hops to make.
     * @param rightward true → use next_; false → use prev_.
     * @return Pointer after walking steps times.
     * @note Uses __builtin_prefetch to hint CPU caching.
     */
    static Node *walk(Node *__restrict__ start, size_t steps, bool rightward) noexcept {
        while (steps--) {
            __builtin_prefetch(rightward ? start->right_ : start->left_, 0, 1);
            start = rightward ? start->right_ : start->left_;
        }
        return start;
    }

    /**
     * @brief Simultaneously walks multiple node pointers in bulk.
     * @tparam Container A container type holding Node* elements.
     * @param starts  Original pointers.
     * @param steps   Steps to advance each pointer.
     * @param rightward Direction of walk.
     * @return New container with updated pointers.
     * @note Prefetch hints in the inner loop for each node.
     */
    template<typename Container>
    static Container multi_walk(const Container &starts, size_t steps, bool rightward) noexcept {
        Container cur = starts; // copy to avoid mutating caller data
        while (--steps) {
            for (auto &n : cur) {
                __builtin_prefetch(rightward ? n->right_ : n->left_, 0, 1);
                n = rightward ? n->right_ : n->left_;
            }
        }
        return cur;
    }

public:

    explicit CentralDoublyLinkedList(
        const size_t initial_size = 0,
        std::function<bool(const Key &, const Key &)> keyEqFunc = std::equal_to<Key>()
    )
        : center_(nullptr), leftest_(nullptr), rightest_(nullptr), keyEqFunc_(keyEqFunc) {}

    ~CentralDoublyLinkedList() {
        clear();
    }

    CentralDoublyLinkedList(const CentralDoublyLinkedList &other) :
        center_(nullptr), leftest_(nullptr), rightest_(nullptr),
        keyEqFunc_(other.keyEqFunc_) {
        for (Node *cur = other.leftest_; cur; cur = cur->right_) {
            link_rightmost_();
        }
    }

    void insert_at(Key &key, Value &value, const int where = 0) {
        Node *node = new Node(key, value);
        if (where == 0) {
            if (linked_left_) {
                link_right_(*node);
            } else {
                link_left_(*node);
            }
        } else if (where == INT_MIN) {
            link_leftmost_(*node);
        } else if (where == INT_MAX) {
            link_rightmost_(*node);
        } else if () {

        }

    }

    Node *orderedGetNode (const int idx, Node *from = nullptr, const bool debug = false) {
        if (idx == INT_MIN || idx == INT_MAX) {
            return idx == INT_MIN ? leftest_ : rightest_;
        }
        const int N = static_cast<int>(size_);
        int mod_idx = idx % N;              // step 1: wrap index into [−size_..size_)
        

    }
};


#endif //CENTRALDOUBLYLINKEDLIST_HPP

//
// Created by zzy on 2021/12/15.
//

#pragma once

// Modified & optimized from RocksDB's InlineSkipList and inspire from crossbeam.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "utility/common_define.h"
#include "utility/config.h"

#include "../infrastructure/random.h"

namespace zFast {

/**
 * Concurrent skip list with remove.
 * Note: Should work with GC algorithm.
 * @tparam Impl class which implement basic operation
 * @tparam Pointer node pointer
 * @tparam ReclaimContext context for cooperative deletion
 * @tparam DecodedKey compare key
 * @tparam RegionType multi skip list region support
 * @tparam max_height max height that skip list can be, and all node should contain sufficient slot for pointer
 * @tparam branching_factor skip list branching factor(eg. 4)
 */
template <class Impl, class Pointer, class ReclaimContext, class DecodedKey, class RegionType, uint16_t max_height,
          uint16_t branching_factor>
class CconcurrentSkipList {
  static_assert(max_height >= 1, "Bad max height.");
  static_assert(branching_factor > 1, "Bad branching factor.");

public:
  inline Impl *impl() { return static_cast<Impl *>(this); }

  inline const Impl *impl() const { return static_cast<const Impl *>(this); }

  // Need extra slot for highest level.
  static constexpr size_t SPLICE_POINTER_COUNT = max_height + 1;

  /**
   * Splice for insertion and deletion.
   */
  struct splice_t final {
    // The invariant of a splice is that prev[i+1].key <= prev[i].key <
    // next[i].key <= next[i+1].key for all i.  That means that if a
    // key is bracketed by prev[i] and next[i] then it is bracketed by
    // all higher levels.  It is _not_ required that prev[i]->next(i) ==
    // next[i] (it probably did at some point in the past, but intervening
    // or concurrent operations might have inserted nodes in between).
    Pointer prev[SPLICE_POINTER_COUNT];
    Pointer next[SPLICE_POINTER_COUNT];
    int height = 0;

    inline void reset() { height = 0; }
  };

private:
  /**
   * Impl interface(which Impl should override all these functions).
   */

  template <RegionType type> inline Pointer head() const { return impl()->template head<type>(); }

  template <RegionType type> inline Pointer tail() const { return impl()->template tail<type>(); }

  // Ignore the current deleting flag and with acquire barrier.
  template <RegionType type> inline Pointer read_next(const Pointer &node, int level) const {
    dbg_assert(node != tail<type>());
    dbg_assert(level >= 0 && level < max_height);
    return impl()->template read_next<type>(node, level);
  }

  // Ignore the current deleting flag with no barrier.
  template <RegionType type> inline Pointer relax_read_next(const Pointer &node, int level) const {
    dbg_assert(node != tail<type>());
    dbg_assert(level >= 0 && level < max_height);
    return impl()->template relax_read_next<type>(node, level);
  }

  // With reading current deleting flag and with acquire barrier.
  template <RegionType type> inline Pointer read_next(const Pointer &node, int level, bool &current_deleting) const {
    dbg_assert(node != tail<type>());
    dbg_assert(level >= 0 && level < max_height);
    return impl()->template read_next<type>(node, level, current_deleting);
  }

  // Set next with release barrier.
  template <RegionType type> inline void set_next(const Pointer &node, int level, const Pointer &val) {
    dbg_assert(node != tail<type>());
    dbg_assert(level >= 0 && level < max_height);
    return impl()->template set_next<type>(node, level, val);
  }

  // Set next with no barrier.
  template <RegionType type> inline void relax_set_next(const Pointer &node, int level, const Pointer &val) {
    dbg_assert(node != tail<type>());
    dbg_assert(level >= 0 && level < max_height);
    return impl()->template relax_set_next<type>(node, level, val);
  }

  // CAS next strong if current is not deleting.
  template <RegionType type>
  inline bool cas_next(const Pointer &node, int level, Pointer &expected, const Pointer &val, bool &current_deleting) {
    dbg_assert(node != tail<type>());
    dbg_assert(level >= 0 && level < max_height);
    return impl()->template cas_next<type>(node, level, expected, val, current_deleting);
  }

  // CAS next weak if current is not deleting.
  template <RegionType type>
  inline bool weak_cas_next(const Pointer &node, int level, Pointer &expected, const Pointer &val,
                            bool &current_deleting) {
    dbg_assert(node != tail<type>());
    dbg_assert(level >= 0 && level < max_height);
    return impl()->template weak_cas_next<type>(node, level, expected, val, current_deleting);
  }

  // Check deleting flag on specific level.
  // The common impl set deleting flag from high level to low level or all in one atomic operation.
  template <RegionType type> inline bool is_deleting(const Pointer &node, int level) const {
    dbg_assert(node != tail<type>()); // May check on header.
    dbg_assert(level >= 0 && level < max_height);
    return impl()->template is_deleting<type>(node, level);
  }

  // Deleting flag should with next pointer and can read and CAS with pointer atomically.
  // The common implementation is use lower bit of pointer and work with pointer use atomic functions.
  // When multi-level exists and can't set it atomically, should set from upper level to low level.
  // Return true if win the race on level 0.
  template <RegionType type> inline bool cas_deleting(const Pointer &node) {
    dbg_assert(node != head<type>() && node != tail<type>()); // Never set on header.
    return impl()->template cas_deleting<type>(node);
  }

  // Get the linked height of current node.
  // This can read relaxed because height only set once when insert and before CAS to link(with barrier).
  template <RegionType type> inline int relax_read_height(const Pointer &node) const {
    dbg_assert(node != head<type>() && node != tail<type>()); // Never set on header.
    return impl()->template relax_read_height<type>(node);
  }

  // This can write relaxed and only set once when insert and before CAS to link(with barrier).
  template <RegionType type>
  inline void relax_set_height_and_reference_count(const Pointer &node, int height, int reference_count) {
    dbg_assert(node != head<type>() && node != tail<type>()); // Never set on header.
    dbg_assert(height >= 1 && height <= max_height);
    // Set to 1 for level 0 and higher level will invoke increase reference.
    dbg_assert(1 == reference_count);
    return impl()->template relax_set_height_and_reference_count<type>(node, height, reference_count);
  }

  // True if before is not zero, or false returned and actually no increment(prevent double free).
  template <RegionType type> inline bool try_increase_reference(const Pointer &node) {
    dbg_assert(node != head<type>() && node != tail<type>()); // Never set on header.
    return impl()->template try_increase_reference<type>(node);
  }

  // Return the reference count after decrement.
  // A common impl is to do reclaim when reach 0 after decrement.
  template <RegionType type> inline int decrease_reference(ReclaimContext &rc, const Pointer &node) {
    dbg_assert(node != head<type>() && node != tail<type>()); // Never set on header.
    return impl()->template decrease_reference<type>(rc, node);
  }

  // Compare the left and right, and return value is >0 if left>right else <0 if left<right else 0.
  template <RegionType type> inline int compare(const Pointer &left, const Pointer &right) const {
    dbg_assert(left != head<type>() && left != tail<type>());
    dbg_assert(right != head<type>() && right != tail<type>());
    return impl()->template compare<type>(left, right);
  }

  // Get the sort key which node represented.
  template <RegionType type> inline DecodedKey decode_key(const Pointer &node) const {
    dbg_assert(node != head<type>() && node != tail<type>());
    return impl()->template decode_key<type>(node);
  }

  // Compare with node and the sort key, and return value is >0 if node>key else <0 if node<key else 0.
  template <RegionType type> inline int compare(const Pointer &node, const DecodedKey &key) const {
    dbg_assert(node != head<type>() && node != tail<type>());
    return impl()->template compare<type>(node, key);
  }

  // Just a hint. Can always return the max_height.
  template <RegionType type> inline int read_now_max_height() const {
    return impl()->template read_now_max_height<type>();
  }

  // Record the height and return the max height now.
  // Invoke only by insert.
  template <RegionType type> inline int record_height(int height) {
    dbg_assert(height >= 1 && height <= max_height);
    return impl()->template record_height<type>(height);
  }

  template <RegionType type> inline void prefetch_for_read_locality(const Pointer &node) const {
    dbg_assert(node != tail<type>());
    return impl()->template prefetch_for_read_locality<type>(node);
  }

  /** Internal functions of skip list. */

  static constexpr uint32_t scaled_inverse_branching = (Crandom::kMaxNext + 1) / branching_factor;
  static_assert(scaled_inverse_branching > 0, "Scaled inverse branching should larger than 0.");

  inline int random_height(Crandom &rnd) const {
    // Increase height with probability 1 in kBranching
    auto height = 1;
    while (height < max_height && rnd.next() < scaled_inverse_branching)
      height++;
    dbg_assert(height > 0 && height <= max_height);
    return height;
  }

  template <RegionType type> inline bool equal(const DecodedKey &key, const Pointer &n) const {
    return 0 == compare<type>(n, key);
  }

  template <RegionType type> inline bool less_than(const DecodedKey &key, const Pointer &n) const {
    return compare<type>(n, key) > 0;
  }

  // Return true if key is greater than the data stored in "n".
  // Tail n is considered infinite.
  // n should not be head.
  template <RegionType type> inline bool key_is_after_node(const Pointer &key, const Pointer &n) const {
    // tail n is considered infinite
    dbg_assert(n != head<type>());
    return n != tail<type>() && compare<type>(n, key) < 0;
  }

  template <RegionType type> inline bool key_is_after_node(const DecodedKey &key, const Pointer &n) const {
    // tail n is considered infinite
    dbg_assert(n != head<type>());
    return n != tail<type>() && compare<type>(n, key) < 0;
  }

  template <RegionType type, bool prefetch_next>
  inline bool do_cooperative_deletion(ReclaimContext &rc, const Pointer &prev, int level, Pointer &node) {
    dbg_assert(prev != tail<type>());
    dbg_assert(node != head<type>() && node != tail<type>());
    dbg_assert(head<type>() == prev || key_is_after_node<type>(node, prev));
    bool deleting;
    auto nxt = read_next<type>(node, level, deleting);
    dbg_assert(nxt != head<type>());
    dbg_assert(tail<type>() == nxt || key_is_after_node<type>(nxt, node));
    if (prefetch_next && nxt != tail<type>())
      prefetch_for_read_locality<type>(nxt);
    if (LIKELY(!deleting))
      return false;

    // Do cooperative deletion.
    bool prev_deleting;
    auto prev_nxt = node;
    while (UNLIKELY(!weak_cas_next<type>(prev, level, prev_nxt, nxt, prev_deleting))) {
      if (UNLIKELY(prev_deleting)) {
        // Previous is deleting and can't physically delete this node.
        // So forward till first valid one.
        do {
          node = nxt;
          if (tail<type>() == node)
            return false; // Tail reached.
          dbg_assert(node != head<type>() && node != tail<type>());
          nxt = read_next<type>(node, level, prev_deleting);
          dbg_assert(nxt != head<type>());
          dbg_assert(tail<type>() == nxt || key_is_after_node<type>(nxt, node));
        } while (UNLIKELY(prev_deleting));
        if (prefetch_next && nxt != tail<type>())
          prefetch_for_read_locality<type>(nxt);
        return false; // No recheck needed.
      }
      // CAS fail means expected mismatch.
      if (LIKELY(prev_nxt != node)) {
        dbg_assert(head<type>() == prev || tail<type>() == prev_nxt || key_is_after_node<type>(prev_nxt, prev));
        // This node already deleted or new node inserted? Reload new one and retry.
        node = prev_nxt;
        return node != tail<type>(); // Need retry if valid and no need to prefetch.
      }
    }
    // Cooperative deletion success and sub the reference count.
    auto ref_cnt = decrease_reference<type>(rc, node);
    dbg_assert(ref_cnt >= 0);

    // prev not changed.
    node = nxt; // Load next and recheck.
    return node != tail<type>();
  }

  template <RegionType type, bool prefetch_next>
  inline Pointer cooperatively_read_next(ReclaimContext &rc, const Pointer &node, int level) {
    dbg_assert(node != tail<type>());
    bool deleting;
    auto nxt = read_next<type>(node, level, deleting);
    dbg_assert(nxt != head<type>());
    dbg_assert(head<type>() == node || tail<type>() == nxt || key_is_after_node<type>(nxt, node));
    if (UNLIKELY(deleting)) {
      // Previous is deleting, and we can do nothing on next, so just forward till valid one.
      Pointer ret;
      do {
        ret = nxt;
        if (tail<type>() == ret)
          return ret; // Tail reached.
        dbg_assert(ret != head<type>() && ret != tail<type>());
        nxt = read_next<type>(ret, level, deleting);
        dbg_assert(nxt != head<type>());
        dbg_assert(tail<type>() == nxt || key_is_after_node<type>(nxt, ret));
      } while (UNLIKELY(deleting));
      if (prefetch_next && nxt != tail<type>())
        prefetch_for_read_locality<type>(nxt);
      return ret;
    } else if (tail<type>() == nxt)
      return nxt;

    // Normalize link if possible.
    while (do_cooperative_deletion<type, prefetch_next>(rc, node, level, nxt))
      ;
    dbg_assert(head<type>() == node || tail<type>() == nxt || key_is_after_node<type>(nxt, node));
    return nxt;
  }

  // When less is true:
  //   Returns the latest node with a key <=/< key.
  //   Return head if there is no such node.
  // When less is false:
  //   Returns the earliest node with a key >=/> key.
  //   Return tail if there is no such node.
  template <RegionType type, bool less, bool allow_equal>
  inline Pointer find_equal_or_near(ReclaimContext &rc, const DecodedKey &key) {
    auto x = head<type>();
    auto level = read_now_max_height<type>() - 1;
    auto last_bigger = tail<type>();
    while (true) {
      auto next = cooperatively_read_next<type, true>(rc, x, level);
      // Make sure the lists are sorted
      dbg_assert(head<type>() == x || tail<type>() == next || key_is_after_node<type>(next, x));
      // Make sure we haven't overshot during our search
      dbg_assert(head<type>() == x || key_is_after_node<type>(key, x) ||
                 (!less && !allow_equal && equal<type>(key, x)));
      // Note: When deletion exists, last_bigger may not appeal in next level.
      auto cmp = (tail<type>() == next || next == last_bigger) ? 1 : compare<type>(next, key);
      // cooperatively_read_next always return no deleting one(currently) so retry not needed.
      if (LIKELY(cmp < 0))
        x = next; // Keep searching in this list
      else if (0 == cmp) {
        if (allow_equal)
          return next;
        // should skip equal
        if (0 == level) {
          if (less)
            return x;
          x = next; // or we need bigger one, let x be the equal one and get next
        } else {
          if (less)
            last_bigger = next, --level; // can simply treat it is bigger and go lower level
          else
            x = next, level = 0; // let x be the equal one and get next on level 0
        }
      } else {
        // cmp > 0
        if (0 == level)
          return less ? x : next;
        // switch to next list, reuse compare<type>() result
        last_bigger = next, --level;
      }
    }
  }

  // Return the last node in the list.
  // Return head if list is empty.
  template <RegionType type> inline Pointer find_last(ReclaimContext &rc) {
    auto x = head<type>();
    auto level = read_now_max_height<type>() - 1;
    while (true) {
      auto next = cooperatively_read_next<type, false>(rc, x, level);
      if (tail<type>() == next) {
        if (0 == level)
          // cooperatively_read_next always return no deleting one(currently) so retry not needed.
          return x;
        else
          --level; // Switch to next list
      } else
        x = next;
    }
  }

  // Traverses a single level of the list, setting *out_prev to the last
  // node before the key and *out_next to the first node after. Assumes
  // that the key is not present in the skip list. On entry, before should
  // point to a node that is before the key, and after should point to
  // a node that is after the key.  after should be tail if a good after
  // node isn't conveniently available.
  template <RegionType type, bool prefetch_before>
  inline void find_splice_for_level(ReclaimContext &rc, const DecodedKey &key, const Pointer &before,
                                    const Pointer &after, int level, Pointer *out_prev, Pointer *out_next) {
    dbg_assert(level >= 0 && level < max_height);
    dbg_assert(before != tail<type>());
    dbg_assert(head<type>() == before || key_is_after_node<type>(key, before));
    dbg_assert(after != head<type>());
    dbg_assert(tail<type>() == after || !key_is_after_node<type>(key, after));
    auto x = before;
    while (true) {
      dbg_assert(x != tail<type>());
      auto next = cooperatively_read_next<type, true>(rc, x, level);
      if (prefetch_before) {
        if (next != tail<type>() && level > 0) {
          auto prefetch = read_next<type>(next, level - 1);
          dbg_assert(prefetch != head<type>());
          if (prefetch != tail<type>())
            prefetch_for_read_locality<type>(prefetch);
        }
      }
      dbg_assert(head<type>() == x || tail<type>() == next || key_is_after_node<type>(next, x));
      dbg_assert(head<type>() == x || key_is_after_node<type>(key, x));
      // Note: When deletion exists, after may not appeal.
      if (tail<type>() == next || next == after || !key_is_after_node<type>(key, next)) {
        // found it
        *out_prev = x;
        *out_next = next;
        return;
      }
      x = next;
    }
  }

  // Recomputes Splice levels from highest_level (inclusive) down to lowest_level (inclusive).
  template <RegionType type>
  inline void recompute_splice_levels(ReclaimContext &rc, const DecodedKey &key, splice_t *splice,
                                      int recompute_level) {
    dbg_assert(recompute_level > 0 && recompute_level <= max_height);
    dbg_assert(recompute_level <= splice->height);
    for (auto i = recompute_level - 1; i >= 0; --i) {
      static_assert(std::is_signed<decltype(i)>::value, "Signed index needed.");
      find_splice_for_level<type, true>(rc, key, splice->prev[i + 1], splice->next[i + 1], i, &splice->prev[i],
                                        &splice->next[i]);
    }
  }

  // Refresh the splice and make sure prev is not deleting(cooperatively delete it).
  template <RegionType type>
  inline void refresh_splice(ReclaimContext &rc, const DecodedKey &key, splice_t *splice, int level) {
    dbg_assert(level >= 0 && level < max_height);
    dbg_assert(level < splice->height);
    auto probe_level = level + 1; // Check from upper level(current level's prev is deleting).
    // No need to mark unlikely here, because upper level may use the same prev as this level.
    while (splice->prev[probe_level] != head<type>() && is_deleting<type>(splice->prev[probe_level], probe_level)) {
      dbg_assert(key_is_after_node<type>(key, splice->prev[probe_level]));
      ++probe_level;
      // Valid till max height(head and tail was set).
      dbg_assert(probe_level <= max_height);
      dbg_assert(probe_level <= splice->height);
      // Can finally exit by reaching the top of splice(head and tail set).
    }
    // Rescan by good head and till the target level.
    static_assert(std::is_signed<decltype(probe_level)>::value, "Signed index needed.");
    for (--probe_level; probe_level >= level; --probe_level)
      find_splice_for_level<type, true>(rc, key, splice->prev[probe_level + 1], splice->next[probe_level + 1],
                                        probe_level, &splice->prev[probe_level], &splice->next[probe_level]);
  }

  // splice should be initialized properly.
  template <RegionType type>
  bool erase_internal(ReclaimContext &rc, const DecodedKey &key, splice_t *splice,
                      bool *concurrent_deleting = nullptr) {
    auto node = splice->next[0]; // Target is the next of level 0.
    dbg_assert(node != head<type>() && node != tail<type>());
    dbg_assert(equal<type>(key, node));

    // Lock it first.
    if (UNLIKELY(!cas_deleting<type>(node))) {
      if (concurrent_deleting != nullptr)
        *concurrent_deleting = true;
      return false; // Concurrent deletion.
    }

    // Written before release barrier(CAS insert) and read after acquire barrier(build splice).
    // So relax is sufficient.
    auto height = relax_read_height<type>(node);
    dbg_assert(height >= 1 && height <= max_height);
    if (UNLIKELY(height > splice->height)) {
      // Retry.
      splice->prev[height] = head<type>();
      splice->next[height] = tail<type>();
      splice->height = height;
      recompute_splice_levels<type>(rc, key, splice, height);
    }

    // Delete from top to low.
    for (auto i = height - 1; i >= 0; --i) {
      static_assert(std::is_signed<decltype(i)>::value, "Signed index needed.");
      // new_next can be dirty if not linked.
      while (true) {
        auto &prev = splice->prev[i];
        dbg_assert(head<type>() == prev || compare<type>(prev, key) < 0);
        auto &next = splice->next[i];
        if (next != node) {
          // Not linked or deleted by cooperation or deleted and other with same key inserted?
          dbg_assert(tail<type>() == next || compare<type>(next, key) >= 0);
          break; // Check next level.
        }
        // Found link and remove it manually.

        // Written before release barrier(CAS insert) and read after acquire barrier(build splice).
        // So relax is sufficient.
        auto new_next = relax_read_next<type>(node, i);
        dbg_assert(new_next != head<type>());
        dbg_assert(equal<type>(key, node));
        dbg_assert(is_deleting<type>(node, i));
        bool deleting;
        // Note: next and splice->next[i] become stale after invoke
        if (LIKELY(cas_next<type>(prev, i, next, new_next, deleting))) {
          // Decrease the reference.
          auto ref_cnt = decrease_reference<type>(rc, node);
          dbg_assert(ref_cnt >= 0);
          if (0 == ref_cnt) {
            // All link detached.
            dbg_assert(equal<type>(key, node)); // Assert that not changed.
            return true;
          }
          break; // success and next level.
        }
        if (UNLIKELY(deleting))
          // Previous is deleting.
          refresh_splice<type>(rc, key, splice, i);
        else
          // Simply re-find in same level.
          find_splice_for_level<type, false>(rc, key, prev, tail<type>(), i, &splice->prev[i], &splice->next[i]);
      }
    }

    // Because concurrent insertion exists. Ref count may not zero and new link attached.
    // But it will be reclaimed when all link removed(deleting mark is set on all levels).
    dbg_assert(equal<type>(key, node)); // Assert that not changed.
    return true;
  }

public:
  // Inserts a node into the skip list.  Node should have sufficient pointer
  // slots for list with max_height. If use_cas is true, then external
  // synchronization is not required, otherwise this method may not be
  // called concurrently with any other insertions and deletions.
  //
  // Regardless of whether use_cas is true, the splice must be owned
  // exclusively by the current thread.  If allow_partial_splice_fix is
  // true, then the cost of insertion is amortized O(log D), where D is
  // the distance from the splice to the inserted key (measured as the
  // number of intervening nodes).  Note that this bound is very good for
  // sequential insertions!  If allow_partial_splice_fix is false then
  // the existing splice will be ignored unless the current key is being
  // inserted immediately after the splice.  allow_partial_splice_fix ==
  // false has worse running time for the non-sequential case O(log N),
  // but a better constant factor.
  //
  // Use random height if height <= 0(target node should have sufficient slots for link pointer).
  //
  // Note: Do node reclaim in caller if return false(even reference is not zero(typically 1)).
  template <RegionType type, bool use_cas>
  bool insert(Crandom &rnd, ReclaimContext &rc, const Pointer &node, splice_t *splice, bool allow_partial_splice_fix,
              int height = 0, bool *concurrent_deleting = nullptr) {
    dbg_assert(node != head<type>() && node != tail<type>());
    auto x = node;
    const auto key_decoded = decode_key<type>(x);
    if (LIKELY(height <= 0))
      height = random_height(rnd); // Use random height is common case.
    else if (height > max_height)
      throw std::runtime_error("Bad skip list node height over the max_height.");
    dbg_assert(height >= 1 && height <= max_height);

    // Just level 0 ref count and higher level invoke increase ref.
    relax_set_height_and_reference_count<type>(node, height, 1);

    // Reset levels link for insertion and also clear the deleting flag.
    // Ignore level 0 because no concurrent deletion when inserting level 0.
    for (auto i = 1; i < height; ++i)
      relax_set_next<type>(node, i, head<type>()); // Special mark.

    // Record max height.
    auto now_max_height = record_height<type>(height);
    dbg_assert(now_max_height <= max_height);
    dbg_assert(now_max_height >= height);

    auto recompute_height = 0;
    if (splice->height < now_max_height) {
      // Either splice has never been used or now_max_height has grown since
      // last use.  We could potentially fix it in the latter case, but
      // that is tricky.
      splice->prev[now_max_height] = head<type>();
      splice->next[now_max_height] = tail<type>();
      splice->height = now_max_height;
      recompute_height = now_max_height;
    } else {
      // Splice is a valid proper-height splice that brackets some
      // key, but does it bracket this one?  We need to validate it and
      // recompute a portion of the splice (levels 0..recompute_height-1)
      // that is a superset of all levels that don't bracket the new key.
      // Several choices are reasonable, because we have to balance the work
      // saved against the extra comparisons required to validate the Splice.
      //
      // One strategy is just to recompute all of orig_splice_height if the
      // bottom level isn't bracketing.  This pessimistically assumes that
      // we will either get a perfect splice hit (increasing sequential
      // inserts) or have no locality.
      //
      // Another strategy is to walk up the splice's levels until we find
      // a level that brackets the key.  This strategy lets the splice
      // hint help for other cases: it turns insertion from O(log N) into
      // O(log D), where D is the number of nodes in between the key that
      // produced the Splice and the current insert (insertion is aided
      // whether the new key is before or after the splice).  If you have
      // a way of using a prefix of the key to map directly to the closest
      // splice out of O(sqrt(N)) splices and we make it so that splices
      // can also be used as hints during read, then we end up with Oshman's
      // and Shavit's SkipTrie, which has O(log log N) lookup and insertion
      // (compare to O(log N) for skip list).
      //
      // We control the pessimistic strategy with allow_partial_splice_fix.
      // A good strategy is probably to be pessimistic for seq_splice_,
      // optimistic if the caller actually went to the work of providing
      // a splice.
      while (recompute_height < now_max_height) {
        if (read_next<type>(splice->prev[recompute_height], recompute_height) != splice->next[recompute_height])
          // splice isn't tight at this level, there must have been some inserts to this
          // location that didn't update the splice.  We might only be a little stale, but if
          // the splice is very stale it would be O(N) to fix it.  We haven't used up any of
          // our budget of comparisons, so always move up even if we are pessimistic about
          // our chances of success.
          ++recompute_height;
        else if (splice->prev[recompute_height] != head<type>() &&
                 !key_is_after_node<type>(key_decoded, splice->prev[recompute_height])) {
          // key is from before splice
          if (allow_partial_splice_fix) {
            // skip all levels with the same node without more comparisons
            auto bad = splice->prev[recompute_height];
            while (splice->prev[recompute_height] == bad)
              ++recompute_height;
          } else // we're pessimistic, recompute everything
            recompute_height = now_max_height;
        } else if (key_is_after_node<type>(key_decoded, splice->next[recompute_height])) {
          // key is from after splice
          if (allow_partial_splice_fix) {
            auto bad = splice->next[recompute_height];
            while (splice->next[recompute_height] == bad)
              ++recompute_height;
          } else
            recompute_height = now_max_height;
        } else // this level brackets the key, we won!
          break;
      }
      if (use_cas) {
        if (UNLIKELY(splice->height > now_max_height)) {
          // Reset to now max height.
          splice->prev[now_max_height] = head<type>();
          splice->next[now_max_height] = tail<type>();
          splice->height = now_max_height;
        }
        // Note: Caused by concurrently deletion, upper level's next may less than checked level.
        //       So just invalid it if not meet the condition.
        //       Upper level's prev always keep the correct order.
        for (auto i = recompute_height + 1; i < splice->height; ++i) {
          if (UNLIKELY(key_is_after_node<type>(key_decoded, splice->next[i]))) {
            recompute_height = i + 1; // Will set to highest which not meet the cond.
            dbg_assert(recompute_height <= now_max_height && now_max_height <= splice->height);
          }
        }
      } else
        dbg_assert(splice->height == now_max_height);
      dbg_assert(head<type>() == splice->prev[splice->height]);
      dbg_assert(tail<type>() == splice->next[splice->height]);
    }
    dbg_assert(recompute_height <= now_max_height);
    if (recompute_height > 0)
      recompute_splice_levels<type>(rc, key_decoded, splice, recompute_height);

    auto splice_is_valid = true;
    if (use_cas) {
      for (auto i = 0; i < height; ++i) {
        auto expected_store = head<type>(); // Special mark set before. And change every CAS.
        while (true) {
          dbg_assert(equal<type>(key_decoded, x)); // Never reuse/change.
          if (0 == i) {
            // Actually prev never make mistake.
            dbg_assert(head<type>() == splice->prev[0] || compare<type>(splice->prev[0], key_decoded) < 0);
            // Checking for duplicate keys on the level 0 is sufficient.
            if (UNLIKELY(splice->next[0] != tail<type>() && compare<type>(splice->next[0], key_decoded) <= 0)) {
              dbg_assert(splice->next[0] != x); // Never inserted.
              // When level 0, next is always >= key.
              dbg_assert(equal<type>(key_decoded, splice->next[0]));
              return false; // duplicate key
            }

            // When insert on level 0, no concurrent can happens, so relax set next.
            // This also clear the deleting flag on level 0.
            relax_set_next<type>(node, 0, splice->next[0]);
          } else {
            // Note: We may get duplicate in upper level caused by dirty deleting node.
            // Actually prev never make mistake.
            dbg_assert(head<type>() == splice->prev[i] || compare<type>(splice->prev[i], key_decoded) < 0);
            while (UNLIKELY(splice->next[i] != tail<type>() && compare<type>(splice->next[i], key_decoded) <= 0)) {
              dbg_assert(splice->next[i] != x); // Impossible myself.
              // Dirty deleting node.
              dbg_assert(is_deleting<type>(splice->next[i], i));
              // compare<type>(splice->next[i], key_decoded) < 0 never happens
              // because we check and reset upper level, the only one is equal
              // and deleting one.
              dbg_assert(equal<type>(key_decoded, splice->next[i]));
              // Splice's next in this level should refresh.
              // Dirty node will be removed by cooperative deletion.
              if (UNLIKELY(is_deleting<type>(splice->prev[i], i)))
                refresh_splice<type>(rc, key_decoded, splice, i);
              else
                find_splice_for_level<type, false>(rc, key_decoded, splice->prev[i], tail<type>(), i, &splice->prev[i],
                                                   &splice->next[i]);
              // previous should still good.
              dbg_assert(head<type>() == splice->prev[i] || compare<type>(splice->prev[i], key_decoded) < 0);
            }

            bool deleting;
            auto original = expected_store;
            if (UNLIKELY(!cas_next<type>(node, i, expected_store, splice->next[i], deleting))) {
              dbg_assert(expected_store == original); // This should never change.
              // Concurrent deletion detected and just abort.
              dbg_assert(deleting);
              dbg_assert(is_deleting<type>(node, i));
              splice->reset(); // Reset splice and just return success.
              if (concurrent_deleting != nullptr)
                *concurrent_deleting = true;
              return true;
            }
            expected_store = splice->next[i]; // Update it.

            // Try increase the reference count.
            if (UNLIKELY(!try_increase_reference<type>(node))) {
              // Concurrent deletion detected and just abort.
              // Note: Deleting flag may set with subsequence step,
              //       and this may happen on any level so is_deleting<type>(node, i)
              //       may also incorrect.
              splice->reset(); // Reset splice and just return success.
              if (concurrent_deleting != nullptr)
                *concurrent_deleting = true;
              return true;
            }
          }

          // Now splice is ok.
          dbg_assert(tail<type>() == splice->next[i] || compare<type>(splice->next[i], key_decoded) > 0);
          dbg_assert(head<type>() == splice->prev[i] || compare<type>(splice->prev[i], key_decoded) < 0);
          bool deleting;
          // Retry is heavy so use strong CAS.
          // Note: splice->next[i] become stale after invoke
          if (LIKELY(cas_next<type>(splice->prev[i], i, splice->next[i], x, deleting)))
            break; // success

          if (i != 0) {
            // Sub the reference.
            auto ref_cnt = decrease_reference<type>(rc, node);
            dbg_assert(ref_cnt >= 0);
            if (UNLIKELY(0 == ref_cnt)) {
              // Concurrent deletion detected and just abort.
              // Note: Deleting flag may set with subsequence step,
              //       and this may happen on any level so is_deleting<type>(node, i)
              //       may also incorrect.
              splice->reset(); // Reset splice and just return success.
              if (concurrent_deleting != nullptr)
                *concurrent_deleting = true;
              return true;
            }
          }

          if (UNLIKELY(deleting))
            // Previous is deleting, so we should use upper level to rescan with cooperative deletion.
            refresh_splice<type>(rc, key_decoded, splice, i);
          else
            // CAS failed, we need to recompute prev and next. It is unlikely
            // to be helpful to try to use a different level as we redo the
            // search, because it should be unlikely that lots of nodes have
            // been inserted between prev[i] and next[i]. No point in using
            // next[i] as the after hint, because we know it is stale.
            find_splice_for_level<type, false>(rc, key_decoded, splice->prev[i], tail<type>(), i, &splice->prev[i],
                                               &splice->next[i]);

          // Since we've narrowed the bracket for level i, we might have
          // violated the Splice constraint between i and i-1.  Make sure
          // we recompute the whole thing next time.
          if (i > 0)
            splice_is_valid = false;
        }
      }
    } else {
      // No insertion or deletion concurrently.
      for (int i = 0; i < height; ++i) {
        if (i >= recompute_height && read_next<type>(splice->prev[i], i) != splice->next[i])
          find_splice_for_level<type, false>(rc, key_decoded, splice->prev[i], tail<type>(), i, &splice->prev[i],
                                             &splice->next[i]);
        if (0 == i) {
          // Actually prev never make mistake.
          dbg_assert(head<type>() == splice->prev[0] || compare<type>(splice->prev[0], key_decoded) < 0);
          // Checking for duplicate keys on the level 0 is sufficient
          if (UNLIKELY(splice->next[0] != tail<type>() && compare<type>(splice->next[0], key_decoded) <= 0))
            return false; // duplicate key
        } else {
          auto bret = try_increase_reference<type>(node);
          dbg_assert(bret);
        }
        // No concurrency no bad splice.
        dbg_assert(tail<type>() == splice->next[i] || compare<type>(splice->next[i], key_decoded) > 0);
        dbg_assert(head<type>() == splice->prev[i] || compare<type>(splice->prev[i], key_decoded) < 0);
        dbg_assert(read_next<type>(splice->prev[i], i) == splice->next[i]);
        relax_set_next<type>(x, i, splice->next[i]);
        dbg_assert(!is_deleting<type>(splice->prev[i], i));
        set_next<type>(splice->prev[i], i, x);
      }
    }

    // Update or reset splice when normally finish the insertion.
    if (splice_is_valid) {
      for (int i = 0; i < height; ++i)
        splice->prev[i] = x;
      dbg_assert(head<type>() == splice->prev[splice->height]);
      dbg_assert(tail<type>() == splice->next[splice->height]);
      for (int i = 0; i < splice->height; ++i) {
        // Upper level without linked by x may have deleting one with same key.
        if (i >= height)
          dbg_assert(tail<type>() == splice->next[i] || compare<type>(splice->next[i], key_decoded) > 0 ||
                     (equal<type>(key_decoded, splice->next[i]) && is_deleting<type>(splice->next[i], i)));
        // splice->next[i] never less(all level either checked or reset)
        // larger, or it must be equal(dirty same key) to key_decoded, and it must be deleting.
        else
          // linked splice, and it must be good order.
          dbg_assert(tail<type>() == splice->next[i] || compare<type>(splice->next[i], key_decoded) > 0);
        dbg_assert(head<type>() == splice->prev[i] || compare<type>(splice->prev[i], key_decoded) <= 0);
        dbg_assert(splice->prev[i + 1] == splice->prev[i] || head<type>() == splice->prev[i + 1] ||
                   compare<type>(splice->prev[i + 1], splice->prev[i]) < 0);
        // Because deletion exists, upper level's next may not appeal in lower level(so relation can be any).
      }
    } else
      splice->reset();
    return true;
  }

  template <RegionType type> inline Pointer find(ReclaimContext &rc, const DecodedKey &key) {
    auto x = find_equal_or_near<type, false, true>(rc, key);
    if (tail<type>() == x)
      return x;
    return equal<type>(key, x) ? x : tail<type>();
  }

  // Returns true iff an entry that compares equal to key is in the list.
  template <RegionType type> inline bool contains(ReclaimContext &rc, const DecodedKey &key) {
    auto x = find<type>(rc, key);
    return x != tail<type>();
  }

  template <RegionType type>
  inline Pointer erase(ReclaimContext &rc, const Pointer &node, const DecodedKey &key, splice_t *splice,
                       bool *concurrent_deleting = nullptr) {
    dbg_assert(node != head<type>() && node != tail<type>());
    dbg_assert(equal<type>(key, node));

    // Assume that node is got by find or iterator, so relax read after acquire barrier.
    auto height = relax_read_height<type>(node);
#if defined(_WIN32) && defined(max)
#undef max
    auto now_max_height = std::max(height, read_now_max_height<type>());
#define max(a, b) (((a) > (b)) ? (a) : (b))
#else
    auto now_max_height = std::max(height, read_now_max_height<type>());
#endif
    dbg_assert(now_max_height >= 1 && now_max_height <= max_height);

    // Init splice.
    splice->prev[now_max_height] = head<type>();
    splice->next[now_max_height] = tail<type>();
    splice->height = now_max_height;
    recompute_splice_levels<type>(rc, key, splice, now_max_height);

    if (splice->next[0] != node) {
      if (concurrent_deleting != nullptr)
        *concurrent_deleting = true;
      return tail<type>(); // Already removed?
    }
    return erase_internal<type>(rc, key, splice, concurrent_deleting) ? node : tail<type>();
  }

  template <RegionType type>
  inline Pointer erase(ReclaimContext &rc, const DecodedKey &key, splice_t *splice,
                       bool *concurrent_deleting = nullptr) {
    auto now_max_height = read_now_max_height<type>();
    dbg_assert(now_max_height >= 1 && now_max_height <= max_height);

    // Init splice.
    splice->prev[now_max_height] = head<type>();
    splice->next[now_max_height] = tail<type>();
    splice->height = now_max_height;
    recompute_splice_levels<type>(rc, key, splice, now_max_height);

    auto node = splice->next[0];
    if (tail<type>() == node || !equal<type>(key, node))
      return tail<type>(); // Not found.
    return erase_internal<type>(rc, key, splice, concurrent_deleting) ? node : tail<type>();
  }

  // Iteration over the contents of a skip list
  template <RegionType type> class iterator {

  private:
    ReclaimContext *rc_;
    CconcurrentSkipList *list_;
    Pointer node_;
    // Intentionally copyable

  public:
    // Initialize an iterator over the specified list.
    // The returned iterator is not valid.
    explicit iterator(ReclaimContext *rc, CconcurrentSkipList *list) : rc_(rc), list_(list), node_(list->tail<type>()) {
      dbg_assert(rc_ != nullptr && list_ != nullptr);
    }

    // Change the underlying skiplist used for this iterator
    // This enables us not changing the iterator without deallocating
    // an old one and then allocating a new one
    inline void set_list(ReclaimContext *rc, CconcurrentSkipList *list) {
      rc_ = rc;
      list_ = list;
      node_ = list_->tail<type>();
      dbg_assert(rc_ != nullptr && list_ != nullptr);
    }

    // Returns true iff the iterator is positioned at a valid node.
    inline bool valid() const { return node_ != list_->tail<type>(); }

    // Returns the key at the current position.
    inline const Pointer &node() const { return node_; }

    // Advances to the next position.
    // REQUIRES: valid()
    template <bool prefetch_next> inline void next() {
      if (node_ != list_->tail<type>())
        node_ = list_->cooperatively_read_next<type, prefetch_next>(*rc_, node_, 0);
    }

    // Advances to the previous position.
    // REQUIRES: valid()
    inline void prev() {
      // Instead of using explicit "prev" links, we just search for the
      // last node that falls before key.
      dbg_assert(node_ != list_->head<type>());
      if (node_ != list_->tail<type>())
        node_ = list_->find_equal_or_near<type, true, false>(*rc_, list_->decode_key<type>(node_));
      if (list_->head<type>() == node_)
        node_ = list_->tail<type>();
    }

    // Advance to the last entry with a key <= target
    inline void seek_last_less_equal(const DecodedKey &target) {
      node_ = list_->find_equal_or_near<type, true, true>(*rc_, target);
      if (list_->head<type>() == node_) // may return head
        node_ = list_->tail<type>();
    }

    // Advance to the last entry with a key < target
    inline void seek_last_less(const DecodedKey &target) {
      node_ = list_->find_equal_or_near<type, true, false>(*rc_, target);
      if (list_->head<type>() == node_) // may return head
        node_ = list_->tail<type>();
    }

    // Advance to the first entry with a key >= target
    inline void seek_first_greater_equal(const DecodedKey &target) {
      node_ = list_->find_equal_or_near<type, false, true>(*rc_, target);
      dbg_assert(node_ != list_->head<type>());
    }

    // Advance to the first entry with a key > target
    inline void seek_first_greater(const DecodedKey &target) {
      node_ = list_->find_equal_or_near<type, false, false>(*rc_, target);
      dbg_assert(node_ != list_->head<type>());
    }

    // Retreat to the last entry with a key <= target
    inline void seek_for_prev(const DecodedKey &target) {
      seek(target);
      if (list_->tail<type>() == node_)
        seek_to_last();
      while (node_ != list_->tail<type>() && list_->less_than<type>(target, node_))
        prev();
    }

    // Position at the first entry in list.
    // Final state of iterator is valid() iff list is not tail<type>().
    template <bool prefetch_next> inline void seek_to_first() {
      node_ = list_->cooperatively_read_next<type, prefetch_next>(*rc_, list_->head<type>(), 0);
    }

    // Position at the last entry in list.
    // Final state of iterator is valid() iff list is not tail<type>().
    inline void seek_to_last() {
      node_ = list_->find_last<type>(*rc_);
      if (list_->head<type>() == node_)
        node_ = list_->tail<type>();
    }
  };

  /// visit all level and do cooperative deletion
  template <RegionType type> void refresh_all_links(ReclaimContext &rc) {
    auto now_max_height = read_now_max_height<type>();
    for (auto level = 0; level < now_max_height; ++level) {
      auto x = head<type>();
      do {
        x = cooperatively_read_next<type, true>(rc, x, level);
      } while (x != tail<type>());
    }
  }

  /**
   * Debug code.
   */

  // Can only validate it when no concurrent modify happens.
  template <RegionType type> void dbg_validate() const {
    // Interate over all levels at the same time, and verify nodes appear in
    // the right order, and nodes appear in upper level also appear in lower
    // levels.
    Pointer nodes[max_height];
    auto now_max_height = read_now_max_height<type>();
    if (UNLIKELY(now_max_height <= 0))
      throw std::runtime_error("Bad max height.");
    for (auto i = 0; i < now_max_height; ++i)
      nodes[i] = head<type>();
    while (nodes[0] != tail<type>()) {
      auto l0_next = read_next<type>(nodes[0], 0);
      if (UNLIKELY(head<type>() == l0_next))
        throw std::runtime_error("Bad next circle.");
      // Don't do cooperative deletion, just skip.
      while (l0_next != tail<type>() && is_deleting<type>(l0_next, 0)) {
        l0_next = read_next<type>(l0_next, 0);
        if (UNLIKELY(head<type>() == l0_next))
          throw std::runtime_error("Bad next circle.");
      }
      if (tail<type>() == l0_next)
        break;
      if (UNLIKELY(nodes[0] != head<type>() && compare<type>(nodes[0], l0_next) >= 0))
        throw std::runtime_error("Bad order.");
      nodes[0] = l0_next;

      auto i = 1;
      while (i < now_max_height) {
        auto next = read_next<type>(nodes[i], i);
        if (UNLIKELY(head<type>() == next))
          throw std::runtime_error("Bad next circle.");
        // Don't do cooperative deletion, just skip.
        while (next != tail<type>() && is_deleting<type>(next, i)) {
          next = read_next<type>(next, i);
          if (UNLIKELY(head<type>() == next))
            throw std::runtime_error("Bad next circle.");
        }
        if (tail<type>() == next)
          break;
        auto cmp = compare<type>(nodes[0], next);
        if (UNLIKELY(cmp > 0))
          throw std::runtime_error("Bad order.");
        if (0 == cmp) {
          if (UNLIKELY(next != nodes[0]))
            throw std::runtime_error("Mismatch node.");
          nodes[i] = next;
        } else
          break;
        ++i;
      }
    }
    for (auto i = 1; i < now_max_height; ++i) {
      if (tail<type>() == nodes[i])
        throw std::runtime_error("Unexpected.");
      auto next = read_next<type>(nodes[i], i);
      if (head<type>() == next)
        throw std::runtime_error("Bad next circle.");
      // Don't do cooperative deletion, just skip.
      while (next != tail<type>() && is_deleting<type>(next, i)) {
        next = read_next<type>(next, i);
        if (head<type>() == next)
          throw std::runtime_error("Bad next circle.");
      }
      if (next != tail<type>())
        throw std::runtime_error("Bad unfinished.");
    }
  }
};

} // namespace zFast

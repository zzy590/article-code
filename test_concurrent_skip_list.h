//
// Created by zzy on 2021/12/17.
//

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "utility/common_define.h"
#include "utility/config.h"
#include "utility/log.h"
#include "utility/time.h"

#include "../container/concurrent_skip_list.h"
#include "../infrastructure/epoch.h"
#include "../infrastructure/random.h"
#include "../infrastructure/thread_context.h"

namespace zFast {

class CconcurrentSkipListTest final {

private:
  static thread_context_t *get_tc() {
    static thread_local thread_context_t *tls_tc = nullptr;
    static thread_local std::aligned_storage<sizeof(thread_context_t)>::type tls_tc_bytes;

    auto rv = tls_tc;
    if (UNLIKELY(rv == nullptr)) {
      rv = new (&tls_tc_bytes) thread_context_t();
      tls_tc = rv;
    }
    return rv;
  }

  typedef Cepoch<64> CcslEpoch;

  static constexpr size_t MAX_HEIGHT = 12;

  struct node_t {
    std::atomic<node_t *> free_level_pointers[MAX_HEIGHT]{};
    std::atomic<node_t *> data_level_pointers[MAX_HEIGHT]{};
    int64_t reuse_epoch = 0;   // Used by free region. Set when removed from data region.
    int64_t recycle_epoch = 0; // Check before put in free region. Set when removed from free region.
    int64_t free_height = 0;
    int64_t data_height = 0;
    std::atomic<int64_t> free_ref{0};
    std::atomic<int64_t> data_ref{0};
    int64_t key = 0;
    int64_t value = 0;
    int64_t node_id = 0; // For debug.
  };

  enum RegionType { free, data };

  typedef node_t *Pointer;

  typedef int64_t DecodedKey;

  class CcslImpl final
      : public CconcurrentSkipList<CcslImpl, Pointer, reclaim_t, DecodedKey, RegionType, MAX_HEIGHT, 4> {

  private:
    mutable node_t free_link_;
    mutable node_t data_link_;
    std::atomic<int> free_max_height_{1};
    std::atomic<int> data_max_height_{1};

    friend class CconcurrentSkipList<CcslImpl, Pointer, reclaim_t, DecodedKey, RegionType, MAX_HEIGHT, 4>;

    template <RegionType type> inline Pointer head() const {
      if (free == type)
        return &free_link_;
      else
        return &data_link_;
    }

    template <RegionType type> inline Pointer tail() const { return nullptr; }

    template <RegionType type> inline Pointer read_next(Pointer node, int level) const {
      dbg_assert(node != nullptr);
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      auto val = reinterpret_cast<uintptr_t>((*link)[level].load(std::memory_order_acquire));
      return reinterpret_cast<Pointer>(val & 0xFFFFFFFFFFFFFFFE);
    }

    template <RegionType type> inline Pointer relax_read_next(Pointer node, int level) const {
      dbg_assert(node != nullptr);
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      auto val = reinterpret_cast<uintptr_t>((*link)[level].load(std::memory_order_relaxed));
      return reinterpret_cast<Pointer>(val & 0xFFFFFFFFFFFFFFFE);
    }

    template <RegionType type> inline Pointer read_next(Pointer node, int level, bool &current_deleting) const {
      dbg_assert(node != nullptr);
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      auto val = reinterpret_cast<uintptr_t>((*link)[level].load(std::memory_order_acquire));
      current_deleting = (val & 1) != 0;
      return reinterpret_cast<Pointer>(val & 0xFFFFFFFFFFFFFFFE);
    }

    template <RegionType type> inline void set_next(Pointer node, int level, Pointer val) {
      dbg_assert(node != nullptr);
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      // Clear deleting flag.
      return (*link)[level].store(val, std::memory_order_release);
    }

    template <RegionType type> inline void relax_set_next(Pointer node, int level, Pointer val) {
      dbg_assert(node != nullptr);
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      // Clear deleting flag.
      return (*link)[level].store(val, std::memory_order_relaxed);
    }

    template <RegionType type>
    inline bool cas_next(Pointer node, int level, Pointer &expected, Pointer val, bool &current_deleting) {
      dbg_assert(node != nullptr);
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      auto bret = (*link)[level].compare_exchange_strong(expected, val);
      auto old_val = reinterpret_cast<uintptr_t>(expected);
      current_deleting = (old_val & 1) != 0;
      expected = reinterpret_cast<Pointer>(old_val & 0xFFFFFFFFFFFFFFFE);
      return bret;
    }

    template <RegionType type>
    inline bool weak_cas_next(Pointer node, int level, Pointer &expected, Pointer val, bool &current_deleting) {
      dbg_assert(node != nullptr);
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      auto bret = (*link)[level].compare_exchange_weak(expected, val);
      auto old_val = reinterpret_cast<uintptr_t>(expected);
      current_deleting = (old_val & 1) != 0;
      expected = reinterpret_cast<Pointer>(old_val & 0xFFFFFFFFFFFFFFFE);
      return bret;
    }

    template <RegionType type> inline bool is_deleting(const Pointer &node, int level) const {
      dbg_assert(node != nullptr);
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      auto val = reinterpret_cast<uintptr_t>((*link)[level].load(std::memory_order_acquire));
      auto bret = (val & 1) != 0;
      dbg_assert(node != head<type>() || !bret);
      return bret;
    }

    template <RegionType type> inline bool cas_deleting(Pointer node) {
      dbg_assert(node != nullptr && node != head<type>());
      auto link = free == type ? &node->free_level_pointers : &node->data_level_pointers;
      for (int i = MAX_HEIGHT - 1; i >= 0; --i) {
        auto p = (*link)[i].load(std::memory_order_relaxed), t = p;
        do {
          if ((reinterpret_cast<uintptr_t>(p) & 1) != 0) {
            if (0 == i)
              return false; // Already deleting.
            else
              break;
          }
          t = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(p) | 1); // Mark as deleting.
        } while (!(*link)[i].compare_exchange_weak(p, t));
      }
      return true;
    }

    template <RegionType type> inline int relax_read_height(const Pointer &node) const {
      dbg_assert(node != nullptr && node != head<type>());
      return free == type ? node->free_height : node->data_height;
    }

    template <RegionType type>
    inline void relax_set_height_and_reference_count(const Pointer &node, int height, int reference_count) {
      dbg_assert(node != nullptr && node != head<type>());
      if (free == type) {
        node->free_height = height;
        node->free_ref = reference_count;
      } else {
        node->data_height = height;
        node->data_ref = reference_count;
      }
    }

    template <RegionType type> inline bool try_increase_reference(const Pointer &node) {
      dbg_assert(node != nullptr && node != head<type>());
      auto before = free == type ? node->free_ref.load(std::memory_order_acquire)
                                 : node->data_ref.load(std::memory_order_acquire);
      do {
        if (0 == before)
          return false;
      } while (free == type ? !node->free_ref.compare_exchange_weak(before, before + 1)
                            : !node->data_ref.compare_exchange_weak(before, before + 1));
      return true;
    }

    template <RegionType type> inline int decrease_reference(reclaim_t &, const Pointer &node) {
      dbg_assert(node != nullptr && node != head<type>());
      return free == type ? --node->free_ref : --node->data_ref;
    }

    template <RegionType type> inline int compare(Pointer left, Pointer right) const {
      if (free == type) {
        if (left->reuse_epoch == right->reuse_epoch)
          return 0;
        return left->reuse_epoch - right->reuse_epoch > 0 ? 1 : -1;
      }
      if (left->key == right->key)
        return 0;
      return left->key - right->key > 0 ? 1 : -1;
    }

    template <RegionType type> inline DecodedKey decode_key(Pointer node) const {
      return free == type ? node->reuse_epoch : node->key;
    }

    template <RegionType type> inline int compare(Pointer node, const DecodedKey &key) const {
      if (free == type) {
        if (node->reuse_epoch == key)
          return 0;
        return node->reuse_epoch - key > 0 ? 1 : -1;
      }
      if (node->key == key)
        return 0;
      return node->key - key > 0 ? 1 : -1;
    }

    // Just a hint. Can always return the max_height.
    template <RegionType type> inline int read_now_max_height() const {
      return free == type ? free_max_height_.load(std::memory_order_acquire)
                          : data_max_height_.load(std::memory_order_acquire);
    }

    // Invoke only by insert.
    template <RegionType type> inline int record_height(int height) {
      dbg_assert(height >= 1 && height <= MAX_HEIGHT);
      if (free == type) {
        auto before = free_max_height_.load(std::memory_order_relaxed);
        do {
          if (before >= height)
            return before;
        } while (!free_max_height_.compare_exchange_weak(before, height));
      } else {
        auto before = data_max_height_.load(std::memory_order_relaxed);
        do {
          if (before >= height)
            return before;
        } while (!data_max_height_.compare_exchange_weak(before, height));
      }
      return height;
    }

    template <RegionType type> inline void prefetch_for_read_locality(Pointer node) const {
      dbg_assert(node != nullptr && node != head<type>());
      PREFETCH(node, 0, 1);
    }
  };

  static int &thread_id() {
    static thread_local int thread_id = -1;
    return thread_id;
  }

  static Pointer node_allocate(CcslEpoch &epoch, bool refresh, CcslImpl &csl, CcslImpl::splice_t *splice) {
    CcslEpoch::epoch_counter_t safe;
    auto epoch_block(epoch.enter_block(get_tc()->slots[TCS_RESERVED], refresh, &safe));

    // Find the smallest epoch take it if safe.
    CcslImpl::iterator<free> it(&get_tc()->reclaim, &csl);
    for (it.seek_to_first<false>(); it.valid(); it.next<false>()) {
      auto ptr = it.node();
      auto check_epoch = ptr->reuse_epoch;
      if (CcslEpoch::safe_to_reclaim(check_epoch, safe)) {
        LOG_TRACE(("thread {} find node {} reuse epoch {} is safe try erase", thread_id(), ptr->node_id, check_epoch));
        // Splice init internal.
        if (csl.erase<free>(get_tc()->reclaim, ptr, check_epoch, splice) != nullptr) {
          // Make sure complete removed.
          while (ptr->free_ref.load(std::memory_order_acquire) != 0)
            csl.erase<free>(get_tc()->reclaim, ptr, check_epoch, splice);
          // Removed.
          LOG_TRACE(("thread {} allocate node {} reuse epoch {} success", thread_id(), ptr->node_id, ptr->reuse_epoch));
          ptr->recycle_epoch = epoch.bump_epoch_for_reclaim();
          return ptr;
        }
      } else
        break;
    }
    return nullptr;
  }

  static void node_free(CcslEpoch &epoch, CcslImpl &csl, Pointer ptr, CcslImpl::splice_t *splice,
                        bool allow_partial_splice_fix) {
    auto epoch_block(epoch.enter_block(get_tc()->slots[TCS_RESERVED], false, nullptr));
    splice->reset();
    LOG_TRACE(
        ("thread {} try free node {} to free region with reuse epoch {}", thread_id(), ptr->node_id, ptr->reuse_epoch));
    auto bret = csl.insert<free, true>(get_tc()->rnd, get_tc()->reclaim, ptr, splice, allow_partial_splice_fix);
    if (bret)
      LOG_TRACE(("thread {} free node {} to free region with reuse epoch {} success", thread_id(), ptr->node_id,
                 ptr->reuse_epoch));
    else
      throw std::runtime_error("Unexpected duplicate epoch.");
  }

  static void validate(CcslEpoch &epoch, CcslImpl &csl) {
    {
      auto epoch_block(epoch.enter_block(get_tc()->slots[TCS_RESERVED], false, nullptr));
      CcslImpl::iterator<data> it(&get_tc()->reclaim, &csl);
      Pointer last_key = nullptr;
      for (it.seek_to_first<true>(); it.valid(); it.next<true>()) {
        auto &node = it.node();
        if (nullptr == last_key)
          last_key = node;
        else if (node->key <= last_key->key)
          throw std::runtime_error("Bad order.");
        if (node->value != node->key * 13 + 7)
          throw std::runtime_error("Bad KV.");
        last_key = node;
      }
    }
    {
      auto epoch_block(epoch.enter_block(get_tc()->slots[TCS_RESERVED], false, nullptr));
      CcslImpl::iterator<free> it(&get_tc()->reclaim, &csl);
      Pointer last_epoch = nullptr;
      for (it.seek_to_first<true>(); it.valid(); it.next<true>()) {
        auto &node = it.node();
        if (nullptr == last_epoch)
          last_epoch = node;
        else if (node->reuse_epoch <= last_epoch->reuse_epoch)
          throw std::runtime_error("Bad order.");
        last_epoch = node;
      }
    }
  }

  static bool insert(CcslEpoch &epoch, CcslImpl &csl, Pointer ptr, CcslImpl::splice_t *splice,
                     bool allow_partial_splice_fix) {
    LOG_TRACE(("thread {} try insert node {} with key {}", thread_id(), ptr->node_id, ptr->key));
    auto bret = csl.insert<data, true>(get_tc()->rnd, get_tc()->reclaim, ptr, splice, allow_partial_splice_fix);
    if (bret) {
      LOG_TRACE(("thread {} insert node {} with key {} success", thread_id(), ptr->node_id, ptr->key));
    }
    return bret;
  }

  static void retry_insert(CcslEpoch &epoch, CcslImpl &csl, Pointer ptr, CcslImpl::splice_t *splice,
                           std::mt19937_64 &rnd, size_t range, int &trys) {
    auto epoch_block(epoch.enter_block(get_tc()->slots[TCS_RESERVED], false, nullptr));
    // Reset once.
    splice->reset();

    do {
      ptr->key = static_cast<int64_t>(rnd() % range);
      ptr->value = ptr->key * 13 + 7;
      ++trys;
    } while (!insert(epoch, csl, ptr, splice, rnd() & 1));
  }

  static Pointer erase(CcslEpoch &epoch, CcslImpl &csl, const DecodedKey &key, CcslImpl::splice_t *splice,
                       int64_t &reuse_epoch) {
    auto epoch_block(epoch.enter_block(get_tc()->slots[TCS_RESERVED], false, nullptr));
    splice->reset();
    LOG_TRACE(("thread {} try erase key {}", thread_id(), key));
    auto ptr = csl.erase<data>(get_tc()->reclaim, key, splice);
    if (ptr != nullptr) {
      // Make sure complete removed.
      while (ptr->data_ref.load(std::memory_order_acquire) != 0)
        csl.erase<data>(get_tc()->reclaim, ptr, key, splice);
      // Removed.
      reuse_epoch = epoch.bump_epoch_for_reclaim();
      LOG_TRACE(("thread {} erase node {} with key {} success, erase epoch {}", thread_id(), ptr->node_id, ptr->key,
                 reuse_epoch));
    }
    return ptr;
  }

public:
  static void debug_test() {
    CcslImpl csl;

    node_t n[2];
    n[0].key = 1;
    n[1].key = 0;

    CcslImpl::splice_t splice;
    csl.insert<data, true>(get_tc()->rnd, get_tc()->reclaim, &n[0], &splice, false);
    csl.insert<data, true>(get_tc()->rnd, get_tc()->reclaim, &n[1], &splice, false);

    // Use iterator.
    CcslImpl::iterator<data> it(&get_tc()->reclaim, &csl);
    for (it.seek_to_first<true>(); it.valid(); it.next<true>()) {
      std::cout << "key: " << it.node()->key << std::endl;
    }

    dbg_assert(csl.contains<data>(get_tc()->reclaim, 0));
    dbg_assert(csl.contains<data>(get_tc()->reclaim, 0));
    csl.dbg_validate<data>();
  }

  static void concurrent_exchange_test() {
    CcslEpoch epoch;
    CcslImpl csl;

    constexpr int thread_number = 16;
    static constexpr int node_count = 1024;
    std::vector<std::unique_ptr<node_t>> nodes;

    epoch.register_epoch_thread(get_tc()->slots[TCS_RESERVED]);

    auto safe = epoch.get_safe_reclaim_epoch(true);

    /**
     * First insert into free list.
     */

    CcslImpl::splice_t splice;
    nodes.reserve(node_count);
    for (auto i = 0; i < node_count; ++i) {
      nodes.emplace_back(new node_t{});
      auto &node = nodes.back();
      // Set epoch and clear key and value.
      node->reuse_epoch = safe;
      node->recycle_epoch = safe;
      node->key = -1;
      node->value = -1;
      node->node_id = i;
      // Insert into free.
      auto bret = csl.insert<free, false>(get_tc()->rnd, get_tc()->reclaim, node.get(), &splice, true);
      if (bret)
        LOG_TRACE(("Init node {} with reuse epoch {}", i, safe));
      else
        throw std::runtime_error("Bad init insert free.");
      --safe;
      while (safe == CcslEpoch::EPOCH_UNPROTECTED)
        --safe;
    }
    csl.dbg_validate<free>();
    CcslImpl::iterator<free> it(&get_tc()->reclaim, &csl);
    auto test_cnt = 0;
    for (it.seek_to_first<true>(); it.valid(); it.next<true>())
      ++test_cnt;
    if (test_cnt != node_count)
      throw std::runtime_error("Bad init count.");

    /**
     * Now do multi thread concurrent test.
     */

    std::atomic<bool> exit{false};

    std::vector<std::thread> threads;
    threads.reserve(thread_number);
    for (auto tid = 0; tid < thread_number; ++tid) {
      threads.emplace_back([&epoch, &csl, tid, &exit]() {
        thread_id() = tid;
        CcslImpl::splice_t splice;

        std::mt19937_64 rnd(tid);
        epoch.register_epoch_thread(get_tc()->slots[TCS_RESERVED]);

        auto run_times = 0, move_times = 0, retry_times = 0, spin_times = 0;
        while (!exit.load(std::memory_order_acquire)) {
          ++run_times;

          // Validate first.
          validate(epoch, csl);

          // Find one free and insert.
          auto ptr = node_allocate(epoch, rnd() & 1, csl, &splice);
          if (ptr != nullptr) {
            auto node_id = ptr->node_id;
            ++move_times;
            // Do random insert.
            auto trys = 0;
            retry_insert(epoch, csl, ptr, &splice, rnd, 2 * node_count, trys);
            if (trys > 1)
              retry_times += trys - 1;
          }

          // Random take one data.
          auto key = static_cast<int64_t>(rnd() % (2 * node_count));
          int64_t reuse_epoch;
          ptr = erase(epoch, csl, key, &splice, reuse_epoch);
          if (ptr != nullptr) {
            dbg_assert(key == ptr->key);
            ++move_times;
            // Success remove one.
            while (!CcslEpoch::safe_to_reclaim(ptr->recycle_epoch, epoch.get_safe_reclaim_epoch(true))) {
              ++spin_times;
              std::this_thread::yield();
            }
            // Now safe to put in free. Update reuse epoch.
            ptr->reuse_epoch = reuse_epoch;
            node_free(epoch, csl, ptr, &splice, rnd() & 1);
          }
        }

        // Validate at final.
        validate(epoch, csl);

        LOG_INFO(("thread {} exit with run {} move {} spin {} retry {}", tid, run_times, move_times, spin_times,
                  retry_times));

        epoch.unregister_epoch_thread(get_tc()->slots[TCS_RESERVED]);
      });
    }

    auto start = utility::Ctime::steady_ms();

    std::this_thread::sleep_for(std::chrono::seconds(60));
    exit.store(true, std::memory_order_release);
    for (auto &t : threads)
      t.join();
    threads.clear();

    csl.dbg_validate<free>();
    csl.dbg_validate<data>();

    // Count.
    {
      auto total_cnt = 0;
      CcslImpl::iterator<free> check_fit(&get_tc()->reclaim, &csl);
      for (check_fit.seek_to_first<true>(); check_fit.valid(); check_fit.next<true>())
        ++total_cnt;
      CcslImpl::iterator<data> check_dit(&get_tc()->reclaim, &csl);
      for (check_dit.seek_to_first<true>(); check_dit.valid(); check_dit.next<true>())
        ++total_cnt;
      if (total_cnt != node_count)
        throw std::runtime_error("Missing node.");
    }

    auto end = utility::Ctime::steady_ms();

    LOG_INFO(("Finish within {}s. Now safe epoch: {}", (end - start) / 1000.0f, epoch.get_safe_reclaim_epoch(true)));

    epoch.unregister_epoch_thread(get_tc()->slots[TCS_RESERVED]);
  }
};

} // namespace zFast

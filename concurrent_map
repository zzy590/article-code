//
// Created by zzy on 2022/4/3.
//

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stack>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utility/common_define.h"
#include "utility/config.h"
#include "utility/log.h"

#include "../infrastructure/epoch.h"
#include "../infrastructure/epoch_callback.h"
#include "../infrastructure/thread_context.h"

#include "concurrent_skip_list.h"

namespace zFast {

template <uint16_t max_height> struct concurrent_map_node_header_t final {
  std::atomic<concurrent_map_node_header_t *> level_pointers[max_height]{};
  std::atomic<uint16_t> reference_count{0};
  uint16_t height{0};
};

using concurrent_map_region_t = int; // region no use

struct default_concurrent_map_conf_t final {
  static constexpr size_t max_epoch_thread_number = 96;
  static constexpr uint16_t max_height = 12;
  static constexpr uint16_t branching_factor = 4;
};

template <class Key, class Value, class Conf = default_concurrent_map_conf_t>
class CconcurrentMap final
    : private CconcurrentSkipList<CconcurrentMap<Key, Value, Conf>, concurrent_map_node_header_t<Conf::max_height> *,
                                  reclaim_t, Key, concurrent_map_region_t, Conf::max_height, Conf::branching_factor> {
  NO_COPY_MOVE(CconcurrentMap);

private:
  static constexpr auto max_epoch_thread_number = Conf::max_epoch_thread_number;
  static constexpr auto max_height = Conf::max_height;
  static constexpr auto branching_factor = Conf::branching_factor;

  using header_t = concurrent_map_node_header_t<max_height>;

  struct node_t final {
    header_t header;
    Key key;
    Value value;

    template <class K, class V> node_t(K &&key, V &&value) : key(std::forward<K>(key)), value(std::forward<V>(value)) {}
  };

  using MapEpoch = Cepoch<max_epoch_thread_number>;

  struct reclaim_context_t final {
    NO_COPY(reclaim_context_t); // only allow move
  public:
    const node_t *node;                // for single node reclaim
    std::vector<const node_t *> nodes; // for multi nodes reclaim

    reclaim_context_t() // for CB default init
        : node(nullptr) {}

    reclaim_context_t(reclaim_context_t &&another) noexcept : node(another.node), nodes(std::move(another.nodes)) {
      another.nodes.clear();
    }

    inline reclaim_context_t &operator=(reclaim_context_t &&another) noexcept {
      node = another.node;
      nodes = std::move(another.nodes);
      another.nodes.clear();
      return *this;
    }
  };

  using MapEpochCB = CepochCallback<thread_context_t, reclaim_context_t>;

  const tcs_e tcs_;
  std::atomic<int> now_height_;
  header_t header_;
  MapEpoch epoch_;
  MapEpochCB epochCB_;

  /**
   * Skip list interfaces.
   */

  using RegionType = concurrent_map_region_t;
  using Pointer = header_t *;
  using DecodedKey = Key;
  using SkipListImpl = CconcurrentSkipList<CconcurrentMap<Key, Value, Conf>, Pointer, reclaim_t, Key, RegionType,
                                           max_height, branching_factor>;

  friend SkipListImpl;

  template <RegionType type> inline Pointer head() const { return const_cast<Pointer>(&header_); }

  template <RegionType type> inline Pointer tail() const { return nullptr; }

  template <RegionType type> inline Pointer read_next(const Pointer &node, int level) const {
    auto val = reinterpret_cast<uintptr_t>(node->level_pointers[level].load(std::memory_order_acquire));
    return reinterpret_cast<Pointer>(val & (UINTPTR_MAX - 1));
  }

  template <RegionType type> inline Pointer relax_read_next(const Pointer &node, int level) const {
    auto val = reinterpret_cast<uintptr_t>(node->level_pointers[level].load(std::memory_order_relaxed));
    return reinterpret_cast<Pointer>(val & (UINTPTR_MAX - 1));
  }

  template <RegionType type> inline Pointer read_next(const Pointer &node, int level, bool &current_deleting) const {
    auto val = reinterpret_cast<uintptr_t>(node->level_pointers[level].load(std::memory_order_acquire));
    current_deleting = (val & 1) != 0;
    return reinterpret_cast<Pointer>(val & (UINTPTR_MAX - 1));
  }

  template <RegionType type> inline void set_next(const Pointer &node, int level, const Pointer &val) {
    // deleting flag will clear because point is always aligned
    return node->level_pointers[level].store(val, std::memory_order_release);
  }

  template <RegionType type> inline void relax_set_next(const Pointer &node, int level, const Pointer &val) {
    // deleting flag will clear because point is always aligned
    return node->level_pointers[level].store(val, std::memory_order_relaxed);
  }

  template <RegionType type>
  inline bool cas_next(const Pointer &node, int level, Pointer &expected, const Pointer &val, bool &current_deleting) {
    auto bret = node->level_pointers[level].compare_exchange_strong(expected, val);
    auto old_val = reinterpret_cast<uintptr_t>(expected);
    current_deleting = (old_val & 1) != 0;
    expected = reinterpret_cast<Pointer>(old_val & (UINTPTR_MAX - 1));
    return bret;
  }

  template <RegionType type>
  inline bool weak_cas_next(const Pointer &node, int level, Pointer &expected, const Pointer &val,
                            bool &current_deleting) {
    auto bret = node->level_pointers[level].compare_exchange_weak(expected, val);
    auto old_val = reinterpret_cast<uintptr_t>(expected);
    current_deleting = (old_val & 1) != 0;
    expected = reinterpret_cast<Pointer>(old_val & (UINTPTR_MAX - 1));
    return bret;
  }

  template <RegionType type> inline bool is_deleting(const Pointer &node, int level) const {
    auto val = reinterpret_cast<uintptr_t>(node->level_pointers[level].load(std::memory_order_acquire));
    return (val & 1) != 0;
  }

  template <RegionType type> inline bool cas_deleting(const Pointer &node) {
    for (int i = max_height - 1; i >= 0; --i) {
      auto p = node->level_pointers[i].load(std::memory_order_acquire);
      Pointer t;
      do {
        if ((reinterpret_cast<uintptr_t>(p) & 1) != 0) {
          if (0 == i)
            return false; // already deleting
          else
            break; // just ignore and down level
        }
        t = reinterpret_cast<Pointer>(reinterpret_cast<uintptr_t>(p) | 1); // Mark as deleting.
      } while (UNLIKELY(!node->level_pointers[i].compare_exchange_weak(p, t)));
    }
    return true;
  }

  template <RegionType type> inline int relax_read_height(const Pointer &node) const { return node->height; }

  template <RegionType type>
  inline void relax_set_height_and_reference_count(const Pointer &node, int height, int reference_count) {
    node->height = height;
    node->reference_count.store(reference_count, std::memory_order_relaxed);
  }

  template <RegionType type> inline bool try_increase_reference(const Pointer &node) {
    // This read always after relax_set_height_and_reference_count so never make mistake to zero.
    auto before = node->reference_count.load(std::memory_order_relaxed);
    do {
      if (0 == before)
        return false;
    } while (UNLIKELY(!node->reference_count.compare_exchange_weak(before, before + 1, std::memory_order_relaxed,
                                                                   std::memory_order_relaxed)));
    return true;
  }

  template <RegionType type> inline int decrease_reference(reclaim_t &rc, const Pointer &node) {
    auto ref_cnt = node->reference_count.fetch_sub(1, std::memory_order_release);
    if (1 == ref_cnt) {
      // All removed.
      std::atomic_thread_fence(std::memory_order_acquire); // Barrier for reclaim.
      auto data_node = reinterpret_cast<const node_t *>(reinterpret_cast<uintptr_t>(node) - offsetof(node_t, header));
      rc.emplace(data_node);
    }
    return ref_cnt;
  }

  template <RegionType type> inline int compare(const Pointer &left, const Pointer &right) const {
    auto &left_key =
        reinterpret_cast<const node_t *>(reinterpret_cast<uintptr_t>(left) - offsetof(node_t, header))->key;
    auto &right_key =
        reinterpret_cast<const node_t *>(reinterpret_cast<uintptr_t>(right) - offsetof(node_t, header))->key;
    if (left_key < right_key)
      return -1;
    else if (left_key == right_key)
      return 0;
    return 1;
  }

  template <RegionType type> inline DecodedKey decode_key(const Pointer &node) const {
    return reinterpret_cast<const node_t *>(reinterpret_cast<uintptr_t>(node) - offsetof(node_t, header))->key;
  }

  template <RegionType type> inline int compare(const Pointer &node, const DecodedKey &key) const {
    auto &node_key =
        reinterpret_cast<const node_t *>(reinterpret_cast<uintptr_t>(node) - offsetof(node_t, header))->key;
    if (node_key < key)
      return -1;
    else if (node_key == key)
      return 0;
    return 1;
  }

  template <RegionType type> inline int read_now_max_height() const {
    return now_height_.load(std::memory_order_acquire);
  }

  template <RegionType type> inline int record_height(int height) {
    auto before = now_height_.load(std::memory_order_acquire);
    do {
      if (before >= height)
        return before;
    } while (UNLIKELY(!now_height_.compare_exchange_weak(before, height)));
    return height;
  }

  template <RegionType type> inline void prefetch_for_read_locality(const Pointer &node) const { PREFETCH(node, 0, 1); }

  /**
   * Epoch reclaim callback.
   */

  static void node_reclaim(thread_context_t &, reclaim_context_t &&ctx, bool &) {
    if (ctx.node != nullptr) {
      dbg_assert(ctx.nodes.empty());
      delete ctx.node;
    } else {
      dbg_assert(!ctx.nodes.empty());
      for (const auto &item : ctx.nodes) {
        dbg_assert(item != nullptr);
        delete item;
      }
      ctx.nodes.clear();
    }
  }

  inline void schedule_node_reclaim(
      thread_context_t &tc, reclaim_context_t &&ctx, typename MapEpoch::epoch_counter_t reclaim_epoch,
      typename MapEpoch::epoch_counter_t *now_safe_epoch_ptr = nullptr) { // auto update to new one if valid ptr
    // init safe epoch if needed
    typename MapEpoch::epoch_counter_t tmp_safe;
    auto &now_safe_epoch(nullptr == now_safe_epoch_ptr ? tmp_safe : *now_safe_epoch_ptr);
    if (nullptr == now_safe_epoch_ptr || MapEpoch::EPOCH_UNPROTECTED == now_safe_epoch)
      now_safe_epoch = epoch_.get_safe_reclaim_epoch(true);

    auto help_drain = false;

    // Check first.
    if (now_safe_epoch >= reclaim_epoch)
      // Safe to reclaim.
      node_reclaim(tc, std::forward<reclaim_context_t>(ctx), help_drain);
    else {
      auto retry = 0;
      while (UNLIKELY(!epochCB_.register_callback(tc, now_safe_epoch, reclaim_epoch, node_reclaim,
                                                  std::forward<reclaim_context_t>(ctx), help_drain))) {
        // CB ctx is moved out only when success, so just ignore the warning.
        if (++retry > 500) {
          retry = 0;
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          LOG_WARN(("Slowdown: Unable to add callback to epoch in concurrent map."));
        }
        // Refresh safe epoch.
        now_safe_epoch = epoch_.get_safe_reclaim_epoch(true);
        if (now_safe_epoch >= reclaim_epoch) {
          // Safe to reclaim.
          node_reclaim(tc, std::forward<reclaim_context_t>(ctx), help_drain);
          break; // check help drain
        }
        // Or schedule again.
      }
    }
    while (help_drain) {
      help_drain = false;
      epochCB_.drain(tc, epoch_.get_safe_reclaim_epoch(true), help_drain);
    }
  }

  inline void schedule_async_node_reclaim(
      thread_context_t &tc,
      typename MapEpoch::epoch_counter_t *now_safe_epoch_ptr = nullptr) { // auto update to new one if valid ptr
    auto &tls_container = tc.reclaim;                                     // reuse the thread reclaim container
    if (LIKELY(tls_container.empty()))
      return;

    // bump epoch first
    auto reclaim_epoch = epoch_.bump_epoch_for_reclaim();

    reclaim_context_t ctx;
    if (1 == tls_container.size()) {
      // only one need to reclaim
      ctx.node = reinterpret_cast<const node_t *>(tls_container.top());
      tls_container.pop();
      dbg_assert(tls_container.empty());
    } else {
      // multiple nodes reclaim
      ctx.nodes.reserve(tls_container.size());
      while (!tls_container.empty()) {
        ctx.nodes.emplace_back(reinterpret_cast<const node_t *>(tls_container.top()));
        tls_container.pop();
      }
      dbg_assert(ctx.nodes.size() >= 2);
    }

    // init safe epoch if needed
    typename MapEpoch::epoch_counter_t tmp_safe;
    auto &now_safe_epoch(nullptr == now_safe_epoch_ptr ? tmp_safe : *now_safe_epoch_ptr);
    if (nullptr == now_safe_epoch_ptr || MapEpoch::EPOCH_UNPROTECTED == now_safe_epoch)
      now_safe_epoch = epoch_.get_safe_reclaim_epoch(true);
    schedule_node_reclaim(tc, std::move(ctx), reclaim_epoch, &now_safe_epoch);
  }

  /**
   * Auto epoch access framework.
   */

  class CautoEpochBlock final {
    NO_COPY_MOVE(CautoEpochBlock);

  private:
    thread_context_t &tc_;
    CconcurrentMap *ctx_;
    typename MapEpoch::CautoEpochBlock base_;

  public:
    explicit CautoEpochBlock(thread_context_t &tc, CconcurrentMap *ctx, bool refresh = false,
                             typename MapEpoch::epoch_counter_t *safe_reclaim_epoch = nullptr)
        : tc_(tc), ctx_(ctx), base_(ctx->epoch_.enter_block(tc.slots[ctx->tcs_], refresh, safe_reclaim_epoch)) {
      if (tc_.reclaim_in_use)
        throw std::runtime_error("TLS reclaim container already in use.");
      tc_.reclaim_in_use = true;
      dbg_assert(tc_.reclaim.empty()); // will reuse the thread reclaim container, so assert empty before use
    }

    ~CautoEpochBlock() { leave(); }

    inline void leave() {
      if (ctx_ != nullptr) {
        base_.leave();
        // do reclaim
        ctx_->schedule_async_node_reclaim(tc_);
        dbg_assert(tc_.reclaim.empty()); // after schedule reclaim, container should be empty
        if (!tc_.reclaim_in_use)
          throw std::runtime_error("TLS reclaim container unexpected released.");
        tc_.reclaim_in_use = false;
        ctx_ = nullptr;
      }
    }
  };

public:
  explicit CconcurrentMap(tcs_e tcs, size_t epoch_callback_slot_size = 256)
      : tcs_(tcs), now_height_(1), epochCB_(epoch_callback_slot_size) {}

  ~CconcurrentMap() {
    // check whether skip list is empty
    for (auto i = 0; i < max_height; ++i) {
      if (header_.level_pointers[i].load(std::memory_order_acquire) != nullptr) {
        LOG_ERROR(("CconcurrentMap exit with not all elements removed."));
        ::abort();
      }
    }
  }

  void finalize(thread_context_t &tc) {
    // release and drain all
    typename SkipListImpl::splice_t splice;

    // assert that unregister before destruct
    epoch_.register_epoch_thread(tc.slots[tcs_]);

    CautoEpochBlock block(tc, this);
    typename SkipListImpl::template iterator<0> it(&tc.reclaim, this);
    for (it.template seek_to_first<true>(); it.valid(); it.template next<true>()) {
      auto node = reinterpret_cast<const node_t *>(reinterpret_cast<uintptr_t>(it.node()) - offsetof(node_t, header));
      auto bret = SkipListImpl::template erase<0>(tc.reclaim, it.node(), node->key, &splice);
      dbg_assert(bret);
    }
    // visit all level and make sure all deleting node removed correctly
    SkipListImpl::template refresh_all_links<0>(tc.reclaim);
    // do leave and auto reclaim
    block.leave();

    // unregister
    epoch_.unregister_epoch_thread(tc.slots[tcs_]);

    // drain all
    bool retry_drain;
    do {
      retry_drain = false;
      epochCB_.drain(tc, epoch_.get_safe_reclaim_epoch(true), retry_drain);
    } while (retry_drain);
    // now all memory freed
  }

  inline void register_thread(thread_context_t &tc) { epoch_.register_epoch_thread(tc.slots[tcs_]); }

  inline void unregister_thread(thread_context_t &tc) { epoch_.unregister_epoch_thread(tc.slots[tcs_]); }

  template <class K, class V> inline bool emplace(thread_context_t &tc, K &&key, V &&value) {
    auto ptr = std::make_unique<node_t>(std::forward<K>(key), std::forward<V>(value));

    typename SkipListImpl::splice_t splice;
    CautoEpochBlock block(tc, this);
    auto bret = SkipListImpl::template insert<0, true>(tc.rnd, tc.reclaim, &ptr->header, &splice, false);
    if (bret)
      ptr.release(); // release management to skip list
    // or release via unique_ptr
    return bret;
  }

  inline bool retrieve(thread_context_t &tc, const Key &key, Value &value) {
    CautoEpochBlock block(tc, this);
    auto hdr = SkipListImpl::template find<0>(tc.reclaim, key);
    if (nullptr == hdr)
      return false;
    auto node = reinterpret_cast<const node_t *>(reinterpret_cast<uintptr_t>(hdr) - offsetof(node_t, header));
    dbg_assert(key == node->key);
    value = node->value;
    return true;
  }

  template <class Consumer> inline bool read_modify_write(thread_context_t &tc, const Key &key, Consumer &&consumer) {
    CautoEpochBlock block(tc, this);
    auto hdr = SkipListImpl::template find<0>(tc.reclaim, key);
    if (nullptr == hdr)
      return false;
    auto node = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(hdr) - offsetof(node_t, header));
    dbg_assert(key == node->key);
    consumer(node->value);
    return true;
  }

  inline bool erase(thread_context_t &tc, const Key &key) {
    typename SkipListImpl::splice_t splice;
    CautoEpochBlock block(tc, this);
    return SkipListImpl::template erase<0>(tc.reclaim, key, &splice) != nullptr;
  }

  template <class Consumer> inline bool read_modify_write_min(thread_context_t &tc, Consumer &&consumer) {
    CautoEpochBlock block(tc, this);
    auto it = typename SkipListImpl::template iterator<0>(&tc.reclaim, this);
    it.template seek_to_first<false>();
    if (!it.valid())
      return false;
    auto node = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(it.node()) - offsetof(node_t, header));
    consumer(reinterpret_cast<const node_t *>(node)->key, node->value);
    return true;
  }

  template <class Consumer> inline bool read_modify_write_max(thread_context_t &tc, Consumer &&consumer) {
    CautoEpochBlock block(tc, this);
    auto it = typename SkipListImpl::template iterator<0>(&tc.reclaim, this);
    it.seek_to_last();
    if (!it.valid())
      return false;
    auto node = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(it.node()) - offsetof(node_t, header));
    consumer(reinterpret_cast<const node_t *>(node)->key, node->value);
    return true;
  }
};

} // namespace zFast

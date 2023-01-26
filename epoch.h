//
// Created by zzy on 2021/12/4.
//

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <type_traits>

#include "utility/common_define.h"
#include "utility/config.h"
#include "utility/log.h"

#include "thread_context.h"

namespace zFast {

/**
 * General epoch framework with overflow and rewind dealing.
 * @tparam max_epoch_thread_number Max thread number can register to this epoch. Every 64 for one memory page.
 * @tparam epoch_counter_type Type of epoch counter, which must be signed integer.
 */
template <size_t max_epoch_thread_number, class epoch_counter_type = int64_t> class Cepoch final {
  NO_COPY_MOVE(Cepoch);
  static_assert(std::is_signed<epoch_counter_type>::value && std::is_integral<epoch_counter_type>::value,
                "Epoch counter should be signed integral.");
  static_assert(sizeof(epoch_counter_type) >= 2, "Epoch counter should be at least i16.");

public:
  using epoch_counter_t = epoch_counter_type;
  static constexpr epoch_counter_t EPOCH_UNPROTECTED = 0;

private:
  // For now CPU, int64_t never rewind in 50 years.
  static constexpr bool REWIND_CHECK = sizeof(epoch_counter_t) < 8;
  static constexpr epoch_counter_t MAX_SAFE_GAP =
      static_cast<epoch_counter_t>((UINT64_C(1) << (sizeof(epoch_counter_t) * 8 - 2)) - 1);

#pragma pack(push, 8)

  struct ALIGNED(CACHE_LINE_FETCH_ALIGN) epoch_slot_t {
    std::atomic<epoch_counter_t> local_current_epoch{EPOCH_UNPROTECTED};
    intptr_t reentrant{0};
    std::thread::id thread_id;
    bool is_partitioned{false};
    std::atomic_flag used = ATOMIC_FLAG_INIT;

    inline bool acquire(bool partitioned = false) {
      if (!used.test_and_set(std::memory_order_acquire)) {
        dbg_assert(local_current_epoch.is_lock_free()); // Assume that is totally lock free.
        // Success get the slot.
        local_current_epoch.store(EPOCH_UNPROTECTED, std::memory_order_seq_cst);
        reentrant = 0;
        thread_id = std::this_thread::get_id();
        is_partitioned = partitioned;
        return true;
      }
      return false;
    }

    inline void release(bool partitioned = false) {
      dbg_assert(used.test_and_set(std::memory_order_relaxed)); // Assert already acquire.
      dbg_assert(thread_id == std::this_thread::get_id());
      dbg_assert(is_partitioned == partitioned);
      if (UNLIKELY(UNLIKELY(local_current_epoch.load(std::memory_order_relaxed) != EPOCH_UNPROTECTED) ||
                   UNLIKELY(reentrant != 0)))
        throw std::runtime_error("Bad epoch thread not exit protection.");
      thread_id = std::thread::id(); // Reset the thread.
      is_partitioned = false;        // Reset the partition flag.
      used.clear(std::memory_order_release);
    }

    inline void enter(epoch_counter_t now_epoch) {
      dbg_assert(now_epoch != EPOCH_UNPROTECTED);
      dbg_assert(used.test_and_set(std::memory_order_relaxed)); // Assert already in use.
      dbg_assert(std::this_thread::get_id() == thread_id);
      // Relax read because this only written by myself.
      auto expected = EPOCH_UNPROTECTED;
      // Use CAS and acq_rel order make others see this value before enter the scope and prevent reorder.
      if (local_current_epoch.compare_exchange_strong(expected, now_epoch, std::memory_order_acq_rel))
        dbg_assert(0 == reentrant);
      else
        dbg_assert(reentrant > 0);
      ++reentrant;
    }

    inline bool is_protected() const {
      dbg_assert(std::this_thread::get_id() == thread_id);
      // Relax read because this only written by myself.
      return local_current_epoch.load(std::memory_order_relaxed) != EPOCH_UNPROTECTED;
    }

    inline void leave() {
      dbg_assert(used.test_and_set(std::memory_order_relaxed)); // Assert already in use.
      dbg_assert(std::this_thread::get_id() == thread_id);
      // Relax read because this only written by myself.
      dbg_assert(local_current_epoch.load(std::memory_order_relaxed) != EPOCH_UNPROTECTED);
      dbg_assert(reentrant > 0);
      if (0 == --reentrant)
        // Release write make this never sees before finish scope.
        local_current_epoch.store(EPOCH_UNPROTECTED, std::memory_order_release);
    }
  };

#pragma pack(pop)

  static_assert(CACHE_LINE_FETCH_ALIGN == sizeof(epoch_slot_t), "Bad epoch_slot_t align and size");

  cache_padded_t<std::atomic<epoch_counter_t>> epoch_;
  mutable std::atomic<epoch_counter_t> safe_reclaim_epoch_; // For fast safe epoch acquire.
  epoch_slot_t slots_[max_epoch_thread_number];

  inline epoch_counter_t get_safe_reclaim_epoch_internal(epoch_counter_t now_epoch) const {
    auto safe_epoch = now_epoch;
    for (size_t i = 0; i < max_epoch_thread_number; ++i) {
      auto ongoing = slots_[i].local_current_epoch.load(std::memory_order_acquire);
      if (ongoing != EPOCH_UNPROTECTED && ongoing < safe_epoch)
        safe_epoch = ongoing;
    }
    --safe_epoch;
    if (REWIND_CHECK && UNLIKELY(EPOCH_UNPROTECTED == safe_epoch))
      --safe_epoch; // Skip magic value.
    // Strictly grow forwardly is ok(Even some threads enter with lower epoch, they can't see the freed one).
    auto last_safe_epoch = safe_reclaim_epoch_.load(std::memory_order_acquire);
    while (safe_epoch - last_safe_epoch > 0) {
      if (safe_reclaim_epoch_.compare_exchange_weak(last_safe_epoch, safe_epoch))
        return safe_epoch;
    }
    return last_safe_epoch;
  }

  /**
   * Get epoch of enter. Spin if now epoch far away from safe epoch to prevent bad safe check when rewind.
   * @param refresh Whether refresh and get realtime safe reclaim epoch.
   * @param safe_reclaim_epoch Set if want to get safe epoch.
   * @return Now epoch.
   */
  inline epoch_counter_t get_epoch_for_enter(bool refresh, epoch_counter_t *safe_reclaim_epoch) {
    auto now_epoch = epoch_().load(std::memory_order_acquire);
    if (REWIND_CHECK && UNLIKELY(EPOCH_UNPROTECTED == now_epoch))
      now_epoch = ++epoch_(); // Skip magic.
    if (REWIND_CHECK || safe_reclaim_epoch != nullptr) {
      auto safe = refresh ? get_safe_reclaim_epoch_internal(now_epoch)           // Refresh and get realtime value.
                          : safe_reclaim_epoch_.load(std::memory_order_acquire); // Lazy acquire.
      // Spin wait until safe.
      while (REWIND_CHECK && UNLIKELY(now_epoch - safe >= MAX_SAFE_GAP)) {
        std::this_thread::yield();
        now_epoch = epoch_().load(std::memory_order_acquire);
        if (UNLIKELY(EPOCH_UNPROTECTED == now_epoch))
          now_epoch = ++epoch_();                          // Skip magic.
        safe = get_safe_reclaim_epoch_internal(now_epoch); // Acquire and update cache.
      }
      if (safe_reclaim_epoch != nullptr)
        *safe_reclaim_epoch = safe;
    }
    return now_epoch;
  }

public:
  explicit Cepoch() : epoch_(EPOCH_UNPROTECTED + 2), safe_reclaim_epoch_(EPOCH_UNPROTECTED + 1) {
    static_assert(0 == offsetof(Cepoch, epoch_) && CACHE_LINE_FETCH_ALIGN == offsetof(Cepoch, safe_reclaim_epoch_) &&
                      2 * CACHE_LINE_FETCH_ALIGN == offsetof(Cepoch, slots_),
                  "Bad align of Cepoch.");
  }

  ~Cepoch() {
    // Check and acquire all slot to destruct.
    for (size_t i = 0; i < max_epoch_thread_number; ++i) {
      if (UNLIKELY(!slots_[i].acquire())) {
        LOG_ERROR(("Epoch exit with not all threads unregistered."));
        ::abort();
      }
    }
  }

  /**
   * Only one epoch manager(same region) can one thread belongs to as same time.
   * Do not register and unregister frequently.
   */
  inline void register_epoch_thread(tcs_t &thread_id) {
    static_assert(DEFAULT_TCS_VAL < 0, "Bad default TCS value.");
    if (UNLIKELY(thread_id >= 0))
      throw std::runtime_error("Epoch thread already registered.");
    // Scan and find one empty.
    for (size_t i = 0; i < max_epoch_thread_number; ++i) {
      auto &slot = slots_[i];
      if (slot.acquire()) {
        // Success get the slot, and store in thread local.
        thread_id = static_cast<tcs_t>(i);
        break;
      }
    }
    if (UNLIKELY(thread_id < 0))
      throw std::runtime_error("Epoch thread slot already full.");
  }

  inline void unregister_epoch_thread(tcs_t &thread_id) {
    if (UNLIKELY(thread_id < 0))
      throw std::runtime_error("Epoch thread not registered.");
    if (UNLIKELY(thread_id >= max_epoch_thread_number))
      throw std::runtime_error("Bad epoch thread id.");
    slots_[thread_id].release();
    thread_id = -1;
  }

  /**
   * For register on different epoch instance with same region.
   * Should register the main partition first with normal register.
   */
  inline void register_epoch_thread_partitioned(const tcs_t &thread_id) {
    if (UNLIKELY(thread_id < 0))
      throw std::runtime_error("Epoch thread not registered in main partition.");
    // Register the same slot and this must success.
    if (UNLIKELY(thread_id >= max_epoch_thread_number))
      throw std::runtime_error("Bad epoch thread id.");
    if (!slots_[thread_id].acquire(true))
      throw std::runtime_error("Epoch thread slot already registered by others when register partitioned.");
  }

  /**
   * Invoke this in other partition epoch first,
   * and unregister in main partition with normal function.
   */
  inline void unregister_epoch_thread_partitioned(const tcs_t &thread_id) {
    if (UNLIKELY(thread_id < 0))
      throw std::runtime_error("Epoch thread not registered.");
    if (UNLIKELY(thread_id >= max_epoch_thread_number))
      throw std::runtime_error("Bad epoch thread id.");
    slots_[thread_id].release(true);
    // Only erase the slot info in normal unregister.
  }

  inline epoch_counter_t get_safe_reclaim_epoch(bool refresh) const {
    return refresh ? get_safe_reclaim_epoch_internal(epoch_().load(std::memory_order_acquire))
                   : safe_reclaim_epoch_.load(std::memory_order_acquire);
  }

  inline epoch_counter_t enter(const tcs_t &thread_id, bool refresh, epoch_counter_t *safe_reclaim_epoch) {
    dbg_assert(thread_id >= 0 && thread_id < max_epoch_thread_number);
    auto now_epoch = get_epoch_for_enter(refresh, safe_reclaim_epoch);
    slots_[thread_id].enter(now_epoch);
    return now_epoch;
  }

  inline bool is_protected(const tcs_t &thread_id) const {
    dbg_assert(thread_id >= 0 && thread_id < max_epoch_thread_number);
    return slots_[thread_id].is_protected();
  }

  /**
   * Bump epoch version and get previous one for reclaim.
   * @return Epoch for reclaim.
   */
  inline epoch_counter_t bump_epoch_for_reclaim() {
    auto before = epoch_()++;
    if (REWIND_CHECK) {
      while (UNLIKELY(EPOCH_UNPROTECTED == before))
        before = epoch_()++; // Prevent safe one is magic.
      auto after = before + 1;
      while (UNLIKELY(EPOCH_UNPROTECTED == after))
        after = ++epoch_(); // Prevent now is magic.
    }
    return before;
  }

  inline void leave(const tcs_t &thread_id) {
    dbg_assert(thread_id >= 0 && thread_id < max_epoch_thread_number);
    slots_[thread_id].leave();
  }

  class CautoEpochBlock final {
    NO_COPY(CautoEpochBlock);

  private:
    Cepoch *epoch_;
    tcs_t thread_id_;
    Cepoch::epoch_counter_t now_epoch_;

  public:
    CautoEpochBlock() : epoch_(nullptr), thread_id_(-1), now_epoch_(EPOCH_UNPROTECTED) {}

    explicit CautoEpochBlock(Cepoch &epoch, const tcs_t &thread_id, bool refresh, epoch_counter_t *safe_reclaim_epoch)
        : epoch_(&epoch), thread_id_(thread_id), now_epoch_(epoch.enter(thread_id, refresh, safe_reclaim_epoch)) {}

    CautoEpochBlock(CautoEpochBlock &&another) noexcept
        : epoch_(another.epoch_), thread_id_(another.thread_id_), now_epoch_(another.now_epoch_) {
      another.epoch_ = nullptr;
    }

    inline CautoEpochBlock &operator=(CautoEpochBlock &&another) noexcept {
      epoch_ = another.epoch_;
      thread_id_ = another.thread_id_;
      now_epoch_ = another.now_epoch_;
      another.epoch_ = nullptr;
      return *this;
    }

    ~CautoEpochBlock() { leave(); }

    inline bool enter(Cepoch &epoch, const tcs_t &thread_id, bool refresh, epoch_counter_t *safe_reclaim_epoch) {
      if (UNLIKELY(epoch_ != nullptr))
        return false;
      epoch_ = &epoch;
      thread_id_ = thread_id;
      now_epoch_ = epoch.enter(thread_id, refresh, safe_reclaim_epoch);
      return true;
    }

    inline Cepoch &epoch() const {
      dbg_assert(epoch_ != nullptr);
      return *epoch_;
    }

    inline const tcs_t &thread_id() const { return thread_id_; }

    inline Cepoch::epoch_counter_t now_epoch() const { return now_epoch_; }

    inline void leave() {
      if (LIKELY(epoch_ != nullptr)) {
        epoch_->leave(thread_id_);
        epoch_ = nullptr;
      }
    }
  };

  inline CautoEpochBlock enter_block(const tcs_t &thread_id, bool refresh, epoch_counter_t *safe_reclaim_epoch) {
    return CautoEpochBlock(*this, thread_id, refresh, safe_reclaim_epoch);
  }

  static inline bool safe_to_reclaim(epoch_counter_t test_epoch, epoch_counter_t safe_reclaim_epoch) {
    return safe_reclaim_epoch - test_epoch >= 0; // Dealing rewind.
  }
};

} // namespace zFast

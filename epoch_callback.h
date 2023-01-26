//
// Created by zzy on 2022/1/1.
//

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>

#include "utility/common_define.h"
#include "utility/config.h"
#include "utility/log.h"

namespace zFast {

/**
 * Epoch callback only for i64 epoch.
 * Working together with epoch framework.
 */
template <class ThreadContext, class CallbackContext> class CepochCallback final {
  NO_COPY_MOVE(CepochCallback);

public:
  using epoch_counter_t = int64_t;
  using callback_t = void (*)(ThreadContext &, CallbackContext &&, bool &help_drain);

private:
  struct epoch_action_t final {
    // FREE or LOCKED should larger than any normal epoch.
    static constexpr epoch_counter_t FREE = INT64_MAX;
    static constexpr epoch_counter_t LOCKED = INT64_MAX - 1;

    std::atomic<epoch_counter_t> epoch;
    callback_t callback;
    CallbackContext context;

    epoch_action_t() : epoch(FREE), callback(nullptr), context{} {}

    bool try_pop(ThreadContext &tc, epoch_counter_t expected_epoch, bool &help_drain) {
      auto bret = epoch.compare_exchange_strong(expected_epoch, LOCKED);
      if (LIKELY(bret)) {
        auto cb = callback;
        auto ctx(std::move(context)); // Move out.
        callback = nullptr;
        // Release the lock.
        epoch.store(FREE, std::memory_order_release);
        // Perform the action.
        cb(tc, std::move(ctx), help_drain);
      }
      return bret;
    }

    bool try_push(epoch_counter_t reclaim_epoch, callback_t new_callback, CallbackContext &&new_context,
                  std::atomic<size_t> &counter) {
      auto expected_epoch = FREE;
      auto bret = epoch.compare_exchange_strong(expected_epoch, LOCKED);
      if (LIKELY(bret)) {
        callback = new_callback;
        context = std::forward<CallbackContext>(new_context);
        ++counter; // increase counter in lock to prevent missing invoke in drain
        // Release the lock.
        epoch.store(reclaim_epoch, std::memory_order_release);
      }
      return bret;
    }

    bool try_swap(ThreadContext &tc, epoch_counter_t expected_epoch, epoch_counter_t prior_epoch,
                  callback_t new_callback, CallbackContext &&new_context, bool &help_drain) {
      auto bret = epoch.compare_exchange_strong(expected_epoch, LOCKED);
      if (LIKELY(bret)) {
        auto cb = callback;
        auto ctx(std::move(context)); // Move out.
        callback = new_callback;
        context = std::forward<CallbackContext>(new_context);
        // Release the lock.
        epoch.store(prior_epoch, std::memory_order_release);
        // Perform the action.
        cb(tc, std::move(ctx), help_drain);
      }
      return bret;
    }
  };

  // no need to align to cache line because these usually access together
  const size_t slot_number_;
  std::atomic<size_t> drain_count_;
  std::atomic<intptr_t> active_mode_counter_;
  std::unique_ptr<epoch_action_t[]> drain_list_;

public:
  explicit CepochCallback(size_t slot_number)
      : slot_number_(slot_number), drain_count_(0), active_mode_counter_(0),
        drain_list_(std::make_unique<epoch_action_t[]>(slot_number)) {}

  ~CepochCallback() {
    if (UNLIKELY(drain_count_.load(std::memory_order_acquire) != 0)) {
      LOG_ERROR(("Epoch callback exit with pending operations."));
      ::abort();
    }
  }

  // indicator from active drain when leave
  std::atomic<intptr_t> &get_active_mode_counter() { return active_mode_counter_; }

  // retry_drain only set when retry needed, so init to false before invoke
  void drain(ThreadContext &tc, epoch_counter_t safe_reclaim_epoch, bool &retry_drain) {
    dbg_assert(safe_reclaim_epoch > 0);
    if (drain_count_.load(std::memory_order_acquire) > 0) {
      for (size_t idx = 0; idx < slot_number_; ++idx) {
        auto &action = drain_list_[idx];
        auto trigger_epoch = action.epoch.load(std::memory_order_acquire);
        // FREE or LOCKED larger than any normal epoch.
        if (trigger_epoch <= safe_reclaim_epoch) {
          auto help_drain = false;
          if (action.try_pop(tc, trigger_epoch, help_drain)) {
            if (help_drain)
              retry_drain = true;
            // drain_count_ added within lock so when this got 0,
            // all inserted item before drain invoked have been cleared
            if (--drain_count_ == 0)
              return;
          }
        }
      }
    }
  }

  /**
   * It is recommended to invoke this outside the epoch to prevent infinite loop.
   * Retry with new safe reclaim epoch if return false.
   * help_drain only set when successful register, and it should be inited to false before invoke
   */
  bool register_callback(ThreadContext &tc, epoch_counter_t safe_reclaim_epoch, epoch_counter_t reclaim_epoch,
                         callback_t callback, CallbackContext &&context, bool &help_drain) {
    dbg_assert(safe_reclaim_epoch > 0);
    // Only not safe to reclaim and then register an async callback.
    dbg_assert(safe_reclaim_epoch < reclaim_epoch);
    for (size_t i = 0; i < slot_number_; ++i) {
      auto &action = drain_list_[i];
      auto trigger_epoch = action.epoch.load(std::memory_order_acquire);
      if (epoch_action_t::FREE == trigger_epoch) {
        if (action.try_push(reclaim_epoch, callback, std::forward<CallbackContext>(context), drain_count_))
          return true;
      } else if (trigger_epoch <= safe_reclaim_epoch) {
        if (action.try_swap(tc, trigger_epoch, reclaim_epoch, callback, std::forward<CallbackContext>(context),
                            help_drain))
          return true;
      }
    }
    return false;
  }

  /**
   * Debug code.
   */

  void dbg_visit(const std::function<void(epoch_counter_t, const CallbackContext &)> &visitor) {
    for (size_t i = 0; i < slot_number_; ++i) {
      auto &action = drain_list_[i];
      visitor(action.epoch.load(std::memory_order_acquire), action.context);
    }
  }
};

} // namespace zFast

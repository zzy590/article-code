//
// Created by zzy on 2021/12/9.
//

#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "utility/config.h"
#include "utility/log.h"
#include "utility/time.h"

#include "../infrastructure/epoch.h"

namespace zFast {

class CepochTest final {

private:
  static constexpr int test_items = 512;
  static constexpr int thread_number = 16;

  typedef Cepoch<64> CepochImpl;

  struct test_item_t final {
    std::atomic<CepochImpl::epoch_counter_t> epoch{0};
    volatile uint64_t flag = 0;
  };

  static tcs_t &tls() {
    static thread_local tcs_t tid = DEFAULT_TCS_VAL;
    return tid;
  }

  static void validate(CepochImpl &epoch, std::atomic<test_item_t *> *data) {
    auto block(epoch.enter_block(tls(), false, nullptr));

    test_item_t *ptrs[test_items];
    for (auto i = 0; i < test_items; ++i)
      ptrs[i] = data[i].load(std::memory_order_relaxed);
    for (auto i = 0; i < test_items; ++i) {
      if (ptrs[i] != nullptr && ptrs[i]->flag != i) {
        LOG_ERROR(("Error found. idx: {} val: {}", i, ptrs[i]->flag));
        throw std::runtime_error("Error found.");
      }
    }
  }

  static test_item_t *node_allocate(CepochImpl &epoch, bool refresh, std::atomic<test_item_t *> *free, uint64_t start,
                                    CepochImpl::epoch_counter_t &reclaim_ver) {
    CepochImpl::epoch_counter_t safe;
    auto block(epoch.enter_block(tls(), refresh, &safe));

    for (auto i = 0; i < test_items; ++i) {
      auto &slot = free[(start + i) % test_items];
      auto ptr = slot.load(std::memory_order_relaxed);
      if (nullptr == ptr)
        continue;
      auto item_epoch = ptr->epoch.load(std::memory_order_relaxed);
      if (item_epoch != CepochImpl::EPOCH_UNPROTECTED && CepochImpl::safe_to_reclaim(item_epoch, safe)) {
        // try detach.
        if (slot.compare_exchange_strong(ptr, nullptr)) {
          // Record clean ver.
          reclaim_ver = epoch.bump_epoch_for_reclaim();
          // Clear data should be safe.
          ptr->flag = -1;
          // Detach successfully and lock it epoch.
          if (!ptr->epoch.compare_exchange_strong(item_epoch, CepochImpl::EPOCH_UNPROTECTED)) {
            LOG_ERROR(("Bad detach race."));
            throw std::runtime_error("Bad detach race.");
          }
          return ptr;
        }
      }
    }
    return nullptr;
  }

  static void node_insert(CepochImpl &epoch, test_item_t *node, std::atomic<test_item_t *> *data, uint64_t start,
                          CepochImpl::epoch_counter_t reclaim_ver) {
    auto block(epoch.enter_block(tls(), false, nullptr));

    for (auto i = 0; i < test_items; ++i) {
      auto id = (start + i) % test_items;
      auto &slot = data[id];
      auto ptr = slot.load(std::memory_order_relaxed);
      if (ptr != nullptr)
        continue;
      node->flag = id;
      if (slot.compare_exchange_strong(ptr, node)) {
        // Insert successfully.
        if (CepochImpl::EPOCH_UNPROTECTED == reclaim_ver) {
          LOG_ERROR(("Bad reclaim epoch."));
          throw std::runtime_error("Bad reclaim epoch.");
        }
        // Set epoch to safe to detach from data.
        CepochImpl::epoch_counter_t lock_expect = CepochImpl::EPOCH_UNPROTECTED;
        if (!node->epoch.compare_exchange_strong(lock_expect, reclaim_ver)) {
          LOG_ERROR(("Impossible miss the lock."));
          throw std::runtime_error("Impossible miss the lock.");
        }
        return;
      }
    }

    // No slot?
    LOG_ERROR(("Impossible no slot for data."));
    throw std::runtime_error("Impossible no slot for data.");
  }

  static test_item_t *node_erase(CepochImpl &epoch, bool refresh, std::atomic<test_item_t *> *data, uint64_t start,
                                 CepochImpl::epoch_counter_t &reclaim_ver) {
    CepochImpl::epoch_counter_t safe;
    auto block(epoch.enter_block(tls(), refresh, &safe));

    for (auto i = 0; i < test_items; ++i) {
      auto &slot = data[(start + i) % test_items];
      auto ptr = slot.load(std::memory_order_relaxed);
      if (nullptr == ptr)
        continue;
      auto item_epoch = ptr->epoch.load(std::memory_order_relaxed);
      if (item_epoch != CepochImpl::EPOCH_UNPROTECTED && CepochImpl::safe_to_reclaim(item_epoch, safe)) {
        // try detach.
        if (slot.compare_exchange_strong(ptr, nullptr)) {
          // Record clean ver.
          reclaim_ver = epoch.bump_epoch_for_reclaim();
          // Data should keep it's original.
          // Detach successfully and lock it epoch.
          if (!ptr->epoch.compare_exchange_strong(item_epoch, CepochImpl::EPOCH_UNPROTECTED)) {
            LOG_ERROR(("Bad detach race."));
            throw std::runtime_error("Bad detach race.");
          }
          return ptr;
        }
      }
    }
    return nullptr;
  }

  static void node_free(CepochImpl &epoch, test_item_t *node, std::atomic<test_item_t *> *free, uint64_t start,
                        CepochImpl::epoch_counter_t reclaim_ver) {
    auto block(epoch.enter_block(tls(), false, nullptr));

    for (auto i = 0; i < test_items; ++i) {
      auto &slot = free[(start + i) % test_items];
      auto ptr = slot.load(std::memory_order_relaxed);
      if (ptr != nullptr)
        continue;
      if (slot.compare_exchange_strong(ptr, node)) {
        // Insert successfully.
        if (CepochImpl::EPOCH_UNPROTECTED == reclaim_ver) {
          LOG_ERROR(("Bad reclaim epoch."));
          throw std::runtime_error("Bad reclaim epoch.");
        }
        // Set epoch to safe to detach from data.
        CepochImpl::epoch_counter_t lock_expect = CepochImpl::EPOCH_UNPROTECTED;
        if (!node->epoch.compare_exchange_strong(lock_expect, reclaim_ver)) {
          LOG_ERROR(("Impossible miss the lock."));
          throw std::runtime_error("Impossible miss the lock.");
        }
        return;
      }
    }

    // No slot?
    LOG_ERROR(("Impossible no slot for data."));
    throw std::runtime_error("Impossible no slot for data.");
  }

public:
  static void debug_test() {
    CepochImpl epoch;

    epoch.register_epoch_thread(tls());

    CepochImpl::epoch_counter_t safe;
    auto e = epoch.enter(tls(), true, &safe);
    dbg_assert(2 == e);
    std::cout << "enter" << std::endl;
    dbg_assert(1 == safe);
    std::cout << "safe(1): " << safe << std::endl;
    auto reclaim = epoch.bump_epoch_for_reclaim();
    dbg_assert(2 == reclaim);
    std::cout << "reclaim(2): " << reclaim << std::endl;
    auto bret = CepochImpl::safe_to_reclaim(reclaim, safe);
    dbg_assert(false == bret);
    std::cout << "safe to reclaim(0:false): " << bret << std::endl;
    safe = epoch.get_safe_reclaim_epoch(true);
    dbg_assert(1 == safe);
    std::cout << "safe(1): " << safe << std::endl;
    bret = CepochImpl::safe_to_reclaim(reclaim, safe);
    dbg_assert(false == bret);
    std::cout << "safe to reclaim(0:false): " << bret << std::endl;
    epoch.leave(tls());
    std::cout << "leave" << std::endl;

    safe = epoch.get_safe_reclaim_epoch(true);
    dbg_assert(2 == safe);
    std::cout << "safe(2): " << safe << std::endl;
    bret = CepochImpl::safe_to_reclaim(reclaim, safe);
    dbg_assert(true == bret);
    std::cout << "safe to reclaim(1:true): " << bret << std::endl;

    epoch.unregister_epoch_thread(tls());
  }

  static void concurrent_test() {
    CepochImpl epoch;

    std::mt19937_64 rnd(0xCC);
    std::vector<std::unique_ptr<test_item_t>> items;
    items.reserve(test_items);
    for (auto i = 0; i < test_items; ++i)
      items.emplace_back((rnd() % 4 > 2) ? new test_item_t : nullptr);

    epoch.register_epoch_thread(tls());

    auto safe = epoch.get_safe_reclaim_epoch(true);

    std::atomic<test_item_t *> free[test_items];
    std::atomic<test_item_t *> data[test_items];

    // Init all to free.
    for (auto i = 0; i < test_items; ++i) {
      auto ptr = items[i].get();
      if (ptr != nullptr) {
        ptr->epoch.store(safe);
        ptr->flag = -1;
      }
      free[i].store(ptr);
      data[i].store(nullptr);
    }

    std::atomic<bool> exit{false};

    std::vector<std::thread> threads;
    threads.reserve(thread_number);
    for (auto thread_id = 0; thread_id < thread_number; ++thread_id) {
      threads.emplace_back([&epoch, thread_id, &free, &data, &exit]() {
        std::mt19937_64 rnd(thread_id);
        epoch.register_epoch_thread(tls());

        auto run_times = 0, move_times = 0;
        while (!exit.load(std::memory_order_acquire)) {
          ++run_times;

          // Validate first.
          validate(epoch, data);

          // Try to allocate one from free region.
          auto reclaim_ver = CepochImpl::EPOCH_UNPROTECTED;
          auto allocated = node_allocate(epoch, rnd() & 1, free, rnd(), reclaim_ver);
          // Add to data if got one free.
          if (allocated != nullptr) {
            node_insert(epoch, allocated, data, rnd(), reclaim_ver);
            ++move_times;
          }

          // Find one in data to free.
          reclaim_ver = CepochImpl::EPOCH_UNPROTECTED;
          allocated = node_erase(epoch, rnd() & 1, data, rnd(), reclaim_ver);
          // Insert to free.
          if (allocated != nullptr) {
            node_free(epoch, allocated, free, rnd(), reclaim_ver);
            ++move_times;
          }
        }

        LOG_INFO(("thread {} exit with run {} move {}", thread_id, run_times, move_times));

        epoch.unregister_epoch_thread(tls());
      });
    }

    auto start = utility::Ctime::steady_ms();

    std::this_thread::sleep_for(std::chrono::seconds(10));
    exit.store(true, std::memory_order_release);
    for (auto &t : threads)
      t.join();
    threads.clear();

    auto end = utility::Ctime::steady_ms();

    LOG_INFO(("Finish within {}s. Now safe epoch: {}", (end - start) / 1000.0f, epoch.get_safe_reclaim_epoch(true)));

    epoch.unregister_epoch_thread(tls());
  }
};

} // namespace zFast

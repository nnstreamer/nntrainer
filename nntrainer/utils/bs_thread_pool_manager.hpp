#ifndef THREAD_POOL_MANAGER_HPP
#define THREAD_POOL_MANAGER_HPP

#pragma once
#include "bs_thread_pool.h"

namespace nntrainer {
class ThreadPoolManager {
protected:
    static BS::thread_pool<> pool;
public:
    // Delete copy and move constructors and assignment operators
    ThreadPoolManager(const ThreadPoolManager&) = delete;
    ThreadPoolManager& operator=(const ThreadPoolManager&) = delete;
    ThreadPoolManager(ThreadPoolManager&&) = delete;
    ThreadPoolManager& operator=(ThreadPoolManager&&) = delete;

    // Static method to access the single instance
    static BS::thread_pool<> & getInstance() {
        return pool;
    }

private:
    ThreadPoolManager() = default;
    ~ThreadPoolManager() = default;
};
}

#endif // THREAD_POOL_MANAGER_HPP
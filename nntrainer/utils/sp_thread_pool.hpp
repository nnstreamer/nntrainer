/**
 * @brief A thread pool with work-stealing for parallel task execution.
 * 
 * Manages worker threads that process tasks from per-thread deques, allowing work-stealing for load balancing.
 * Supports submitting range-based tasks and dynamic thread count configuration.
 * All public methods are implemented as static. Call any methods by SP::ThreadPool::method_name
 * When you submit a task and the pool hasn't been initialized, the pool will automatically be initialized.
 * You can also use soft_boot() to forcefully initialize the pool before usage.
 */

#pragma once

#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>
#include <algorithm>

#ifdef _WIN32
    #include <Windows.h>
#else
    #include <pthread.h>
    #include <sched.h>
#endif

#include <iostream>

namespace SP {

class ThreadPool {
private:
    inline static std::unique_ptr<ThreadPool> instance = nullptr;
    inline static std::mutex instance_mutex;
    inline static size_t configured_threads = 0;
    inline static size_t default_chunk_multiplier = 4;

    std::vector<std::thread> workers;
    std::vector<std::deque<std::function<void()>>> deques;
    std::vector<std::mutex> deque_mutex;
    std::vector<std::atomic<size_t>> deque_sizes;
    std::atomic<size_t> global_queue_items{0};

    std::condition_variable cv;
    std::mutex cv_mutex;
    std::atomic<bool> stop_flag{false};
    std::atomic<size_t> threads_cap_global{0};
    std::atomic<bool> work_stealing_enabled{true};

    std::atomic<size_t> tasks_total{0};
    std::mutex idle_mutex;
    std::condition_variable idle_cv;
    std::atomic<size_t> rr_index{0};


    ThreadPool() {
        size_t n = get_thread_count();
        start(n);
    }
    explicit ThreadPool(size_t thread_count) {
        configured_threads = thread_count;
        start(thread_count);
    }
    
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    

    static ThreadPool& getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex);
        if (!instance) {
            instance.reset(new ThreadPool(get_thread_count()));
        }
        return *instance;
    }

    void start(size_t thread_count) {
        stop_flag = false;
        workers.clear();
        deques.clear();
        deque_sizes.clear();

        workers.reserve(thread_count);
        deques.resize(thread_count);
        deque_mutex = std::vector<std::mutex>(thread_count);
        deque_sizes = std::vector<std::atomic<size_t>>(thread_count);
        threads_cap_global.store(thread_count, std::memory_order_release);

        for (size_t i = 0; i < thread_count; ++i) {
            deque_sizes[i].store(0, std::memory_order_relaxed);
        }
        for (size_t i = 0; i < thread_count; ++i) {
            workers.emplace_back([this, i] { worker_loop(i); });
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(cv_mutex);
            stop_flag = true;
        }
        cv.notify_all();
        for (auto& t : workers) {
            if (t.joinable()) t.join();
        }
        workers.clear();
        deques.clear();
        deque_mutex.clear();
        deque_sizes.clear();
    }

    void worker_loop(size_t index) {
        while (true) {
            std::function<void()> task;

            size_t cap = threads_cap_global.load(std::memory_order_relaxed);
            if (index < cap) {
            // Try local work
                {
                    std::lock_guard<std::mutex> lk(deque_mutex[index]);
                    if (!deques[index].empty()) {
                        task = std::move(deques[index].front());
                        deques[index].pop_front();
                        deque_sizes[index].fetch_sub(1, std::memory_order_relaxed);
                        global_queue_items.fetch_sub(1, std::memory_order_relaxed);
                    }
                }
                if (task) {
                    task();
                    continue;
                }
                // Try steal from others
                if (work_stealing_enabled.load(std::memory_order_relaxed)) {
                    size_t best_i = index;
                    size_t best_sz = 0;
                    for (size_t i = 0; i < deque_sizes.size(); ++i) {
                        if (i == index) continue;
                        size_t sz = deque_sizes[i].load(std::memory_order_relaxed);
                        if (sz > best_sz) {
                            best_sz = sz;
                            best_i = i;
                        }
                    }


                    if (best_sz > 0 && best_i != index) {
                        std::lock_guard<std::mutex> lk(deque_mutex[best_i]);
                        if (!deques[best_i].empty()) {
                            task = std::move(deques[best_i].back());
                            deques[best_i].pop_back();
                            deque_sizes[best_i].fetch_sub(1, std::memory_order_relaxed);
                            global_queue_items.fetch_sub(1, std::memory_order_relaxed);
                        }
                    }
                    if (task) {
                        task();
                        continue;
                    }
                }
            }
            // Wait until new work arrives or stopping
            std::unique_lock<std::mutex> lock(cv_mutex);
            cv.wait(lock, [this, index]() {
                if (stop_flag) return true;
                if (index >= threads_cap_global.load(std::memory_order_relaxed)) return true;
                return global_queue_items.load(std::memory_order_relaxed) > 0;
            });
            if (stop_flag) break;
        }
    }

    template<typename F>
    std::future<void> submit_task_with_chunk_size_impl(size_t start, size_t end, const size_t chunk_size, F&& f) {
        size_t total = (end > start ? end - start : 0);
        size_t n = threads_cap_global.load(std::memory_order_relaxed);
        if (n <= 0) n = 1;
        size_t num_tasks = (total + chunk_size - 1) / chunk_size;
        size_t chunk_multiplier = (num_tasks + n - 1) / n;

        return submit_task_impl(start, end, chunk_multiplier, std::forward<F>(f));
    }

    template<typename F>
    std::future<void> submit_task_with_chunk_multiplier_impl(size_t start, size_t end, size_t chunk_multiplier, F&& f, size_t threads_cap = 0) {
        size_t n = threads_cap_global.load(std::memory_order_relaxed);
        if (n <= 0) n = 1;
        return submit_task_impl(start, end, chunk_multiplier * n, std::forward<F>(f), threads_cap);
    }

    /**
     * @brief Submits a range-based task for parallel execution.
     * 
     * This function divides the range [start, end) into chunks and distributes them across worker threads.
     * Each chunk processes a subset of the range by applying the callable `f` to every index in its subset.
     * The function returns a future that becomes ready when all chunks complete execution.
     * 
     * @tparam F Callable type (automatically deduced).
     * @param start Starting index of the range (inclusive).
     * @param end Ending index of the range (exclusive).
     * @param f Callable object to apply to each index in the range.
     * @return std::future<void> Future to track completion of all submitted tasks.
     * 
     * @details
     * - Chunk sizes are calculated to balance workload, with remainder distributed to earlier chunks.
     * - Tasks are enqueued to worker-specific deques for thread-safe execution.
     * - Synchronization is handled via a shared promise and atomic remaining-task counter.
     */
    template<typename F>
    std::future<void> submit_task_impl(size_t start, size_t end, size_t chunks_count, F&& f,
                                       size_t threads_cap = 0) {
        size_t total = (end > start ? end - start : 0);
        size_t old_cap = threads_cap_global.load(std::memory_order_relaxed);
        size_t n = threads_cap == 0 ? old_cap : threads_cap;
        if (n <= 0) n = 1;
        threads_cap_global.store(n, std::memory_order_release);
        // Controls task distribution to threads
        size_t tasks_per_threads = chunks_count / n;
        size_t number_of_threads_with_extra_tasks = chunks_count % n;
        size_t large_group = (tasks_per_threads + 1) * number_of_threads_with_extra_tasks;
        // Controls dividing the loop into tasks
        size_t base = chunks_count ? total / chunks_count : 0;
        size_t rem  = chunks_count ? total % chunks_count : 0;

        auto promise   = std::make_shared<std::promise<void>>();
        auto future    = promise->get_future();

        auto remaining = std::make_shared<std::atomic<size_t>>(0); // Number of remaining tasks
        // Count valid chunks
        size_t real_chunks = 0;
        for (size_t i = 0; i < chunks_count; ++i) {
            size_t s = start + i * base;
            size_t e = s + base + (i == chunks_count - 1 ? rem : 0);
            if (s < e) ++real_chunks;
        }
        remaining->store(real_chunks, std::memory_order_relaxed);
        tasks_total.fetch_add(1);


        for (size_t i = 0; i < chunks_count; ++i) {
            size_t s = start + i * base;
            size_t e = s + base + (i == chunks_count - 1 ? rem : 0);
            if (s >= e) continue;
            
            // Guarantees each threads gets contiguous tasks
            // size_t tid = i % n;
            size_t tid;
            if (i < large_group) tid = i / (tasks_per_threads + 1);
            else tid = number_of_threads_with_extra_tasks + (i - large_group) / tasks_per_threads;

            auto task = [this, s, e, tid, old_cap, f_copy = std::decay_t<F>(f), promise, remaining]() mutable {
                for (size_t j = s; j < e; ++j) {
                    if constexpr (std::is_invocable_v<decltype(f_copy), size_t, size_t>) {
                        f_copy(j, tid);
                    } else {
                        f_copy(j);
                    }
                }
                if (remaining->fetch_sub(1, std::memory_order_relaxed) == 1) {
                    promise->set_value();
                    if (tasks_total.fetch_sub(1, std::memory_order_relaxed) == 1) {
                        std::lock_guard<std::mutex> lk2(idle_mutex);
                        idle_cv.notify_all();
                    }
                    this->threads_cap_global.store(old_cap, std::memory_order_relaxed);
                }
            };
            {
                std::lock_guard<std::mutex> lk(deque_mutex[tid]);
                deques[tid].push_back(std::move(task));
                deque_sizes[tid].fetch_add(1, std::memory_order_relaxed);
            }
        }
        global_queue_items.fetch_add(real_chunks, std::memory_order_release);

        cv.notify_all();
        if (real_chunks == 0) promise->set_value();
        return future;
    }

    template<typename F>
    std::future<void> submit_task_impl(F&& f) {
        tasks_total.fetch_add(1, std::memory_order_relaxed);
        size_t n = threads_cap_global.load(std::memory_order_relaxed);
        if (n <= 0) n = 1;
        size_t tid = rr_index.fetch_add(1, std::memory_order_relaxed) % n;

        auto promise   = std::make_shared<std::promise<void>>();
        auto future    = promise->get_future();

        auto task = [this, f_copy = std::decay_t<F>(f), promise]() mutable {
            f_copy();
            promise->set_value();
            if (tasks_total.fetch_sub(1, std::memory_order_relaxed) == 1) {
                std::lock_guard<std::mutex> lk2(idle_mutex);
                idle_cv.notify_all();
            }
        };
        {
            std::lock_guard<std::mutex> lk(deque_mutex[tid]);
            deques[tid].push_back(std::move(task));
            deque_sizes[tid].fetch_add(1, std::memory_order_relaxed);
            global_queue_items.fetch_add(1, std::memory_order_relaxed);
        }
        cv.notify_one();
        return future;
    }

public:
    ~ThreadPool() {
        stop();
    }

    template<typename F>
    static std::future<void> submit_task(F&& f) {
        return getInstance().submit_task_impl(std::forward<F>(f));
    }
    
    template<typename F>
    static std::future<void> submit_task(size_t start, size_t end, F&& f) {
        return getInstance().submit_task_with_chunk_multiplier_impl(start, end, default_chunk_multiplier, std::forward<F>(f));
    }

    template<typename F>
    static std::future<void> submit_task(size_t start, size_t end, size_t chunk_count, F&& f) {
        return getInstance().submit_task_impl(start, end, chunk_count, std::forward<F>(f));
    }

    template<typename F>
    static std::future<void> submit_task_with_chunk_multiplier(size_t start, size_t end, size_t chunk_multiplier, F&& f) {
        return getInstance().submit_task_with_chunk_multiplier_impl(start, end, chunk_multiplier, std::forward<F>(f));
    }

    template<typename F>
    static std::future<void> submit_task_with_chunk_size(size_t start, size_t end, size_t chunk_size, F&& f) {
        return getInstance().submit_task_with_chunk_size_impl(start, end, chunk_size, std::forward<F>(f));
    }

    template<typename F>
    static std::future<void> submit_task_with_threads_cap(size_t start, size_t end, size_t threads_cap, F&& f) {
        return getInstance().submit_task_with_chunk_multiplier_impl(start, end, default_chunk_multiplier, std::forward<F>(f), threads_cap);
    }

    template<typename F>
    static std::future<void> submit_task_with_threads_cap(size_t start, size_t end, size_t threads_cap, size_t chunks_multiplier, F&& f) {
        return getInstance().submit_task_impl(start, end, chunks_multiplier, std::forward<F>(f), threads_cap);
    }

    static void wait_for_all() {
        auto &p = getInstance();
        std::unique_lock<std::mutex> lk(p.idle_mutex);
        p.idle_cv.wait(lk, [&]{ return p.tasks_total.load() == 0; });
    }

    static void set_thread_count(size_t thread_count) {
        std::lock_guard<std::mutex> lock(instance_mutex);
        configured_threads = thread_count > 0 ? thread_count : 1;
        if (instance) {
            instance->stop();
            instance->start(configured_threads);
        } else {
            instance.reset(new ThreadPool(configured_threads));
        }
    }
    
    static size_t get_thread_count() {
        if (configured_threads == 0) {
            size_t hc = std::thread::hardware_concurrency();
            configured_threads = hc > 0 ? hc : 1;
        }
        return configured_threads;
    }

    static void set_work_stealing(bool flag) {
        getInstance().work_stealing_enabled.store(flag, std::memory_order_relaxed);
    }

    static size_t get_threads_cap() {
        return getInstance().threads_cap_global.load(std::memory_order_relaxed);
    }

    static void set_threads_cap(size_t threads_cap) {
        getInstance().threads_cap_global.store(threads_cap, std::memory_order_release);
    }

    static void shutdown() {
        std::lock_guard<std::mutex> lock(instance_mutex);
        if (instance) {
            instance->stop();
            instance.reset();
        }
    }

    static void soft_boot() {
        std::lock_guard<std::mutex> lock(instance_mutex);
        if (!instance) {
            instance.reset(new ThreadPool(get_thread_count()));
        }
    }

    static void set_processor_affinity() {
        auto &pool = getInstance();
        size_t cpu_cnt = std::thread::hardware_concurrency();
        if (cpu_cnt == 0) cpu_cnt = 1;

        for (size_t i = 0; i < pool.workers.size(); ++i) {
            size_t cpu = i % cpu_cnt;
        
        #ifdef _WIN32
            DWORD_PTR mask = 1ULL << cpu;
            ::SetThreadAffinityMask(pool.workers[i].native_handle(), mask);
        #else
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu, &cpuset);
            pthread_setaffinity_np(pool.workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
        #endif
        }
    }
    
};

} // namespace SP

#ifndef THREAD_POOL_MANAGER_CPP
#define THREAD_POOL_MANAGER_CPP

#include "bs_thread_pool_manager.hpp"

namespace nntrainer{
    BS::thread_pool<> ThreadPoolManager::pool(std::thread::hardware_concurrency());
}

#endif // THREAD_POOL_MANAGER_CPP
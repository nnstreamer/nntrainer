// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file        unittest_nntrainer_task.cpp
 * @date        04 Nov 2022
 * @brief       Unit test utility for task
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jiho Chu <jiho.chu@samsung.com>
 * @bug         No known bugs
 */

#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <task.h>
#include <task_executor.h>
#include <util_func.h>

/**
 * @brief Mock class of Task Manager
 */
class MockTaskExecutor : public nntrainer::TaskExecutor {
public:
  /**
   * @brief MockTaskExecutor constructor
   */
  MockTaskExecutor() {
    ON_CALL(*this, handleWork)
      .WillByDefault([&](int id, nntrainer::Task::Work &work, void *data) {
        nntrainer::TaskExecutor::handleWork(id, work, data);
      });
    ON_CALL(*this, handleCompleteStatus)
      .WillByDefault([&](int id, const std::future_status status) {
        nntrainer::TaskExecutor::handleCompleteStatus(id, status);
      });
  }

  /**
   * @brief Mock method for worker
   */
  MOCK_METHOD(void, handleWork, (int, nntrainer::Task::Work &, void *),
              (override));

  /**
   * @brief Mock method for completeChecker
   */
  MOCK_METHOD(void, handleCompleteStatus, (int, const std::future_status),
              (override));
};

/**
 * @brief Test class for TaskExecutor
 */
class TaskExecutorTest : public ::testing::Test {
protected:
  void SetUp() override { executor = std::make_shared<MockTaskExecutor>(); }

  std::shared_ptr<MockTaskExecutor> executor;
};

TEST_F(TaskExecutorTest, create_01_p) { ASSERT_NE(executor, nullptr); }

TEST_F(TaskExecutorTest, task_create_01_p) {
  auto work = [](std::atomic_bool &running, void *data) { return 0; };

  int data = 0xAA;
  auto task = nntrainer::Task(work, static_cast<void *>(&data));

  EXPECT_EQ(task.getType(), nntrainer::Task::Type::SYNC);
  EXPECT_EQ(task.started(), false);
  EXPECT_EQ(task.done(), false);
  EXPECT_EQ(task.getData(), &data);
  EXPECT_EQ(*static_cast<int *>(task.getData()), 0xAA);
}

TEST_F(TaskExecutorTest, task_async_create_01_p) {
  auto work = [](std::atomic_bool &running, void *data) { return 0; };

  int data = 0xAA;
  auto task = nntrainer::TaskAsync<>(work, static_cast<void *>(&data));

  EXPECT_NO_THROW(task.setTimeout(100));
  EXPECT_EQ(task.getTimeout(), 100);

  task.setPriority(nntrainer::TaskAsync<>::HIGH);
  EXPECT_EQ(task.getPriority(), nntrainer::TaskAsync<>::HIGH);
}

TEST_F(TaskExecutorTest, run_01_p) {
  int user_data = 0xAA;
  auto work = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(running, true);
    EXPECT_EQ(data, &user_data);
    return 0;
  };

  auto task =
    std::make_shared<nntrainer::Task>(work, static_cast<void *>(&user_data));
  EXPECT_EQ(executor->run(task), 0);
}

TEST_F(TaskExecutorTest, run_aync_01_p) {
  EXPECT_CALL(*executor, handleWork).Times(1);
  EXPECT_CALL(*executor, handleCompleteStatus).Times(1);

  std::promise<void> p;
  auto f = p.get_future();

  int user_data = 0xAA;
  auto work = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(data, &user_data);
    return 0;
  };

  auto complete = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(id, 1);
    EXPECT_EQ(status, nntrainer::TaskExecutor::CompleteStatus::SUCCESS);
    p.set_value();
  };

  // Execute a asynchronous task
  auto task = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));
  task->setTimeout(1000);
  EXPECT_EQ(executor->run(task, complete), 1);

  // Wait until complete callback called
  f.wait();
}

TEST_F(TaskExecutorTest, run_aync_02_p) {
  EXPECT_CALL(*executor, handleWork).Times(3);
  EXPECT_CALL(*executor, handleCompleteStatus).Times(3);

  std::promise<void> p;
  auto f = p.get_future();
  int user_data = 0xAA;

  auto work1 = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(data, &user_data);
    return 0;
  };

  auto work2 = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(data, &user_data);
    return 0;
  };

  auto work3 = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(data, &user_data);
    return 0;
  };

  int complete_cnt = 0;
  auto complete = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::CompleteStatus::SUCCESS);
    complete_cnt++;
    if (complete_cnt >= 3)
      p.set_value();
  };

  // Execute three asynchronous tasks
  auto task1 = std::make_shared<nntrainer::TaskAsync<>>(
    work1, static_cast<void *>(&user_data));
  auto task2 = std::make_shared<nntrainer::TaskAsync<>>(
    work2, static_cast<void *>(&user_data));
  auto task3 = std::make_shared<nntrainer::TaskAsync<>>(
    work3, static_cast<void *>(&user_data));

  task1->setTimeout(1000);
  task2->setTimeout(1000);
  task3->setTimeout(1000);

  EXPECT_EQ(executor->run(task1, complete), 1);
  EXPECT_EQ(executor->run(task2, complete), 2);
  EXPECT_EQ(executor->run(task3, complete), 3);

  // Wait until complete callback called
  f.wait();
}

TEST_F(TaskExecutorTest, cancel_01_p) {
  EXPECT_CALL(*executor, handleWork).Times(1);
  EXPECT_CALL(*executor, handleCompleteStatus).Times(1);

  std::promise<void> p;
  auto f = p.get_future();
  int user_data = 0xAA;

  auto work = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(data, &user_data);

    int cnt = 0;
    while (running.load()) {
      if (cnt++ >= 10)
        break;
      usleep(100 * 1000);
    }

    /* check whether it is canceled */
    if (cnt < 10)
      return -1;

    return 0;
  };

  auto complete = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::CompleteStatus::FAIL_CANCEL);
    p.set_value();
  };

  // Execute a asynchronous task
  auto task = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));
  task->setTimeout(20000);

  ASSERT_EQ(executor->run(task, complete), 1);

  usleep(100 * 1000);

  // Cancel asynchronous task
  executor->cancel(1);

  // Wait until complete callback called
  f.wait();
}

TEST_F(TaskExecutorTest, cancel_all_01_p) {
  EXPECT_CALL(*executor, handleWork).Times(3);
  EXPECT_CALL(*executor, handleCompleteStatus).Times(3);

  std::promise<void> p;
  auto f = p.get_future();
  int user_data = 0xAA;

  auto work = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(data, &user_data);

    int cnt = 0;
    while (running.load()) {
      if (cnt++ >= 10)
        break;
      usleep(100 * 1000);
    }

    /* check whether it is canceled */
    if (cnt < 10)
      return -1;

    return 0;
  };

  int complete_cnt = 0;
  auto complete = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::CompleteStatus::FAIL_CANCEL);
    complete_cnt++;
    if (complete_cnt >= 3)
      p.set_value();
  };

  // Execute three asynchronous tasks
  auto task1 = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));
  auto task2 = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));
  auto task3 = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));
  task1->setTimeout(20000);
  task2->setTimeout(20000);
  task3->setTimeout(20000);

  ASSERT_EQ(executor->run(task1, complete), 1);
  ASSERT_EQ(executor->run(task2, complete), 2);
  ASSERT_EQ(executor->run(task3, complete), 3);

  usleep(100 * 1000);

  // Cancel all task
  executor->cancelAll();

  // Wait until complete callback called
  f.wait();
}

TEST_F(TaskExecutorTest, timeout_01_p) {
  EXPECT_CALL(*executor, handleWork).Times(1);
  EXPECT_CALL(*executor, handleCompleteStatus).Times(1);

  std::promise<void> p;
  auto f = p.get_future();
  int user_data = 0xAA;

  auto work = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(data, &user_data);

    int cnt = 0;
    while (running.load()) {
      if (cnt++ >= 10)
        break;
      usleep(100 * 1000);
    }

    /* check whether it is canceled */
    if (cnt < 10)
      return -1;

    return 0;
  };

  auto complete = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::CompleteStatus::FAIL_TIMEOUT);
    p.set_value();
  };

  // Execute a asynchronous task
  auto task = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));

  // timeout will be occurs
  task->setTimeout(100);

  ASSERT_EQ(executor->run(task, complete), 1);

  // Wait until complete callback called
  f.wait();
}

TEST_F(TaskExecutorTest, timeout_all_01_p) {
  EXPECT_CALL(*executor, handleWork).Times(3);
  EXPECT_CALL(*executor, handleCompleteStatus).Times(3);

  std::promise<void> p;
  auto f = p.get_future();
  int user_data = 0xAA;

  auto work = [&](std::atomic_bool &running, void *data) {
    EXPECT_EQ(data, &user_data);

    int cnt = 0;
    while (running.load()) {
      if (cnt++ >= 10)
        break;
      usleep(100 * 1000);
    }

    /* check whether it is canceled */
    if (cnt < 10)
      return -1;

    return 0;
  };

  int complete_cnt = 0;
  auto complete = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::CompleteStatus::FAIL_TIMEOUT);
    complete_cnt++;
    if (complete_cnt >= 3)
      p.set_value();
  };

  // Execute three asynchronous tasks
  auto task1 = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));
  auto task2 = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));
  auto task3 = std::make_shared<nntrainer::TaskAsync<>>(
    work, static_cast<void *>(&user_data));

  // timeout will be occurs
  task1->setTimeout(100);
  task2->setTimeout(200);
  task3->setTimeout(300);

  ASSERT_EQ(executor->run(task1, complete), 1);
  ASSERT_EQ(executor->run(task2, complete), 2);
  ASSERT_EQ(executor->run(task3, complete), 3);

  // Wait until complete callback called
  f.wait();
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to init gtest\n";
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run test.\n";
  }

  return result;
}

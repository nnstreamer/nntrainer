// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   unittest_nntrainer_profiler.cpp
 * @date   09 December 2020
 * @brief  Profiler Tester
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <sstream>

#include <profiler.h>

using namespace nntrainer::profile;

/**
 * @brief Mock class for profile listener test
 *
 */
class MockProfileListener : public ProfileListener {
public:
  MockProfileListener() :
    ProfileListener(),
    time_event_cnt(0),
    memory_alloc(0) {
    ON_CALL(*this, notify)
      .WillByDefault(
        [&](PROFILE_EVENT event, const std::shared_ptr<ProfileEventData> data) {
          switch (event) {
          case EVENT_TIME_START:
          case EVENT_TIME_END:
            time_event_cnt++;
            last = data->duration;
            break;
          case EVENT_MEM_ALLOC:
          case EVENT_MEM_DEALLOC:
            this->memory_alloc = data->alloc_total;
            break;
          default:
            break;
          }
        });
  }

  ~MockProfileListener() {}

  void reset(const int event, const std::string &str) override {
    time_event_cnt = 0;
    memory_alloc = 0;
  }

  virtual void report(std::ostream &out) const override {
    out << time_event_cnt << " " << memory_alloc;
  }

  const std::chrono::microseconds result(const int event) override {
    return last;
  };

  /**
   * @brief mock method for notify
   */
  MOCK_METHOD(void, notify,
              (PROFILE_EVENT event,
               const std::shared_ptr<ProfileEventData> data));

private:
  int time_event_cnt;             /**< time event count */
  size_t memory_alloc;            /**< allocated memory size */
  std::chrono::microseconds last; /** last event duration */
};

/**
 * @brief Test class for ProfileTest
 */
class ProfileTest : public ::testing::Test {
protected:
  void SetUp() override {
    listener = std::make_shared<MockProfileListener>();
    profiler = std::make_shared<Profiler>();
  }

  std::shared_ptr<MockProfileListener> listener;
  std::shared_ptr<Profiler> profiler;
};

TEST_F(ProfileTest, subscribe_01_n) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(0);

  EXPECT_THROW(profiler->subscribe(nullptr), std::invalid_argument);
}

TEST_F(ProfileTest, subscribe_02_n) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(0);

  EXPECT_THROW(profiler->subscribe(nullptr, {1}), std::invalid_argument);
}

TEST_F(ProfileTest, subscribe_03_p) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(0);

  EXPECT_NO_THROW(profiler->subscribe(listener));
}

TEST_F(ProfileTest, unsubscribe_01_p) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(0);

  EXPECT_NO_THROW(profiler->subscribe(listener));
  EXPECT_NO_THROW(profiler->unsubscribe(listener));
}

TEST_F(ProfileTest, notify_01_p) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(3);

  EXPECT_NO_THROW(profiler->subscribe(listener));

  auto data1 = std::make_shared<ProfileEventData>(
    1, 0, 0, "", std::chrono::microseconds{10});
  listener->notify(PROFILE_EVENT::EVENT_TIME_START, data1);

  auto data2 = std::make_shared<ProfileEventData>(
    1, 0, 0, "", std::chrono::microseconds{100});
  listener->notify(PROFILE_EVENT::EVENT_TIME_END, data2);

  auto data3 = std::make_shared<ProfileEventData>(
    1, 0, 0, "", std::chrono::microseconds{150});
  listener->notify(PROFILE_EVENT::EVENT_TIME_END, data3);

  auto result = listener->result(1);
  EXPECT_EQ(result, std::chrono::microseconds{150});

  std::stringstream ss;
  listener->report(ss);
  EXPECT_EQ(ss.str(), "3 0");
}

TEST_F(ProfileTest, start_01_n) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(0);

  EXPECT_THROW(profiler->start(1), std::invalid_argument);
}

TEST_F(ProfileTest, end_01_n) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(0);

  EXPECT_THROW(profiler->end(1), std::invalid_argument);
}

TEST_F(ProfileTest, timeTest_01_p) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(2);

  int nn_forward = profiler->registerTimeItem("nn_forward");

  EXPECT_NO_THROW(profiler->subscribe(listener, {nn_forward}));

  profiler->start(nn_forward);
  profiler->end(nn_forward);

  std::stringstream ss;
  listener->report(ss);
  EXPECT_EQ(ss.str(), "2 0");
}

TEST_F(ProfileTest, timeTest_02_n) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(0);

  int nn_forward = profiler->registerTimeItem("nn_forward");

  EXPECT_THROW(profiler->start(2), std::invalid_argument);
  EXPECT_THROW(profiler->start(2), std::invalid_argument);

  std::stringstream ss;
  listener->report(ss);
  EXPECT_EQ(ss.str(), "0 0");
}

TEST_F(ProfileTest, timeTest_01_n) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_)).Times(2);

  int nn_forward = profiler->registerTimeItem("nn_forward");

  EXPECT_NO_THROW(profiler->subscribe(listener, {nn_forward}));

  profiler->start(nn_forward);
  profiler->end(nn_forward);

  std::stringstream ss;
  listener->report(ss);
  EXPECT_EQ(ss.str(), "2 0");
}

TEST_F(ProfileTest, memoryTestAlloc_01_p) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_))
    .Times(testing::AtLeast(1));
  profiler->subscribe(listener);

  profiler->alloc((void *)0x1, (size_t)10, "");

  std::stringstream ss;
  listener->report(ss);
  EXPECT_EQ(ss.str(), "0 10");
}

TEST_F(ProfileTest, memoryTestDealloc_01_p) {
  EXPECT_CALL(*listener, notify(testing::_, testing::_))
    .Times(testing::AtLeast(1));
  profiler->subscribe(listener);

  profiler->alloc((void *)0x1, (size_t)10, "");
  profiler->dealloc((void *)0x1);

  /// Check if notified on all profiler listener
  std::stringstream ss;
  listener->report(ss);
  EXPECT_EQ(ss.str(), "0 0");
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

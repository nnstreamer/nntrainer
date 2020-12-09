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
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <sstream>

#include <profiler.h>

using namespace nntrainer::profile;

class MockProfileListener : public ProfileListener {
public:
  MockProfileListener() : hit(false){};

  ~MockProfileListener(){};

  void onNotify(const EVENT event,
                const std::chrono::milliseconds &value) override {
    hit = true;
  }

  void reset(const EVENT event) override { hit = false; }

  void report(std::ostream &out) const override {
    out << (hit ? "hit" : "no hit");
  }

  const std::chrono::milliseconds result(const EVENT event) override {
    return std::chrono::milliseconds();
  };

private:
  bool hit; /**< check if onNotify has been called */
};

TEST(Profiler, profilePositiveTests_p) {
  /// Initiate and run Profile
  Profiler prof;

  /// subscribe listener
  std::unique_ptr<ProfileListener> all_listener(new MockProfileListener);
  std::unique_ptr<ProfileListener> event_listener(new MockProfileListener);

  prof.subscribe(all_listener.get());
  prof.subscribe(event_listener.get(), {EVENT::NN_FORWARD});

  /// measure
  prof.start(EVENT::NN_FORWARD);
  prof.end(EVENT::NN_FORWARD);

  /// Check if notified on all profiler listener
  {
    std::stringstream ss;
    all_listener->report(ss);
    EXPECT_EQ(ss.str(), "hit");
  }

  /// Check if notified on event profiler listener
  {
    std::stringstream ss;
    event_listener->report(ss);
    EXPECT_EQ(ss.str(), "hit");
  }

  /// measure that are not registered for event listener
  all_listener->reset(EVENT::NN_FORWARD);
  event_listener->reset(EVENT::NN_FORWARD);
  prof.start(EVENT::TEMP);
  prof.end(EVENT::TEMP);

  /// Check if notified on event profiler does not hit
  {
    std::stringstream ss;
    event_listener->report(ss);
    EXPECT_EQ(ss.str(), "no hit");
  }
}

TEST(Profiler, cannotStartTwice_n) {
  Profiler prof;
  prof.start(EVENT::NN_FORWARD);

  EXPECT_THROW(prof.start(EVENT::NN_FORWARD), std::invalid_argument);
}

TEST(Profiler, cannotEndTwice_n) {
  Profiler prof;
  prof.end(EVENT::NN_FORWARD);

  EXPECT_THROW(prof.end(EVENT::NN_FORWARD), std::invalid_argument);
}

TEST(Profiler, subscribeNullListener_n) {
  Profiler prof;

  prof.subscribe(nullptr);
}

TEST(Profiler, subscribeNullListener2_n) {
  Profiler prof;

  prof.subscribe(nullptr, {EVENT::NN_FORWARD});
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

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
  MockProfileListener(Profiler *profiler, std::vector<int> events = {}) :
    ProfileListener(profiler, events),
    hit(false){};

  ~MockProfileListener(){};

  void onNotify(const int event,
                const std::chrono::milliseconds &value) override {
    hit = true;
  }

  void reset(const int event) override { hit = false; }

  void report(std::ostream &out) const override {
    out << (hit ? "hit" : "no hit");
  }

  const std::chrono::milliseconds result(const int event) override {
    return std::chrono::milliseconds();
  };

private:
  bool hit; /**< check if onNotify has been called */
};

TEST(GenericProfileListener, listenerBasicScenario_p) {

  GenericProfileListener listener{nullptr};

  /// assuming library-side code is calling onNotify
  listener.onNotify(EVENT::NN_FORWARD, std::chrono::milliseconds{10});
  listener.onNotify(EVENT::NN_FORWARD, std::chrono::milliseconds{100});
  listener.onNotify(EVENT::NN_FORWARD, std::chrono::milliseconds{50});

  auto result = listener.result(EVENT::NN_FORWARD);
  EXPECT_EQ(result, std::chrono::milliseconds{50});

  std::cout << listener;
}

TEST(GenericProfileListener, noResultButQueryResult_n) {
  GenericProfileListener listener{nullptr};
  EXPECT_THROW(listener.result(EVENT::NN_FORWARD), std::invalid_argument);
}

TEST(Profiler, profilePositiveTests_p) {
  /// Initiate and run Profile
  Profiler prof;

  /// subscribe listener
  std::unique_ptr<ProfileListener> all_listener(new MockProfileListener{&prof});
  std::unique_ptr<ProfileListener> event_listener(
    new MockProfileListener{&prof, {EVENT::NN_FORWARD}});

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

  /// unsubscribe event_listener
  event_listener->reset(EVENT::NN_FORWARD);
  prof.unsubscribe(event_listener.get());
  prof.start(EVENT::NN_FORWARD);
  prof.end(EVENT::NN_FORWARD);

  {
    std::stringstream ss;
    event_listener->report(ss);
    EXPECT_EQ(ss.str(), "no hit");
  }
}

TEST(Profiler, cannotStartTwice_n) {
  Profiler prof;
  prof.start(EVENT::NN_FORWARD);

#ifdef DEBUG
  EXPECT_THROW(prof.start(EVENT::NN_FORWARD), std::invalid_argument);
#endif
}

TEST(Profiler, endThatNeverStarted_n) {
  Profiler prof;
#ifdef DEBUG
  EXPECT_THROW(prof.end(EVENT::NN_FORWARD), std::invalid_argument);
#endif
}

TEST(Profiler, cannotEndTwice_n) {
  Profiler prof;
  prof.start(EVENT::NN_FORWARD);
  prof.end(EVENT::NN_FORWARD);

#ifdef DEBUG
  EXPECT_THROW(prof.end(EVENT::NN_FORWARD), std::invalid_argument);
#endif
}

TEST(Profiler, subscribeNullListener_n) {
  Profiler prof;

  EXPECT_THROW(prof.subscribe(nullptr), std::invalid_argument);
}

TEST(Profiler, subscribeNullListener2_n) {
  Profiler prof;

  EXPECT_THROW(prof.subscribe(nullptr, {EVENT::NN_FORWARD}),
               std::invalid_argument);
}

TEST(Profiler, subscribeAllTwice_n) {
  Profiler prof;
  std::unique_ptr<ProfileListener> all_listener(new MockProfileListener{&prof});

  EXPECT_THROW(prof.subscribe(all_listener.get()), std::invalid_argument);
}

TEST(Profiler, subscribePartialTwice_n) {
  Profiler prof;
  std::unique_ptr<ProfileListener> all_listener(new MockProfileListener{&prof});

  EXPECT_THROW(prof.subscribe(all_listener.get(), {EVENT::NN_FORWARD}),
               std::invalid_argument);
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

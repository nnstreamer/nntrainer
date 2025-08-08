/**
 * Copyright (C) 2025 Daekyoung Jung <daekyoung.jung@gmail.com>
 *
 * @file        generate_random_vector.cpp
 * @date        08 August 2025
 * @brief       Declare the function to generate random vector for test
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Daekyoung Jung <daekyoung.jung@gmail.com>
 * @bug         No known bugs
 */

#include <vector>

template <typename T, bool random_init = false>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_nntrainer_tensor_pool.cpp
 * @date 20 August 2021
 * @brief Tensor Pool Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <basic_planner.h>
#include <tensor_pool.h>

constexpr unsigned int MEM_BYTES = 128;
constexpr unsigned int MEM_QUANT = 100;
constexpr unsigned int INTERVAL_SIZE = 5;

/**
 * @brief creation and destruction
 */
TEST(TensorPool, create_destroy) { EXPECT_NO_THROW(nntrainer::TensorPool()); }

/**
 * @brief request empty tensor
 */
TEST(TensorPool, request_mem_01_n) {
  nntrainer::TensorPool pool;

  EXPECT_THROW(pool.requestTensor(nntrainer::TensorDim(), {},
                                  nntrainer::TensorLifespan::ZERO_LIFESPAN,
                                  "abc"),
               std::invalid_argument);
}

/**
 * @brief request empty name
 */
TEST(TensorPool, request_mem_02_n) {
  nntrainer::TensorPool pool;

  EXPECT_THROW(pool.requestTensor(nntrainer::TensorDim({1}), {},
                                  nntrainer::TensorLifespan::ZERO_LIFESPAN, ""),
               std::invalid_argument);
}

/**
 * @brief request tensor
 */
TEST(TensorPool, request_mem_03_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t;

  EXPECT_NO_THROW(
    t = pool.requestTensor(nntrainer::TensorDim({1}), {},
                           nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc"));
  EXPECT_NE(t, nullptr);
  EXPECT_FALSE(t->isAllocated());
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, request_mem_04_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.requestTensor(nntrainer::TensorDim({1}), {},
                                     nntrainer::TensorLifespan::ZERO_LIFESPAN,
                                     "abc"));

  EXPECT_THROW(pool.requestTensor(nntrainer::TensorDim({2}), {},
                                  nntrainer::TensorLifespan::ZERO_LIFESPAN,
                                  "abc"),
               std::invalid_argument);
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, request_mem_05_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1, *t2;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({1}), {},
                            nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc"));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(t2 = pool.requestPrerequestedTensor(
                    nntrainer::TensorDim({1}), {},
                    nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc1", "abc"));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, request_mem_06_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.requestTensor(nntrainer::TensorDim({1}), {},
                                     nntrainer::TensorLifespan::ZERO_LIFESPAN,
                                     "abc"));

  EXPECT_THROW(pool.requestPrerequestedTensor(
                 nntrainer::TensorDim({2}), {},
                 nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc1", "abc"),
               std::invalid_argument);
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, request_mem_07_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.requestTensor(nntrainer::TensorDim({1}), {},
                                     nntrainer::TensorLifespan::ZERO_LIFESPAN,
                                     "abc"));

  EXPECT_THROW(pool.requestPrerequestedTensor(
                 nntrainer::TensorDim({1}), {},
                 nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc1", "not_exist"),
               std::invalid_argument);
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, request_mem_08_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1, *t2;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({1}), {},
                            nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc"));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(t2 = pool.requestPrerequestedTensor(
                    nntrainer::TensorDim({1}), {},
                    nntrainer::TensorLifespan::MAX_LIFESPAN, "abc1", "abc"));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(t2 = pool.requestPrerequestedTensor(
                    nntrainer::TensorDim({1}), {},
                    nntrainer::TensorLifespan::MAX_LIFESPAN, "abc2", "abc"));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, request_mem_09_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.requestTensor(nntrainer::TensorDim({1}), {},
                                     nntrainer::TensorLifespan::ZERO_LIFESPAN,
                                     "abc"));

  EXPECT_THROW(pool.requestPrerequestedTensor(
                 nntrainer::TensorDim({1}), {},
                 nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc", "abc"),
               std::invalid_argument);
}

/**
 * @brief set batch
 */
TEST(TensorPool, set_batch_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({1}), {},
                            nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc"));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_EQ(t1->batch(), 1u);
  EXPECT_NO_THROW(pool.setBatchSize("abc", 10));
  EXPECT_EQ(t1->batch(), 10u);
}

/**
 * @brief set batch
 */
TEST(TensorPool, set_batch_02_n) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({1}), {},
                            nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc"));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_THROW(pool.setBatchSize("not_exist", 10), std::invalid_argument);
  EXPECT_EQ(t1->batch(), 1u);
}

/**
 * @brief zero size pool as no usage
 */
TEST(TensorPool, finalize_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1, *t2;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({1}), {},
                            nntrainer::TensorLifespan::ZERO_LIFESPAN, "abc1"));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.requestTensor(nntrainer::TensorDim({1}), {},
                            nntrainer::TensorLifespan::MAX_LIFESPAN, "abc2"));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), 0u);

  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief max lifespan tensors
 */
TEST(TensorPool, finalize_02_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1, *t2;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({1}), {0},
                            nntrainer::TensorLifespan::MAX_LIFESPAN, "abc1"));
  EXPECT_NE(t1, nullptr);

  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.requestTensor(nntrainer::TensorDim({1}), {1},
                            nntrainer::TensorLifespan::MAX_LIFESPAN, "abc2"));
  EXPECT_NE(t2, nullptr);

  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes() + t2->bytes());

  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief max lifespan tensors
 */
TEST(TensorPool, finalize_03_p) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
}

/**
 * @brief allocate
 */
TEST(TensorPool, allocate_deallocate_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1, *t2;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({1}), {0},
                            nntrainer::TensorLifespan::MAX_LIFESPAN, "abc1"));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.requestTensor(nntrainer::TensorDim({1}), {1},
                            nntrainer::TensorLifespan::MAX_LIFESPAN, "abc2"));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));

  EXPECT_NO_THROW(pool.allocate());
  EXPECT_TRUE(t1->isAllocated());
  EXPECT_TRUE(t2->isAllocated());

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief allocate
 */
TEST(TensorPool, allocate_deallocate_02_n) {
  nntrainer::TensorPool pool;

  EXPECT_THROW(pool.allocate(), std::runtime_error);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, allocate_deallocate_03_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1, *t2, *t3;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({1}), {0},
                            nntrainer::TensorLifespan::MAX_LIFESPAN, "abc"));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(t2 = pool.requestPrerequestedTensor(
                    nntrainer::TensorDim({1}), {1},
                    nntrainer::TensorLifespan::MAX_LIFESPAN, "abc1", "abc"));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(t3 = pool.requestPrerequestedTensor(
                    nntrainer::TensorDim({1}), {0},
                    nntrainer::TensorLifespan::MAX_LIFESPAN, "abc2", "abc"));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NE(t1, t3);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));

  EXPECT_NO_THROW(pool.allocate());
  EXPECT_TRUE(t1->isAllocated());
  EXPECT_TRUE(t2->isAllocated());
  EXPECT_TRUE(t3->isAllocated());

  EXPECT_EQ(t1->getData(), t2->getData());
  EXPECT_EQ(t1->getData(), t3->getData());

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
  EXPECT_FALSE(t3->isAllocated());
}

/**
 * @brief validate memory full overlap
 */
TEST(TensorPool, validate_memory) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.requestTensor(nntrainer::TensorDim({100}), {0},
                            nntrainer::TensorLifespan::MAX_LIFESPAN, "abc1"));

  EXPECT_NO_THROW(
    t2 = pool.requestTensor(nntrainer::TensorDim({100}), {1},
                            nntrainer::TensorLifespan::MAX_LIFESPAN, "abc2"));

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_NO_THROW(pool.allocate());

  nntrainer::Tensor g1 = nntrainer::Tensor(nntrainer::TensorDim({100}));
  g1.setRandNormal();
  nntrainer::Tensor g2 = nntrainer::Tensor(nntrainer::TensorDim({100}));
  g2.setRandNormal();

  t1->copy(g1);
  t2->copy(g2);

  EXPECT_EQ(*t1, g1);
  EXPECT_EQ(*t2, g2);

  EXPECT_NO_THROW(pool.deallocate());
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

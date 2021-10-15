// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file compiler_test_util.cpp
 * @date 09 October 2021
 * @brief NNTrainer graph compiler related common functions
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <compiler_test_util.h>

#include <app_context.h>
#include <nntrainer_test_util.h>

#include <gtest/gtest.h>

const std::string compilerPathResolver(const std::string &path) {
  return getResPath(path, {"test", "test_models", "models"});
}

void graphEqual(const nntrainer::GraphRepresentation &lhs,
                const nntrainer::GraphRepresentation &rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());

  auto is_node_equal = [](const nntrainer::LayerNode &l,
                          const nntrainer::LayerNode &r) {
    nntrainer::Exporter lhs_export;
    nntrainer::Exporter rhs_export;

    l.exportTo(lhs_export, nntrainer::ExportMethods::METHOD_STRINGVECTOR);
    r.exportTo(rhs_export, nntrainer::ExportMethods::METHOD_STRINGVECTOR);

    /*** fixme, there is one caveat that order matters in this form */
    EXPECT_EQ(
      *lhs_export.getResult<nntrainer::ExportMethods::METHOD_STRINGVECTOR>(),
      *rhs_export.getResult<nntrainer::ExportMethods::METHOD_STRINGVECTOR>());
  };

  if (lhs.size() == rhs.size()) {
    auto lhs_iter = lhs.cbegin();
    auto rhs_iter = rhs.cbegin();
    for (; lhs_iter != lhs.cend(), rhs_iter != rhs.cend();
         lhs_iter++, rhs_iter++) {
      auto lhs = *lhs_iter;
      auto rhs = *rhs_iter;
      is_node_equal(*lhs.get(), *rhs.get());
    }
  }
}

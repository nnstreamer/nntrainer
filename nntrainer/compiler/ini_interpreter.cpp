// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   ini_interpreter.cpp
 * @date   02 April 2021
 * @brief  NNTrainer Ini Interpreter (partly moved from model_loader.c)
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <ini_interpreter.h>

#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

static constexpr const char *FUNC_TAG = "[IniInterpreter] ";

static constexpr const char *UNKNOWN_STR = "UNKNOWN";
static constexpr const char *NONE_STR = "NONE";
static constexpr const char *MODEL_STR = "model";
static constexpr const char *DATASET_STR = "dataset";
static constexpr const char *OPTIMIZER_STR = "optimizer";

namespace nntrainer {

namespace {

/** @todo:
 *  1. deprecate tag dispatching along with #1072
 *  2. deprecate getMergeableGraph (extendGraph should accept graph itself)
 */
class PlainLayer {};    /**< Plain Layer tag */
class BackboneLayer {}; /**< Backbone Layer tag */

template <typename T>
static std::shared_ptr<Layer> section2layer(dictionary *ini,
                                            const std::string &sec_name) {
  /// NYI!
  return nullptr;
}

static std::shared_ptr<GraphRepresentation>
section2graph(dictionary *ini, const std::string &sec_name) {
  /// NYI!
  return nullptr;
}

bool graphSupported(const std::string &backbone_name) {
  /// NYI!
  return true;
}

std::vector<std::shared_ptr<Layer>>
getMergeableGraph(std::shared_ptr<const GraphRepresentation>) {
  return {};
};

} // namespace

void IniGraphInterpreter::serialize(
  std::shared_ptr<const GraphRepresentation> representation,
  const std::string &out) {}

std::shared_ptr<GraphRepresentation>
IniGraphInterpreter::deserialize(const std::string &in) {

  NNTR_THROW_IF(in.empty(), std::invalid_argument)
    << FUNC_TAG << "given in file is empty";

  NNTR_THROW_IF(!isFileExist(in), std::invalid_argument)
    << FUNC_TAG << "given ini file does not exist";

  dictionary *ini = iniparser_load(in.c_str());
  NNTR_THROW_IF(!ini, std::runtime_error) << "loading ini failed";

  auto freedict = [ini] { iniparser_freedict(ini); };

  /** Get number of sections in the file */
  int num_ini_sec = iniparser_getnsec(ini);
  NNTR_THROW_IF_CLEANUP(num_ini_sec < 0, std::invalid_argument, freedict)
    << FUNC_TAG << "invalid number of sections.";

  std::shared_ptr<GraphRepresentation> graph =
    std::make_shared<GraphRepresentation>();

  try {
    for (int idx = 0; idx < num_ini_sec; ++idx) {
      auto sec_name_ = iniparser_getsecname(ini, idx);
      NNTR_THROW_IF_CLEANUP(!sec_name_, std::runtime_error, freedict)
        << "parsing a section name returned error, filename: " << in
        << "idx: " << idx;

      ml_logd("probing section_name: %s", sec_name_);
      std::string sec_name(sec_name_);

      if (istrequal(sec_name, MODEL_STR) || istrequal(sec_name, DATASET_STR) ||
          istrequal(sec_name, OPTIMIZER_STR)) {
        /// dedicated sections so skip
        continue;
      }
      /** Parse all the layers defined as sections in order */
      std::shared_ptr<Layer> layer;

      /**
       * If this section is a backbone, load backbone section from this
       * @note The order of backbones in the ini file defines the order on the
       * backbones in the model graph
       */
      const char *backbone_path =
        iniparser_getstring(ini, (sec_name + ":Backbone").c_str(), UNKNOWN_STR);

      const std::string &backbone = pathResolver(backbone_path);
      if (graphSupported(backbone)) {
        auto g = section2graph(ini, sec_name);

        /// @todo: deprecate this. We should extend graph from a graph
        auto tmp = getMergeableGraph(g);
        graph->extendGraph(tmp, sec_name);
        continue;
      }

      if (backbone_path == UNKNOWN_STR) {
        layer = section2layer<PlainLayer>(ini, sec_name);
      } else {
        /// @todo deprecate this as well with #1072
        layer = section2layer<BackboneLayer>(ini, sec_name);
      }

      graph->addLayer(layer);
    }
  } catch (...) {
    /** clean up and rethrow */
    freedict();
    throw;
  }

  freedict();
  return graph;
}

} // namespace nntrainer

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

#include <sstream>
#include <vector>

#include <ini_wrapper.h>
#include <layer.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
// #include <time_dist.h>
#include <util_func.h>

#if defined(ENABLE_NNSTREAMER_BACKBONE)
#include <nnstreamer_layer.h>
#endif

#if defined(ENABLE_TFLITE_BACKBONE)
#include <tflite_layer.h>
#endif

static constexpr const char *FUNC_TAG = "[IniInterpreter] ";

static constexpr const char *UNKNOWN_STR = "UNKNOWN";
static constexpr const char *NONE_STR = "NONE";
static constexpr const char *MODEL_STR = "model";
static constexpr const char *DATASET_STR = "dataset";
static constexpr const char *TRAINSET_STR = "train_set";
static constexpr const char *VALIDSET_STR = "valid_set";
static constexpr const char *TESTSET_STR = "test_set";
static constexpr const char *OPTIMIZER_STR = "optimizer";
static constexpr const char *LRSCHED_STR = "LearningRateScheduler";

namespace nntrainer {

IniGraphInterpreter::IniGraphInterpreter(
  const AppContext &app_context_,
  std::function<const std::string(const std::string &)> pathResolver_) :
  app_context(app_context_), pathResolver(pathResolver_) {}

IniGraphInterpreter::~IniGraphInterpreter() {}

namespace {

/** @todo:
 *  1. deprecate tag dispatching along with #1072
 *  2. deprecate getMergeableGraph (extendGraph should accept graph itself)
 *
 * @brief Plain Layer tag
 */
class PlainLayer {};

/**
 * @brief Backbone Layer tag
 */
class BackboneLayer {};

/**
 * @brief convert section to list of string based properties
 *
 * @param ini ini handler
 * @param sec_name section name
 * @param pathResolver path resolver of ini. If relative path is given to ini,
 * it is prioritized to be interpreted relative to ini file. So this resolver is
 * required @see ModelLoader::resolvePath for detail.
 * @return std::vector<std::string> list of properties
 */
std::vector<std::string> section2properties(
  dictionary *ini, const std::string &sec_name,
  std::function<const std::string(std::string)> &pathResolver) {
  int num_entries = iniparser_getsecnkeys(ini, sec_name.c_str());
  NNTR_THROW_IF(num_entries < 1, std::invalid_argument)
    << "there are no entries in the layer section: " << sec_name;
  ml_logd("number of entries for %s: %d", sec_name.c_str(), num_entries);

  std::unique_ptr<const char *[]> key_refs(new const char *[num_entries]);
  NNTR_THROW_IF(iniparser_getseckeys(ini, sec_name.c_str(), key_refs.get()) ==
                  nullptr,
                std::invalid_argument)
    << "failed to fetch keys for the section: " << sec_name;

  std::vector<std::string> properties;
  properties.reserve(num_entries - 1);

  for (int i = 0; i < num_entries; ++i) {
    std::string key(key_refs[i]);
    std::string prop_key = key.substr(key.find(":") + 1);

    if (istrequal(prop_key, "type") || istrequal(prop_key, "backbone")) {
      continue;
    }

    std::string value = iniparser_getstring(ini, key_refs[i], UNKNOWN_STR);
    NNTR_THROW_IF(value == UNKNOWN_STR || value.empty(), std::invalid_argument)
      << "parsing property failed key: " << key << " value: " << value;

    /// @todo systematically manage those props
    if (istrequal(prop_key, "model_path")) {
      value = pathResolver(value);
    }

    ml_logd("parsed properties: %s=%s", prop_key.c_str(), value.c_str());
    properties.push_back(prop_key + "=" + value);
  }

  properties.push_back("name=" + sec_name);

  return properties;
}

/**
 * @brief convert section to a layer object (later it might be layer node)
 *
 * @tparam T tag
 * @param ini dictionary * ini
 * @param sec_name section name
 * @param ac app context to search for the layer
 * @param backbone_file full path of backbone file (defaulted to be omitted)
 * @param pathResolver if path is given and need resolving, called here
 * @return std::shared_ptr<Layer> return layer object
 */
template <typename T>
std::shared_ptr<LayerNode>
section2layer(dictionary *ini, const std::string &sec_name,
              const AppContext &ac, const std::string &backbone_file,
              std::function<const std::string(std::string)> &pathResolver) {
  throw std::invalid_argument("supported only with a tag for now");
}

template <>
std::shared_ptr<LayerNode> section2layer<PlainLayer>(
  dictionary *ini, const std::string &sec_name, const AppContext &ac,
  const std::string &backbone_file,
  std::function<const std::string(std::string)> &pathResolver) {

  const std::string &layer_type =
    iniparser_getstring(ini, (sec_name + ":Type").c_str(), UNKNOWN_STR);
  NNTR_THROW_IF(layer_type == UNKNOWN_STR, std::invalid_argument)
    << FUNC_TAG << "section type is invalid for section name: " << sec_name;

  auto properties = section2properties(ini, sec_name, pathResolver);

  auto layer = createLayerNode(ac.createObject<Layer>(layer_type), properties);
  return layer;
}

template <>
std::shared_ptr<LayerNode> section2layer<BackboneLayer>(
  dictionary *ini, const std::string &sec_name, const AppContext &ac,
  const std::string &backbone_file,
  std::function<const std::string(std::string)> &pathResolver) {
  std::string type;

#if defined(ENABLE_NNSTREAMER_BACKBONE)
  type = NNStreamerLayer::type;
#endif

/** TfLite has higher priority */
#if defined(ENABLE_TFLITE_BACKBONE)
  if (endswith(backbone_file, ".tflite")) {
    type = TfLiteLayer::type;
  }
#endif

  NNTR_THROW_IF(type.empty(), std::invalid_argument)
    << FUNC_TAG
    << "This nntrainer does not support external section: " << sec_name
    << " backbone: " << backbone_file;

  auto properties = section2properties(ini, sec_name, pathResolver);
  properties.push_back("model_path=" + backbone_file);

  auto layer = createLayerNode(type, properties);

  return layer;
}

/**
 * @brief check if graph is supported
 *
 * @param backbone_name name of the backbone
 * @retval true if the file extension is supported to make a graph
 * @retval false if the file extension is not supported
 */
static bool graphSupported(const std::string &backbone_name) {
  return endswith(backbone_name, ".ini");
}

/**
 * @brief Get the Mergeable Graph object
 * @note This function is commented to prohibit using this but left intact to be
referenced
 * @param graph currently, extendGraph accepts
 * std::vector<std::shared_ptr<Layer>>, so return in this format
 * @param ini ini to parse property
 * @param sec_name section name
 * @return std::vector<std::shared_ptr<Layer>> mergeable graph
 */
// std::vector<std::shared_ptr<LayerNode>>
// getMergeableGraph(const GraphRepresentation& graph,
//                   dictionary *ini, const std::string &sec_name) {
//   std::string input_layer =
//     iniparser_getstring(ini, (sec_name + ":InputLayer").c_str(), "");
//   std::string output_layer =
//     iniparser_getstring(ini, (sec_name + ":OutputLayer").c_str(), "");

//   NNTR_THROW_IF(g.empty(), std::invalid_argument)
//     << FUNC_TAG << "backbone graph is empty";

//   /** Wait for #361 Load the backbone from its saved file */
//   // bool preload =
//   //   iniparser_getboolean(ini, (sec_name + ":Preload").c_str(), true);

// const std::string &trainable =
//   iniparser_getstring(ini, (sec_name + ":Trainable").c_str(), "true");

//   for (auto &lnode : g) {
//     lnode->setProperty({"trainable=" + trainable});
//     /** TODO #361: this needs update in model file to be of dictionary format
//     */
//     // if (preload) {
//     //   layer->weight_initializer = Initializer::FILE_INITIALIZER;
//     //   layer->bias_initializer = Initializer::FILE_INITIALIZER;
//     //   layer->initializer_file = backbone.save_path;
//     // }
//   }

// // set input dimension for the first layer in the graph

// /** FIXME :the layers is not the actual model_graph. It is just the vector of
//  * layers generated by Model Loader. so graph[0] is still valid. Also we need
//  * to consider that the first layer of ini might be the first of layers. Need
//  * to change by compiling the backbone before using here. */
// std::string input_shape =
//   iniparser_getstring(ini, (sec_name + ":Input_Shape").c_str(), "");
// if (!input_shape.empty()) {
//   g[0]->setProperty({"input_shape=" + input_shape});
// }

// std::string input_layers =
//   iniparser_getstring(ini, (sec_name + ":Input_Layers").c_str(), "");
// if (!input_layers.empty()) {
//   g[0]->setProperty({"input_layers=" + input_layers});
// }

// return g;
// };

} // namespace

void IniGraphInterpreter::serialize(const GraphRepresentation &representation,
                                    const std::string &out) {

  std::vector<IniSection> sections;
  for (auto iter = representation.cbegin(); iter != representation.cend();
       iter++) {
    const auto &ln = *iter;

    IniSection s = IniSection::FromExportable(ln->getName(), *ln);
    s.setEntry("type", ln->getType());

    sections.push_back(s);
  }

  auto ini = IniWrapper(out, sections);
  ini.save_ini(out);
}

GraphRepresentation IniGraphInterpreter::deserialize(const std::string &in) {
  NNTR_THROW_IF(in.empty(), std::invalid_argument)
    << FUNC_TAG << "given in file is empty";

  NNTR_THROW_IF(!isFileExist(in), std::invalid_argument)
    << FUNC_TAG << "given ini file does not exist, file_path: " << in;

  dictionary *ini = iniparser_load(in.c_str());
  NNTR_THROW_IF(!ini, std::runtime_error) << "loading ini failed";

  auto freedict = [ini] { iniparser_freedict(ini); };

  /** Get number of sections in the file */
  int num_ini_sec = iniparser_getnsec(ini);
  NNTR_THROW_IF_CLEANUP(num_ini_sec < 0, std::invalid_argument, freedict)
    << FUNC_TAG << "invalid number of sections.";

  GraphRepresentation graph;

  try {
    ml_logi("==========================parsing ini...");
    ml_logi("not-allowed property for the layer throws error");
    ml_logi("valid property with invalid value throws error as well");
    for (int idx = 0; idx < num_ini_sec; ++idx) {
      auto sec_name_ = iniparser_getsecname(ini, idx);
      NNTR_THROW_IF_CLEANUP(!sec_name_, std::runtime_error, freedict)
        << FUNC_TAG << "parsing a section name returned error, filename: " << in
        << "idx: " << idx;

      std::string sec_name(sec_name_);

      if (istrequal(sec_name, MODEL_STR) || istrequal(sec_name, DATASET_STR) ||
          istrequal(sec_name, TRAINSET_STR) ||
          istrequal(sec_name, VALIDSET_STR) ||
          istrequal(sec_name, TESTSET_STR) ||
          istrequal(sec_name, OPTIMIZER_STR) ||
          istrequal(sec_name, LRSCHED_STR)) {
        /// dedicated sections so skip
        continue;
      }
      /** Parse all the layers defined as sections in order */
      ml_logd("probing section_name: %s", sec_name_);
      std::shared_ptr<LayerNode> layer;

      /**
       * If this section is a backbone, load backbone section from this
       * @note The order of backbones in the ini file defines the order on the
       * backbones in the model graph
       */
      const char *backbone_path =
        iniparser_getstring(ini, (sec_name + ":Backbone").c_str(), UNKNOWN_STR);

      NNTR_THROW_IF(backbone_path == nullptr, std::invalid_argument)
        << FUNC_TAG << "backbone path is null";

      const std::string &backbone = pathResolver(backbone_path);
      if (graphSupported(backbone)) {
        /// @todo: this will be changed to a general way to add a graph
        auto bg = this->deserialize(backbone);
        const std::string &trainable =
          iniparser_getstring(ini, (sec_name + ":Trainable").c_str(), "true");

        for (auto &node : bg) {
          node->setProperty({"trainable=" + trainable});
        }
        graph.insert(graph.end(), bg.begin(), bg.end());
        continue;
      }

      if (std::strcmp(backbone_path, UNKNOWN_STR) == 0) {
        layer = section2layer<PlainLayer>(ini, sec_name, app_context, "",
                                          pathResolver);
      } else {
        layer = section2layer<BackboneLayer>(ini, sec_name, app_context,
                                             backbone, pathResolver);
      }

      graph.push_back(layer);
    }
    /// @todo if graph Model Type is of recurrent_wrapper, parse model and
    /// realize before return
  } catch (...) {
    /** clean up and rethrow */
    freedict();
    throw;
  }

  freedict();
  return graph;
}

} // namespace nntrainer

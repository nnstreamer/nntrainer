/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	neuralnet.cpp
 * @date	04 December 2019
 * @brief	This is Neural Network Class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include <databuffer.h>
#include <ini_interpreter.h>
#include <ini_wrapper.h>
#include <model_loader.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <optimizer_context.h>
#include <profiler.h>
#include <util_func.h>

/**
 * @brief Internal enum values for nntrainer to summarize model accuracy & loss
 */
#define ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS 101
#define ML_TRAIN_SUMMARY_MODEL_VALID_LOSS 102
#define ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY 103

namespace nntrainer {

NeuralNetwork::NeuralNetwork(AppContext app_context_, bool in_place_opt) :
  model_props(props::LossType(), {}, {}),
  model_flex_props(props::Epochs(), props::TrainingBatchSize(),
                   props::SavePath(), props::ContinueTrain(),
                   props::SaveBestPath()),
  load_path(std::string()),
  epoch_idx(0),
  iter(0),
  loss(0.0f),
  data_buffers({nullptr, nullptr, nullptr}),
  initialized(false),
  compiled(false),
  loadedFromConfig(false),
  app_context(app_context_),
  in_place_optimization(in_place_opt) {}

int NeuralNetwork::loadFromConfig(const std::string &config) {
  if (loadedFromConfig == true) {
    ml_loge("cannnot do loadFromConfig twice");
    return ML_ERROR_INVALID_PARAMETER;
  }

  ModelLoader loader(app_context);
  NeuralNetwork tempNet(*this);
  int status = loader.loadFromConfig(config, tempNet);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  tempNet.loadedFromConfig = true;
  swap(tempNet, *this);

  return ML_ERROR_NONE;
}

void NeuralNetwork::setProperty(const std::vector<std::string> &values) {
  auto left_props = loadProperties(values, model_props);
  setTrainConfig(left_props);
}

void NeuralNetwork::setTrainConfig(const std::vector<std::string> &values) {
  auto left_props = loadProperties(values, model_flex_props);
  NNTR_THROW_IF(left_props.size(), std::invalid_argument)
    << "Model has unparsed properties, size: " << left_props.size()
    << " of first element: " << left_props.front();
}

int NeuralNetwork::compile() {
  std::string loss_type = std::get<props::LossType>(model_props).empty()
                            ? std::string()
                            : std::get<props::LossType>(model_props);

  int status = model_graph.compile(loss_type);
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

int NeuralNetwork::initialize() {
  int status = ML_ERROR_NONE;

  if (initialized) {
    ml_loge("Error: Initializing the model again");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (!compiled) {
    ml_loge("Error: Need to compile first");
    return ML_ERROR_NOT_SUPPORTED;
  }

  unsigned int n_layers = (unsigned int)model_graph.size();

  ml_logd("initializing neural network, layer size: %d", n_layers);

  model_graph.setBatchSize(
    std::get<props::TrainingBatchSize>(model_flex_props));

  auto &input_layer_prop =
    std::get<std::vector<props::InputLayer>>(model_props);
  auto &label_layer_prop =
    std::get<std::vector<props::LabelLayer>>(model_props);

  std::vector<std::string> input_layers;
  std::vector<std::string> label_layers;

  if (!input_layer_prop.empty()) {
    input_layers = std::vector<std::string>(input_layer_prop.begin(),
                                            input_layer_prop.end());
  }
  if (!label_layer_prop.empty()) {
    label_layers = std::vector<std::string>(label_layer_prop.begin(),
                                            label_layer_prop.end());
  }

  status = model_graph.initialize(input_layers, label_layers);
  NN_RETURN_STATUS();

  // initialize optimizer and related variables
  if (opt) {
    /** TODO: update request of optimizer to be of same format as
     * Layer::requestTensor */
    opt->finalize();
    std::function<std::vector<TensorDim>(const TensorDim &)> cb =
      [this](const TensorDim &dim) {
        return opt->getOptimizerVariableDim(dim);
      };
    model_graph.requestOptimizerVariable(cb, true);
  }

  // Allocate weights
  model_graph.allocateWeights();

  if (in_place_optimization) {
    model_graph.inPlaceOptimize();
  }

  initialized = true;

  if (!load_path.empty()) {
    load(load_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  }

  return status;
}

/**
 * @brief     free layers
 */
NeuralNetwork::~NeuralNetwork() = default;

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(bool training) {
  return model_graph.forwarding(training);
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(sharedConstTensors input,
                                             sharedConstTensors label,
                                             bool training) {
  auto current_batch = model_graph.getBatchSize();
  NNTR_THROW_IF(input[0]->batch() != current_batch ||
                  (!label.empty() && label[0]->batch() != current_batch),
                std::logic_error)
    << "Error: mismatch in batchsize for data and model."
    << " input_batch: " << input[0]->batch()
    << " label_batch: " << label[0]->batch()
    << " target_batch: " << current_batch;

  model_graph.setInputsLabels(input, label);

  return forwarding(training);
}

void NeuralNetwork::backwarding(std::shared_ptr<LayerNode> node, int iteration,
                                bool calc_derivative) {
  /**
   * Do not change this order:
   * 1. calcGradient
   * 2. calcDerivative
   * 3. applyGradientsOnLastAccess
   */

  bool apply_gradient = true;
  /** If gradient optimization mode, then calculate gradient first */
  if (dynamic_training_opt.isGradientMode())
    node->calcGradient();

  /**
   * If optimization off, or gradient must be applied, then this will be true
   * @todo This apply gradient should be passed to the each weight and later be
   * queried when updating gradient at once. (after moving apply_gradient out of
   * this function)
   *
   */
  // auto &layer = node->getObject();
  // apply_gradient = dynamic_training_opt.checkIfApply(
  //   layer->getWeightsRef(), layer->net_input[0], layer->net_hidden[0], opt,
  //   iteration);

  /** If gradient must be applied and its not gradient mode, calculate gradient
   */
  if (!dynamic_training_opt.isGradientMode() && apply_gradient)
    node->calcGradient();

  if (calc_derivative)
    node->calcDerivative();

  if (apply_gradient) {
    /// Apply gradient only at the end of the last shared weight access
    model_graph.applyGradientsOnLastAccess(
      node.get(), [iteration, opt_ = opt.get()](Weight &w) {
        w.calcRegularizationGradient();
        RunOptimizerContext opt_context(&w, iteration);
        opt_->applyGradient(opt_context);
      });
  }
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
void NeuralNetwork::backwarding(int iteration) {
  /**
   * last layer backwarding is run out of this loop
   */
  auto iter_begin = model_graph.getBackwardingBeginIter();
  auto iter_end = model_graph.getBackwardingEndIter();

  /// there is no layer to train, so backwarding is essentially noop
  if (iter_begin == iter_end) {
    return;
  }

  auto const &lptr_begin = (*iter_begin);

  if (lptr_begin->requireLabel() == false)
    throw std::runtime_error(
      "Error: last layer does not accept label, we can't train");

  auto iter = iter_begin;
  for (; iter != iter_end - 1; iter++) {
    backwarding(*iter, iteration, true);
  }

  /**
   * The last trainable layer need not calculate the derivatives
   */
#ifdef ENABLE_TEST
  backwarding(*iter, iteration, true);
#else
  backwarding(*iter, iteration, false);
#endif
}

void NeuralNetwork::save(const std::string &file_path,
                         ml::train::ModelFormat format) {
  NNTR_THROW_IF(!initialized, std::runtime_error)
    << "Cannot save model if not initialized yet, path: " << file_path
    << " format: " << static_cast<unsigned>(format);

  /// @todo this switch case should be delegating the function call only. It's
  /// not delegating for now as required logics are managable for now.
  switch (format) {
  case ml::train::ModelFormat::MODEL_FORMAT_BIN: {
    auto model_file = checkedOpenStream<std::ofstream>(
      file_path, std::ios::out | std::ios::binary);
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      (*iter)->save(model_file);
    }
    model_file.write((char *)&epoch_idx, sizeof(epoch_idx));
    model_file.write((char *)&iter, sizeof(iter));
    model_file.close();
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_INI:
    saveModelIni(file_path);
    break;

  case ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN: {
    auto old_save_path = std::get<props::SavePath>(model_flex_props);
    auto bin_file_name =
      file_path.substr(0, file_path.find_last_of('.')) + ".bin";

    std::get<props::SavePath>(model_flex_props).set(bin_file_name);
    save(file_path, ml::train::ModelFormat::MODEL_FORMAT_INI);
    save(bin_file_name, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    std::get<props::SavePath>(model_flex_props) = old_save_path;
    break;
  }
  default:
    throw nntrainer::exception::not_supported(
      "saving with given format is not supported yet");
  }
}

void NeuralNetwork::load(const std::string &file_path,
                         ml::train::ModelFormat format) {
  /// @todo this switch case should be delegating the function call only. It's
  /// not delegating for now as required logics are managable for now.
  switch (format) {
  case ml::train::ModelFormat::MODEL_FORMAT_BIN: {
    NNTR_THROW_IF(!initialized, std::runtime_error)
      << "Cannot load if not initialized yet, path: " << file_path
      << " format: " << static_cast<unsigned>(format);

    auto model_file = checkedOpenStream<std::ifstream>(
      file_path, std::ios::in | std::ios::binary);
    std::cerr << file_path << '\n';
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      (*iter)->read(model_file);
    }

    try {
      /// this is assuming that the failure is allowed at the end of the file
      /// read. so, after this line, additional read shouldn't be called
      checkedRead(model_file, (char *)&epoch_idx, sizeof(epoch_idx),
                  "[NeuralNetwork::readModel] failed to read epoch_idx");
      checkedRead(model_file, (char *)&iter, sizeof(iter),
                  "[NeuralNetwork::readModel] failed to read iteration");
    } catch (...) {
      std::cerr << "failed to read epoch idx, proceeding with default index\n";
    }

    ml_logi("read modelfile: %s", file_path.c_str());
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN: {
    int ret = loadFromConfig(file_path);
    throw_status(ret);
    auto &save_path = std::get<props::SavePath>(model_flex_props);
    if (!save_path.empty()) {
      checkedOpenStream<std::ifstream>(save_path,
                                       std::ios::in | std::ios::binary);
      load_path = save_path;
    }
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_INI: {
    int ret = loadFromConfig(file_path);
    throw_status(ret);
    break;
  }
  default:
    throw nntrainer::exception::not_supported(
      "loading with given format is not supported yet");
  }
}

float NeuralNetwork::getLoss() {
  loss = 0.0f;

  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    loss += (*iter)->getLoss();
  }
  return loss;
}

void NeuralNetwork::setLoss(float l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    model_props = from.model_props;
    model_flex_props = from.model_flex_props;
    loss = from.loss;
    opt = from.opt;

    model_graph.copy(from.model_graph);
  }
  return *this;
}

void NeuralNetwork::saveModelIni(const std::string &file_path) {
  NNTR_THROW_IF(isFileExist(file_path), std::invalid_argument)
    << "There is already a file, overriding to the exisiting file is not "
       "permitted, path: "
    << file_path;

  std::vector<IniSection> sections;

  IniSection model_section = IniSection::FromExportable("model", *this);
  model_section.setEntry("type", "NeuralNetwork");
  sections.push_back(model_section);

  auto add_section_if_any = [&sections](const std::string &section_name,
                                        auto obj_ptr, auto pred) {
    if (pred(obj_ptr)) {
      IniSection s = IniSection::FromExportable(section_name, *obj_ptr);
      s.setEntry("type", obj_ptr->getType());
      sections.push_back(s);
    }
  };

  add_section_if_any("optimizer", opt,
                     [](const auto &obj) { return static_cast<bool>(obj); });

  auto &[train_buffer, valid_buffer, test_buffer] = data_buffers;
  auto data_buffer_valid = [](const auto &buffer) {
    return buffer && buffer->isSerializable(ExportMethods::METHOD_STRINGVECTOR);
  };

  add_section_if_any("train_set", train_buffer, data_buffer_valid);
  add_section_if_any("valid_set", valid_buffer, data_buffer_valid);
  add_section_if_any("test_set", test_buffer, data_buffer_valid);

  IniWrapper wrapper("model_saver", sections);
  wrapper.save_ini(file_path);

  IniGraphInterpreter interpreter;
  interpreter.serialize(model_graph, file_path);
}

bool NeuralNetwork::validateInput(sharedConstTensors X) {

  auto const &first_layer_node = model_graph.getSortedLayerNode(0);
  auto input_dim = first_layer_node->getInputDimensions();
  if (X.size() != input_dim.size()) {
    ml_loge("Error: provided number of inputs %d, required %d", (int)X.size(),
            (int)input_dim.size());
    return false;
  }

  for (unsigned int dim = 0; dim < input_dim.size(); dim++) {
    if (input_dim[dim] != X[dim]->getDim()) {
      ml_loge("Error: provided input shape does not match required shape");
      std::stringstream ss;
      ss << X[dim]->getDim();
      ml_loge("Provided tensor summary : %s", ss.str().c_str());

      ss.str(std::string());
      ss << input_dim[dim];
      ml_loge("Required tensor summary : %s", ss.str().c_str());
      return false;
    }
  }

  return true;
}

sharedConstTensors NeuralNetwork::inference(sharedConstTensors X,
                                            bool free_mem) {
  return inference(X, {}, free_mem);
}

sharedConstTensors NeuralNetwork::inference(sharedConstTensors X,
                                            sharedConstTensors label,
                                            bool free_mem) {
  if (model_graph.getBatchSize() != X[0]->batch()) {
    model_graph.setBatchSize(X[0]->batch());
  }

  sharedConstTensors out;
  if (!validateInput(X))
    throw std::invalid_argument("Input validation failed.");

  allocate(ExecutionMode::INFERENCE);

  START_PROFILE(profile::NN_FORWARD);
  out = forwarding(X, label, false);
  END_PROFILE(profile::NN_FORWARD);

  if (free_mem)
    /**
     * Free the memory needed for training before exiting.
     * Note that this does not free the weights for the model.
     * Weights of the model will be freed when the model is destroyed.
     */
    model_graph.deallocateTensors(false);

  /** Clear the set inputs and labels */
  model_graph.setInputsLabels({}, {});

  return out;
}

std::vector<float *> NeuralNetwork::inference(unsigned int batch_size,
                                              std::vector<float *> &input,
                                              std::vector<float *> &label) {
  sharedConstTensors input_tensors, output_tensors;
  auto in_dim = getInputDimension();

  input_tensors.reserve(input.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    in_dim[idx].batch(batch_size);
    input_tensors.emplace_back(MAKE_SHARED_TENSOR(Tensor::Map(
      input[idx], in_dim[idx].getDataLen() * sizeof(float), in_dim[idx], 0)));
  }

  if (!label.empty()) {
    sharedConstTensors label_tensors;
    auto label_dim = getOutputDimension();
    label_tensors.reserve(label.size());
    for (unsigned int idx = 0; idx < label_dim.size(); idx++) {
      label_dim[idx].batch(batch_size);
      label_tensors.emplace_back(MAKE_SHARED_TENSOR(
        Tensor::Map(label[idx], label_dim[idx].getDataLen() * sizeof(float),
                    label_dim[idx], 0)));
    }
    output_tensors = inference(input_tensors, label_tensors, false);
  } else {
    output_tensors = inference(input_tensors, false);
  }

  std::vector<float *> output;
  output.reserve(output_tensors.size());

  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    output.push_back(out_t.getData());
  }

  return output;
}

int NeuralNetwork::setDataset(const DatasetModeType &mode,
                              std::shared_ptr<ml::train::Dataset> dataset) {
  return setDataBuffer(mode, std::static_pointer_cast<DataBuffer>(dataset));
}

int NeuralNetwork::allocate(ExecutionMode mode) {
  model_graph.deallocateTensors();
  model_graph.allocateTensors(mode);

  return ML_ERROR_NONE;
}

int NeuralNetwork::deallocate() {
  model_graph.deallocateTensors(true);

  return ML_ERROR_NONE;
}

int NeuralNetwork::train(const std::vector<std::string> &values) {
  int status = ML_ERROR_NONE;

  if (data_buffers[static_cast<int>(DatasetModeType::MODE_TRAIN)] == nullptr) {
    ml_loge("Cannot initialize the model without the train data buffer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!opt) {
    ml_loge("Cannot train network without optimizer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  setTrainConfig(values);

  /** set batch size just before training */
  model_graph.setBatchSize(
    std::get<props::TrainingBatchSize>(model_flex_props));

  status = allocate(ExecutionMode::TRAIN);
  NN_RETURN_STATUS();

  status = train_run();
  NN_RETURN_STATUS();

  /**
   * Free the memory needed for training before exiting.
   * Note that this does not free the weights for the model.
   * Weights of the model will be freed when the model is destroyed.
   */
  model_graph.deallocateTensors(false);
  return status;
}

/**
 * @brief     Run NeuralNetwork train with callback function by user
 */
int NeuralNetwork::train_run() {
  int status = ML_ERROR_NONE;

  if (!std::get<props::ContinueTrain>(model_flex_props)) {
    epoch_idx = 0;
    iter = 0;
  }
  auto const &first_layer_node = model_graph.getSortedLayerNode(0);
  auto const &last_layer_node =
    model_graph.getSortedLayerNode(model_graph.size() - 1);

  auto batch_size = std::get<props::TrainingBatchSize>(model_flex_props);

  auto &output = last_layer_node->getOutput(0);

  /** @todo use model_graph.getInputDimensions() and getOutputDimensions() */
  auto in_dims = first_layer_node->getInputDimensions();
  auto label_dims = last_layer_node->getOutputDimensions();

  auto &[train_buffer, valid_buffer, test_buffer] = data_buffers;

  if (train_buffer == nullptr) {
    ml_loge("[NeuralNetworks] there is no train dataset!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /**
   * @brief run a single epoch with given callback, @a auto is used instead of
   * std::function for performance measure
   * @param buffer buffer to run
   * @param shuffle whether to shuffle or not
   * @param on_iteration_fetch function that will recieve reference to stat,
   * buffer which will be called every time data is fetched and set
   * @param on_epoch_end function that will recieve reference to stat,
   * buffer which will be called on the epoch end
   */
  auto run_epoch = [this, &in_dims, &label_dims, &output, batch_size](
                     DataBuffer *buffer, bool shuffle,
                     auto &&on_iteration_fetch, auto &&on_iteration_update_stat,
                     auto &&on_epoch_end) {
    /// @todo managing metrics must be handled here as well!! for now it is
    /// handled in individual callbacks
    RunStats stat;
    std::future<std::shared_ptr<IterationQueue>> future_iq =
      buffer->startFetchWorker(in_dims, label_dims, shuffle);
    while (true) {
      ScopedView<Iteration> iter_view = buffer->fetch();
      if (iter_view.isEmpty()) {
        break;
      }
      auto &iteration = iter_view.get();
      if (iteration.batch() != batch_size) {
        /// @todo support partial batch
        continue;
      }

      auto const &labels = iteration.getLabelsRef();
      auto const &inputs = iteration.getInputsRef();
      model_graph.setInputsLabels(inputs, labels);

      on_iteration_fetch(stat, *buffer);
      on_iteration_update_stat(stat, {output}, labels);
    }
    future_iq.get();
    on_epoch_end(stat, *buffer);

    if (stat.num_iterations == 0) {
      throw std::runtime_error("No data came while buffer ran");
    }

    return stat;
  };

  auto train_for_iteration = [this](RunStats &stat, DataBuffer &buffer) {
    forwarding(true);
    backwarding(iter++);

    std::cout << "#" << epoch_idx << "/" << getEpochs();
    ml_logi("# %d / %d", epoch_idx, getEpochs());
    auto loss = getLoss();
    buffer.displayProgress(stat.num_iterations, loss);
  };

  auto update_train_stat = [this](RunStats &stat,
                                  const std::vector<Tensor> &outputs,
                                  const std::vector<Tensor> &labels) {
    stat.loss += getLoss();
    stat.num_iterations++;
  };

  auto train_epoch_end = [this](RunStats &stat, DataBuffer &buffer) {
    stat.loss /= static_cast<float>(stat.num_iterations);
    auto &save_path = std::get<props::SavePath>(model_flex_props);
    if (!save_path.empty()) {
      save(save_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    }

    std::cout << "#" << epoch_idx << "/" << getEpochs()
              << " - Training Loss: " << stat.loss;
    ml_logi("# %d / %d - Training Loss: %f", epoch_idx, getEpochs(), stat.loss);
  };

  auto eval_for_iteration = [this, batch_size](RunStats &stat,
                                               DataBuffer &buffer) {
    forwarding(false);
  };

  auto update_eval_stat = [this, batch_size, &update_train_stat](
                            RunStats &stat, const std::vector<Tensor> &outputs,
                            const std::vector<Tensor> &labels) {
    auto model_out = outputs[0].argmax();
    auto label_out = labels[0].argmax();

    for (unsigned int b = 0; b < batch_size; b++) {
      if (model_out[b] == label_out[b])
        stat.num_correct_predictions++;
    }

    update_train_stat(stat, outputs, labels);
  };

  auto eval_epoch_end = [this, batch_size, max_acc = 0.0f,
                         min_loss = std::numeric_limits<float>::max()](
                          RunStats &stat, DataBuffer &buffer) mutable {
    stat.loss /= static_cast<float>(stat.num_iterations);
    stat.accuracy = stat.num_correct_predictions /
                    static_cast<float>(stat.num_iterations * batch_size) *
                    100.0f;

    if (stat.accuracy > max_acc ||
        (stat.accuracy == max_acc && stat.loss < min_loss)) {
      max_acc = stat.accuracy;
      /// @note this is not actually 'the' min loss for whole time but records
      /// when data change
      min_loss = stat.loss;
      auto &save_best_path = std::get<props::SaveBestPath>(model_flex_props);
      if (!save_best_path.empty()) {
        save(save_best_path);
      }
    }
    std::cout << " >> [ Accuracy: " << stat.accuracy
              << "% - Validation Loss : " << stat.loss << " ]";
    ml_logi("[ Accuracy: %.2f %% - Validataion Loss: %.5f", stat.accuracy,
            stat.loss);
  };

  auto epochs = getEpochs();
  for (epoch_idx = epoch_idx + 1; epoch_idx <= epochs; ++epoch_idx) {
    training = run_epoch(train_buffer.get(), true, train_for_iteration,
                         update_train_stat, train_epoch_end);
    if (valid_buffer) {
      validation = run_epoch(valid_buffer.get(), false, eval_for_iteration,
                             update_eval_stat, eval_epoch_end);
    }
    std::cout << '\n';
  }

  if (test_buffer) {
    std::cout << "Evaluation with test data...\n";
    testing = run_epoch(test_buffer.get(), false, eval_for_iteration,
                        update_eval_stat, eval_epoch_end);
  }

  /** Clear the set inputs and labels */
  model_graph.setInputsLabels({}, {});

  return status;
}

void swap(NeuralNetwork &lhs, NeuralNetwork &rhs) {
  {
    using std::swap;

    swap(lhs.model_props, rhs.model_props);
    swap(lhs.model_flex_props, rhs.model_flex_props);
    swap(lhs.load_path, rhs.load_path);
    swap(lhs.epoch_idx, rhs.epoch_idx);
    swap(lhs.iter, rhs.iter);
    swap(lhs.loss, rhs.loss);
    swap(lhs.opt, rhs.opt);
    swap(lhs.data_buffers, rhs.data_buffers);
    swap(lhs.initialized, rhs.initialized);
    swap(lhs.model_graph, rhs.model_graph);
    swap(lhs.compiled, rhs.compiled);
    swap(lhs.loadedFromConfig, rhs.loadedFromConfig);
  }
}

int NeuralNetwork::addLayer(NodeType layer) {
  int status = ML_ERROR_NONE;

  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  /** Insert the layer to the graph */
  model_graph.addLayer(layer);

  return status;
}

int NeuralNetwork::extendGraph(GraphType graph, std::string prefix) {
  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (graph.size() == 0)
    return ML_ERROR_NONE;

  model_graph.extendGraph(graph, prefix);
  return ML_ERROR_NONE;
}

NeuralNetwork::GraphType
NeuralNetwork::getUnsortedLayers(const std::string &input_layer,
                                 const std::string &output_layer) {
  return model_graph.getUnsortedLayers(input_layer, output_layer);
}

int NeuralNetwork::setOptimizer(
  std::shared_ptr<ml::train::Optimizer> optimizer) {
  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  opt = std::static_pointer_cast<Optimizer>(optimizer);

  return ML_ERROR_NONE;
}

int NeuralNetwork::setDataBuffer(const DatasetModeType &mode,
                                 std::shared_ptr<DataBuffer> data_buffer) {
  if (data_buffer == nullptr) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->data_buffers[static_cast<int>(mode)] = data_buffer;

  return ML_ERROR_NONE;
}

int NeuralNetwork::getLayer(const char *name,
                            std::shared_ptr<ml::train::Layer> *layer) {
  *layer = std::static_pointer_cast<ml::train::Layer>(
    model_graph.getLayerNode(std::string(name)));
  return ML_ERROR_NONE;
}

void NeuralNetwork::printMetrics(std::ostream &out, unsigned int flags) {
  switch (flags) {
  case ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS:
    out << training.loss << std::endl;
    break;

  case ML_TRAIN_SUMMARY_MODEL_VALID_LOSS:
    out << validation.loss << std::endl;
    break;

  case ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY:
    out << validation.accuracy << std::endl;
    break;

  default:
    break;
  }
}

void NeuralNetwork::printPreset(std::ostream &out, unsigned int preset) {
  /** print neuralnet metrics */
  printMetrics(out, preset);
  if (preset > ML_TRAIN_SUMMARY_TENSOR)
    return;

  LayerNode::PrintPreset layer_preset = LayerNode::PrintPreset::PRINT_NONE;

  ///@todo match flags with preset
  unsigned int flags = PRINT_INST_INFO | PRINT_GRAPH_INFO | PRINT_PROP |
                       PRINT_OPTIMIZER | PRINT_METRIC;

  switch (preset) {
  case ML_TRAIN_SUMMARY_TENSOR:
    layer_preset = LayerNode::PrintPreset::PRINT_ALL;
    break;
  case ML_TRAIN_SUMMARY_LAYER:
    layer_preset = initialized ? LayerNode::PrintPreset::PRINT_SUMMARY
                               : LayerNode::PrintPreset::PRINT_SUMMARY_META;
    break;
  case ML_TRAIN_SUMMARY_MODEL:
    break;
  default:
    throw std::invalid_argument("given verbosity is invalid");
  }

  print(out, flags, layer_preset);
}

void NeuralNetwork::exportTo(Exporter &exporter,
                             const ExportMethods &method) const {
  exporter.saveResult(model_props, method, this);
  exporter.saveResult(model_flex_props, method, this);
}

void NeuralNetwork::print(std::ostream &out, unsigned int flags,
                          LayerNode::PrintPreset layerPrintPreset) {
  if (flags & PRINT_INST_INFO) {
    out << "===================";
    printInstance(out, this);
  }

  if (flags & PRINT_GRAPH_INFO) {
    out << "graph contains " << model_graph.size() << " operation nodes\n";
    /// @todo print graph info
  }

  if (flags & PRINT_PROP) {
    /// @todo print neuralnet property
    /// @todo print mode (if it is eval or training)
  }

  if (flags & PRINT_OPTIMIZER) {
    /// @todo print optimizer (with print optimizer prop)
  }

  if (flags & PRINT_METRIC) {
    /// @todo print metric (currently it is done at printPreset as a
    /// workaround)
    /// @todo print loss function when it is not initialized. (if it is
    /// initialized, loss layer will be printed)
  }

  if (model_graph.empty()) {
    out << "model is empty!" << std::endl;
    return;
  }

  /** print layer properties */
  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++)
    (*iter)->printPreset(out, layerPrintPreset);

  /// @todo Add status to check neuralnet has been run. #290
}
} /* namespace nntrainer */

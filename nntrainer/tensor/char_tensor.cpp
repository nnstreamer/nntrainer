// SPDX-License-Identifier: Apache-2.0
/**
 * @file	char_tensor.cpp
 * @date	02 April 2024
 * @brief	This is CharTensor class for 8-bit integer calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <blas_interface.h>
#include <char_tensor.h>
#include <tensor.h>

namespace nntrainer {

CharTensor::CharTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::QINT8) {}

CharTensor::CharTensor(const TensorDim &d, bool alloc_now, Initializer init,
                       std::string name) :
  TensorBase(d, alloc_now, init, name) {
  if (alloc_now)
    allocate();
}

CharTensor::CharTensor(const TensorDim &d, const void *buf) :
  CharTensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

CharTensor::CharTensor(
  std::vector<std::vector<std::vector<std::vector<int8_t>>>> const &d,
  Tformat fm) {
  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize CharTensor from empty vector");
  }

  dim.setTensorDim(0, d.size());
  if (fm == Tformat::NCHW) {
    dim.setTensorDim(1, d[0].size());
    dim.setTensorDim(2, d[0][0].size());
    dim.setTensorDim(3, d[0][0][0].size());
  } else {
    dim.setTensorDim(2, d[0].size());
    dim.setTensorDim(3, d[0][0].size());
    dim.setTensorDim(1, d[0][0][0].size());
  }

  dim.setTensorType({fm, Tdatatype::QINT8});

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;

  MemoryData *mem_data =
    new MemoryData((void *)(new int8_t[dim.getDataLen()]()));
  data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
    delete[] mem_data->getAddr<int8_t>();
  });

  offset = 0;

  // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2]
  // == height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
  // dim[1] == height, dim[2] == width, dim[3] == channel
  if (fm == Tformat::NCHW) {
    for (unsigned int i = 0; i < batch(); ++i)
      for (unsigned int j = 0; j < channel(); ++j)
        for (unsigned int k = 0; k < height(); ++k)
          for (unsigned int l = 0; l < width(); ++l)
            this->setValue(i, j, k, l, d[i][j][k][l]);
  } else {
    for (unsigned int i = 0; i < batch(); ++i)
      for (unsigned int j = 0; j < height(); ++j)
        for (unsigned int k = 0; k < width(); ++k)
          for (unsigned int l = 0; l < channel(); ++l)
            this->setValue(i, l, j, k, d[i][j][k][l]);
  }
}

bool CharTensor::operator==(const CharTensor &rhs) const {
  const int8_t *_data = (int8_t *)getData();
  const int8_t *_rdata = (int8_t *)rhs.getData();
  for (size_t i = 0; i < size(); ++i) {
    if (_data[i] != _rdata[i])
      return false;
  }

  return true;
}

void CharTensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new int8_t[dim.getDataLen()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<int8_t>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void CharTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *CharTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<int8_t>() + offset;
}

void *CharTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<int8_t>() + offset + idx;
}

void *CharTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((int8_t *)getData())[i];
}

const void *CharTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((int8_t *)getData())[i];
}

const int8_t &CharTensor::getValue(unsigned int i) const {
  return ((int8_t *)getData())[i];
}

int8_t &CharTensor::getValue(unsigned int i) {
  return ((int8_t *)getData())[i];
}

const int8_t &CharTensor::getValue(unsigned int b, unsigned int c,
                                   unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

int8_t &CharTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

void CharTensor::setValue(float value) {
  int8_t *data = (int8_t *)getData();
  std::fill(data, data + size(), value);
}

void CharTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  float output = ((int8_t *)getData())[idx];
  output *= beta;
  output += value;

  ((int8_t *)getData())[idx] = std::trunc(output);
}

void CharTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w, float value) {
  ((int8_t *)getData())[getIndex(b, c, h, w)] = (int8_t)value;
}

void CharTensor::setZero() {
  /// @todo replace with apply_i or scal
  setValue(0);
}

void CharTensor::initialize() {
  if (empty() || !isAllocated())
    return;

  /// @note Sampling from the normal/uniform distribution is invalid
  switch (initializer) {
  case Initializer::ZEROS:
    setZero();
    break;
  case Initializer::ONES:
    setValue(1.0f);
    break;
  case Initializer::NONE:
    break;
  default:
    throw std::invalid_argument("Initializer not valid for " +
                                getStringDataType());
    break;
  }

  putData();
}

void CharTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

void CharTensor::copy(const Tensor &from) {
  reshape(from.getDim());
  copy(from.getData());
}

void CharTensor::copyData(const Tensor &from) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of tensor to copy must match";

  /// @todo support copy from float32 & float16 to int8 data
  /// @note this could require scale factor
  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::QINT8:
    copy(from.getData());
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void CharTensor::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w, input.getValue<int8_t>(b, c, h, w));
        }
      }
    }
  }
}

std::vector<unsigned int> CharTensor::argmax() const {
  std::vector<unsigned int> result;
  const int8_t *data = (int8_t *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto max_iter =
      std::max_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, max_iter) - (b * feature_len);
  }
  return result;
}

float CharTensor::max_abs() const {
  const int8_t *data = (int8_t *)getData();
  unsigned int idx;

  int8_t max_val = data[0];
  for (unsigned int i = 1; i < size(); i += 1) {
    int8_t cur_val = (data[i] >= 0) ? data[i] : -1 * data[i];
    if (cur_val > max_val) {
      max_val = cur_val;
    }
  }

  return max_val;
}

float CharTensor::maxValue() const {
  const int8_t *data = (int8_t *)getData();
  return *std::max_element(data, data + size());
}

float CharTensor::minValue() const {
  const int8_t *data = (int8_t *)getData();
  return *std::min_element(data, data + size());
}

void CharTensor::print(std::ostream &out) const {
  const int8_t *data = (int8_t *)getData();
  unsigned int len = size();
  out << "data addr: " << reinterpret_cast<const float *>(data) << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << (int)data[0] << ' ' << (int)data[1] << ' ' << (int)data[2]
        << " ... " << (int)data[len - 3] << ' ' << (int)data[len - 2] << ' '
        << (int)data[len - 1] << ']' << std::endl;
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);
  if (getFormat() == Tformat::NCHW) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < height(); i++) {
          for (unsigned int j = 0; j < width(); j++) {
            out << std::setw(10) << (int)this->getValue(k, l, i, j) << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  } else {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int i = 0; i < height(); i++) {
        for (unsigned int j = 0; j < width(); j++) {
          for (unsigned int l = 0; l < channel(); l++) {
            out << std::setw(10) << (int)this->getValue(k, l, i, j) << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
    out.copyfmt(init);
  }
}

void CharTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  /// @todo need to optimize
  for (unsigned int i = 0; i < size(); ++i) {
    ((int8_t *)getData())[i] = ((int8_t *)buf)[i];
  }
}

} // namespace nntrainer

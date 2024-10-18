// SPDX-License-Identifier: Apache-2.0
/**
 * @file	uint_tensor.cpp
 * @date	02 April 2024
 * @brief	This is UIntTensor class for unsigned integer calculation
 *          This uint_tensor.cpp contains some codes to define
 *          UIntTensor template methods. This file cannot be used directly but
 *          included by uint_tensor.h only.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef __UINT_TENSOR_H__

template <typename T>
UIntTensor<T>::UIntTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, checkTensorDataType()) {}

template <typename T>
UIntTensor<T>::UIntTensor(const TensorDim &d, bool alloc_now, Initializer init,
                          std::string name) :
  TensorBase(d, alloc_now, init, name) {
  if (alloc_now)
    allocate();
}

template <typename T>
UIntTensor<T>::UIntTensor(const TensorDim &d, const void *buf) :
  UIntTensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

template <typename T>
UIntTensor<T>::UIntTensor(
  std::vector<std::vector<std::vector<std::vector<T>>>> const &d, Tformat fm) {
  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize UIntTensor from empty vector");
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

  dim.setTensorType({fm, checkTensorDataType()});

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;

  MemoryData *mem_data = new MemoryData((void *)(new T[dim.getDataLen()]()));
  data = std::shared_ptr<MemoryData>(
    mem_data, [](MemoryData *mem_data) { delete[] mem_data->getAddr<T>(); });

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

template <typename T>
bool UIntTensor<T>::operator==(const UIntTensor<T> &rhs) const {
  const T *_data = (T *)getData();
  const T *_rdata = (T *)rhs.getData();
  for (size_t i = 0; i < size(); ++i) {
    if (_data[i] != _rdata[i])
      return false;
  }

  return true;
}

template <typename T> void UIntTensor<T>::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new T[dim.getDataLen()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<T>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

template <typename T> void UIntTensor<T>::deallocate() {
  data = nullptr;
  offset = 0;
}
template <typename T> void *UIntTensor<T>::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<T>() + offset;
}

template <typename T> void *UIntTensor<T>::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<T>() + offset + idx;
}

template <typename T> void *UIntTensor<T>::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((T *)getData())[i];
}

template <typename T>
const void *UIntTensor<T>::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((T *)getData())[i];
}

template <typename T> const T &UIntTensor<T>::getValue(unsigned int i) const {
  return ((T *)getData())[i];
}

template <typename T> T &UIntTensor<T>::getValue(unsigned int i) {
  return ((T *)getData())[i];
}

template <typename T>
const T &UIntTensor<T>::getValue(unsigned int b, unsigned int c, unsigned int h,
                                 unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

template <typename T>
T &UIntTensor<T>::getValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

template <typename T> void UIntTensor<T>::setValue(float value) {
  T *data = (T *)getData();
  std::fill(data, data + size(), value);
}

template <typename T>
void UIntTensor<T>::addValue(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  float output = ((T *)getData())[idx];
  output *= beta;
  output += value;

  ((T *)getData())[idx] = std::trunc(output);
}

template <typename T>
void UIntTensor<T>::setValue(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w, float value) {
  ((T *)getData())[getIndex(b, c, h, w)] = (T)value;
}

template <typename T> void UIntTensor<T>::setZero() {
  /// @todo replace with apply_i or scal
  setValue(0);
}

template <typename T> void UIntTensor<T>::initialize() {
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

template <typename T> void UIntTensor<T>::initialize(Initializer init) {
  initializer = init;
  initialize();
}

template <typename T> void UIntTensor<T>::copy(const Tensor &from) {
  reshape(from.getDim());
  copy(from.getData());
}

template <typename T> void UIntTensor<T>::copyData(const Tensor &from) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of tensor to copy must match";

  /// @todo support copy from other data types
  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::UINT16:
    copy(from.getData());
    break;
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

template <typename T>
void UIntTensor<T>::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w, input.getValue<T>(b, c, h, w));
        }
      }
    }
  }
}

template <typename T> std::vector<unsigned int> UIntTensor<T>::argmax() const {
  std::vector<unsigned int> result;
  const T *data = (T *)getData();
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

template <typename T> float UIntTensor<T>::max_abs() const {
  return maxValue();
}

template <typename T> float UIntTensor<T>::maxValue() const {
  const T *data = (T *)getData();
  return *std::max_element(data, data + size());
}

template <typename T> float UIntTensor<T>::minValue() const {
  const T *data = (T *)getData();
  return *std::min_element(data, data + size());
}

template <typename T> void UIntTensor<T>::print(std::ostream &out) const {
  const T *data = (T *)getData();
  unsigned int len = size();
  out << "data addr: " << reinterpret_cast<const float *>(data) << '\n';
  out << dim;

  if (len > 512) {
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

template <typename T> void UIntTensor<T>::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  /// @todo need to optimize
  for (unsigned int i = 0; i < size(); ++i) {
    ((T *)getData())[i] = ((T *)buf)[i];
  }
}

#endif

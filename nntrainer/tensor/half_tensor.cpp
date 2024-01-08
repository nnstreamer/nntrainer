// SPDX-License-Identifier: Apache-2.0
/**
 * @file	half_tensor.cpp
 * @date	01 December 2023
 * @brief	This is a HalfTensor class for 16-bit floating point calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <blas_interface.h>
#include <half_tensor.h>
#include <util_func.h>

namespace nntrainer {

HalfTensor::HalfTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::FP16) {}

HalfTensor::HalfTensor(const TensorDim &d, bool alloc_now, Initializer init,
                       std::string name) :
  TensorBase(d, alloc_now, init, name) {
  if (alloc_now)
    allocate();
}

HalfTensor::HalfTensor(const TensorDim &d, const void *buf) :
  HalfTensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

HalfTensor::HalfTensor(
  std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
  Tformat fm) {

  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize HalfTensor from empty vector");
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

  dim.setTensorType({fm, Tdatatype::FP16});

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;

  MemoryData *mem_data =
    new MemoryData((void *)(new _FP16[dim.getDataLen()]()));
  data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
    delete[] mem_data->getAddr<_FP16>();
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

bool HalfTensor::operator==(const HalfTensor &rhs) const {
  const _FP16 *_data = (_FP16 *)getData();
  const _FP16 *_rdata = (_FP16 *)rhs.getData();
  for (size_t i = 0; i < size(); ++i) {
    if (std::isnan((float)_data[i]) || std::isnan((float)_rdata[i]) ||
        std::fabs((float)(_data[i] - _rdata[i])) > epsilon)
      return false;
  }

  return true;
}

/// @todo support allocation by src_tensor
void HalfTensor::allocate() {
  if (empty() || data)
    /// already allocated
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new _FP16[dim.getDataLen()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<_FP16>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void HalfTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *HalfTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<_FP16>() + offset;
}

void *HalfTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<_FP16>() + offset + idx;
}

void *HalfTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((_FP16 *)getData())[i];
}

const void *HalfTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((_FP16 *)getData())[i];
}

void HalfTensor::setValue(float value) {
  _FP16 *data = (_FP16 *)getData();
  std::fill(data, data + size(), static_cast<_FP16>(value));
}

void HalfTensor::setValue(unsigned int batch, unsigned int c, unsigned int h,
                          unsigned int w, float value) {
  ((_FP16 *)getData())[getIndex(batch, c, h, w)] = static_cast<_FP16>(value);
}

void HalfTensor::setZero() {
  if (contiguous) {
    sscal(size(), 0, (_FP16 *)getData(), 1);
  } else {
    /// @todo implement apply_i
    // apply_i<_FP16>([](_FP16 val) -> _FP16 { return 0; });
    setValue(0);
  }
}

/// @todo support additional initializer
void HalfTensor::initialize() {
  if (empty() || !isAllocated())
    return;

  switch (initializer) {
  case Initializer::ZEROS:
    setZero();
    break;
  case Initializer::ONES:
    setValue(1.0f);
    break;
  default:
    break;
  }

  putData();
}

void HalfTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

void HalfTensor::print(std::ostream &out) const {
  printInstance(out, this);
  const _FP16 *data = (_FP16 *)getData();
  unsigned int len = size();
  out << "data addr: " << data << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << (float)data[0] << ' ' << (float)data[1] << ' '
        << (float)data[2] << " ... " << (float)data[len - 3] << ' '
        << (float)data[len - 2] << ' ' << (float)data[len - 1] << ']'
        << std::endl;
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);

  if (getFormat() == Tformat::NCHW) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < height(); i++) {
          for (unsigned int j = 0; j < width(); j++) {
            out << std::setw(10) << std::setprecision(10)
                << (float)data[getIndex(k, l, i, j)] << " ";
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
            out << std::setw(10) << std::setprecision(10)
                << (float)data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  }
  out.copyfmt(init);
}

/// @todo include getName()
void HalfTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << "Tensor is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  scopy(size(), (_FP16 *)buf, 1, (_FP16 *)getData(), 1);
}

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_base.h
 * @date	27 November 2023
 * @brief	This is Tensor concept and base class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __TENSOR_BASE_H__
#define __TENSOR_BASE_H__
#ifdef __cplusplus

namespace nntrainer {

/**
 * @class TensorConcept class
 * @brief TensorConcept is an abstract class
 */
class TensorConcept {
public:
  /**
   * @brief Basic Destructor of TensorConcept
   */
  virtual ~TensorConcept() {}
  /**
   * @copydoc TensorV2::allocate()
   */
  virtual void allocate() = 0;

  /**
   * @copydoc TensorV2::deallocate()
   */
  virtual void deallocate() = 0;

  /**
   * @copydoc TensorV2::isAllocated()
   */
  virtual bool isAllocated() = 0;

  /**
   * @copydoc TensorV2::setData()
   */
  virtual void setData(const std::shared_ptr<MemoryData> buf, size_t off = 0,
                       bool init = false) = 0;

  /**
   * @copydoc TensorV2::getData()
   */
  virtual const void *getData() const = 0;

  /**
   * @copydoc TensorV2::getData(size_t idx)
   */
  virtual void *getData(size_t idx) const = 0;

  /**
   * @copydoc TensorV2::sizeofData()
   */
  virtual unsigned int sizeofData() const = 0;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  virtual void *getAddress(unsigned int i) = 0;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  virtual const void *getAddress(unsigned int i) const = 0;

  /**
   * @copydoc TensorV2::setValue(float value)
   */
  virtual void setValue(float value) = 0;

  /**
   * @copydoc TensorV2::setValue(b, c, h, w, value)
   */
  virtual void setValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value) noexcept = 0;

  /**
   * @copydoc TensorV2::addValue(b, c, h, w, value, beta)
   */
  virtual void addValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value, float beta) noexcept = 0;

  /**
   * @copydoc TensorV2::setZero()
   */
  virtual void setZero() = 0;

  /**
   * @copydoc TensorV2::setRandNormal(float mean, float std)
   */
  virtual void setRandNormal(float mean, float std) = 0;

  /**
   * @copydoc TensorV2::setRandUniform(float min, float max)
   */
  virtual void setRandUniform(float min, float max) = 0;

  /**
   * @copydoc TensorV2::setRandBernoulli(float probability)
   */
  virtual void setRandBernoulli(float probability) = 0;

  /**
   * @copydoc TensorV2::initialize()
   */
  virtual void initialize() = 0;

  /**
   * @copydoc TensorV2::initialize(Initializer init)
   */
  virtual void initialize(Initializer init) = 0;

  /**
   * @copydoc TensorV2::print(std::ostream &out)
   */
  virtual void print(std::ostream &out) const = 0;

  /**
   * @copydoc TensorV2::size()
   */
  virtual size_t size() const = 0;

  /**
   * @copydoc TensorV2::getIndex()
   */
  virtual size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const noexcept = 0;

  /**
   * @copydoc TensorV2::setName(const std::string &name_)
   */
  virtual void setName(const std::string &name_) = 0;

  /**
   * @copydoc TensorV2::getName()
   */
  virtual const std::string &getName() const = 0;

  /**
   * @copydoc TensorV2::getInitializer()
   */
  virtual Initializer getInitializer() const = 0;

  /**
   * @copydoc TensorV2::getDim()
   */
  virtual TensorDim getDim() const = 0;

  /**
   * @copydoc TensorV2::getStrides()
   */
  virtual const std::array<size_t, TensorDim::MAXDIM> getStrides() const
    noexcept = 0;

  /**
   * @copydoc TensorV2::getContiguous()
   */
  virtual bool getContiguous() const = 0;

  /**
   * @copydoc TensorV2::getTensorType()
   */
  virtual TensorDim::TensorType getTensorType() const = 0;

  /**
   * @copydoc TensorV2::batch()
   */
  virtual size_t batch() const = 0;

  /**
   * @copydoc TensorV2::channel()
   */
  virtual size_t channel() const = 0;

  /**
   * @copydoc TensorV2::height()
   */
  virtual size_t height() const = 0;

  /**
   * @copydoc TensorV2::width()
   */
  virtual size_t width() const = 0;

  /**
   * @copydoc TensorV2::getDataTypeSize()
   */
  virtual uint getDataTypeSize() const = 0;

  /**
   * @copydoc TensorV2::getFormat()
   */
  virtual TensorDim::Format getFormat() const = 0;

  /**
   * @copydoc TensorV2::getDataType()
   */
  virtual Tdatatype getDataType() const = 0;

  /**
   * @copydoc TensorV2::reshape()
   */
  virtual void reshape(const TensorDim &d) = 0;
};

/**
 * @brief TensorBase class
 * @note TensorClass : FloatTensor, HalfTensor, etc.
 */
template <typename TensorClass> class TensorBase : public TensorConcept {
public:
  /**
   * @brief     Basic Constructor of TensorBase
   */
  TensorBase(TensorClass t) : object(t) {}

  /**
   * @brief     Basic Destructor of TensorBase
   */
  virtual ~TensorBase() {}

  /**
   * @copydoc TensorV2::allocate()
   */
  virtual void allocate() { object.allocate(); }

  /**
   * @copydoc TensorV2::deallocate()
   */
  virtual void deallocate() { object.deallocate(); }

  /**
   * @copydoc TensorV2::isAllocated()
   */
  virtual bool isAllocated() { return object.isAllocated(); }

  /**
   * @copydoc TensorV2::setData()
   */
  virtual void setData(const std::shared_ptr<MemoryData> buf, size_t off = 0,
                       bool init = false) {
    object.setData(buf, off, init);
  }

  /**
   * @copydoc TensorV2::getData()
   */
  virtual const void *getData() const { return object.getData(); }

  /**
   * @copydoc TensorV2::getData(size_t idx)
   */
  virtual void *getData(size_t idx) const { return object.getData(idx); }

  /**
   * @copydoc TensorV2::sizeofData()
   */
  virtual unsigned int sizeofData() const { return object.sizeofData(); }

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  virtual void *getAddress(unsigned int i) { return object.getAddress(i); }

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  virtual const void *getAddress(unsigned int i) const {
    return object.getAddress(i);
  }

  /**
   * @copydoc TensorV2::setValue(float value)
   */
  virtual void setValue(float value) { object.setValue(value); }

  /**
   * @copydoc TensorV2::setValue(b, c, h, w, value)
   */
  virtual void setValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value) noexcept {
    object.setValue(b, c, h, w, value);
  }

  /**
   * @copydoc TensorV2::addValue(b, c, h, w, value, beta)
   */
  virtual void addValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value, float beta) noexcept {
    object.addValue(b, c, h, w, value, beta);
  }

  /**
   * @copydoc TensorV2::setZero()
   */
  virtual void setZero() { object.setZero(); }

  /**
   * @copydoc TensorV2::setRandNormal(float mean, float std)
   */
  virtual void setRandNormal(float mean, float std) {
    object.setRandNormal(mean, std);
  }

  /**
   * @copydoc TensorV2::setRandUniform(float min, float max)
   */
  virtual void setRandUniform(float min, float max) {
    object.setRandUniform(min, max);
  }

  /**
   * @copydoc TensorV2::setRandBernoulli(float probability)
   */
  virtual void setRandBernoulli(float probability) {
    object.setRandBernoulli(probability);
  }

  /**
   * @copydoc TensorV2::initialize()
   */
  virtual void initialize() { object.initialize(); }

  /**
   * @copydoc TensorV2::initialize(Initializer init)
   */
  virtual void initialize(Initializer init) { object.initialize(init); }

  /**
   * @copydoc TensorV2::print(std::ostream &out)
   */
  virtual void print(std::ostream &out) const { object.print(out); }

  /**
   * @copydoc TensorV2::size()
   */
  virtual size_t size() const { return object.size(); }

  /**
   * @copydoc TensorV2::getIndex()
   */
  virtual size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const noexcept {
    return object.getIndex(b, c, h, w);
  }

  /**
   * @copydoc TensorV2::setName(const std::string &name_)
   */
  virtual void setName(const std::string &name_) { object.setName(name_); }

  /**
   * @copydoc TensorV2::getName()
   */
  virtual const std::string &getName() const { return object.getName(); }

  /**
   * @copydoc TensorV2::getInitializer()
   */
  virtual Initializer getInitializer() const { return object.getInitializer(); }

  /**
   * @copydoc TensorV2::getDim()
   */
  virtual TensorDim getDim() const { return object.getDim(); }

  /**
   * @copydoc TensorV2::getStrides()
   */
  virtual const std::array<size_t, TensorDim::MAXDIM> getStrides() const
    noexcept {
    return object.getStrides();
  }

  /**
   * @copydoc TensorV2::getContiguous()
   */
  virtual bool getContiguous() const { return object.getContiguous(); }

  /**
   * @copydoc TensorV2::getTensorType()
   */
  virtual TensorDim::TensorType getTensorType() const {
    return object.getTensorType();
  }

  /**
   * @copydoc TensorV2::batch()
   */
  virtual size_t batch() const { return object.batch(); }

  /**
   * @copydoc TensorV2::channel()
   */
  virtual size_t channel() const { return object.channel(); }

  /**
   * @copydoc TensorV2::height()
   */
  virtual size_t height() const { return object.height(); }

  /**
   * @copydoc TensorV2::width()
   */
  virtual size_t width() const { return object.width(); }

  /**
   * @copydoc TensorV2::getDataTypeSize()
   */
  virtual uint getDataTypeSize() const { return object.getDataTypeSize(); }

  /**
   * @copydoc TensorV2::getFormat()
   */
  virtual TensorDim::Format getFormat() const { return object.getFormat(); }

  /**
   * @copydoc TensorV2::getDataType()
   */
  virtual Tdatatype getDataType() const { return object.getDataType(); }

  /**
   * @copydoc TensorV2::reshape()
   */
  virtual void reshape(const TensorDim &d) { object.reshape(d); }

private:
  TensorClass object;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_BASE_H__ */

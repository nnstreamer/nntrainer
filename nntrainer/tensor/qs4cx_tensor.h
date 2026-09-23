// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qs4cx_tensor.h
 * @date	17 June 2026
 * @brief	This is QS4CX_Tensor class for QS4CX quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jaemin Shin <jaemin980311@google.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __QS4CX_TENSOR_H__
#define __QS4CX_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <tensor_base.h>

namespace nntrainer {

/**
 * @class QS4CX_Tensor class
 * @brief QS4CX_Tensor class for QS4CX quantized tensor
 */
class QS4CX_Tensor : public TensorBase {

public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  QS4CX_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new QS4CX_Tensor object
   *
   * @param d Tensor dim for this qs4cx tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  QS4CX_Tensor(const TensorDim &d, bool alloc_now,
               Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new QS4CX_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  QS4CX_Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new QS4CX_Tensor object
   * @param rhs TensorBase object to copy
   */
  QS4CX_Tensor(TensorBase &rhs) : TensorBase(rhs) {}

  /**
   * @copydoc Tensor::allocate()
   */
  void allocate() override;

  /**
   * @copydoc Tensor::deallocate()
   */
  void deallocate() override {
    data = nullptr;
    offset = 0;
  }

  /**
   * @copydoc Tensor::getData()
   */
  void *getData() const override;

  /**
   * @copydoc Tensor::getData()
   */
  void *getData(size_t idx) const override {
    throw std::invalid_argument(
      "QS4CX_Tensor::getData() is not supported. Use getData() instead.");
  }

  /**
   * @copydoc Tensor::getPackedData()
   */
  void *getPackedData() const override;

  /**
   * @copydoc Tensor::getAddress()
   */
  void *getAddress(unsigned int i) override {
    throw std::invalid_argument("QS4CX_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  const void *getAddress(unsigned int i) const override {
    throw std::invalid_argument("QS4CX_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(float value) override {
    throw std::invalid_argument("QS4CX_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("QS4CX_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::addValue()
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("QS4CX_Tensor::addValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize(Initializer init) override {
    throw std::invalid_argument("QS4CX_Tensor::initialize() is not supported.");
  }

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::print()
   */
  void print(std::ostream &out) const override;

  /**
   * @copydoc Tensor::copy()
   */
  void copy(const Tensor &from) override {
    throw std::invalid_argument("QS4CX_Tensor::copy() is not supported.");
  }

  /**
   * @copydoc Tensor::copyData()
   */
  void copyData(const Tensor &from) override {
    throw std::invalid_argument("QS4CX_Tensor::copyData() is not supported.");
  }

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "QS4CX_Tensor::copy_with_stride() is not supported.");
  }

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs() const override {
    throw std::invalid_argument("QS4CX_Tensor::max_abs() is not supported.");
  }

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override {
    throw std::invalid_argument("QS4CX_Tensor::maxValue() is not supported.");
  }

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override {
    throw std::invalid_argument("QS4CX_Tensor::minValue() is not supported.");
  }

  /**
   * @copydoc TensorBase::size()
   */
  size_t size() const override;

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::getScale()
   */
  void *getScale() const override;

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

  /**
   * @brief Eagerly pack the weight data after loading
   * @note Must be called after load_weight() to prepare for computation
   * @note Prepares weight data for efficient matrix multiplication
   */
  void pack() override;

protected:
  /**
   * @brief copy a buffer to @a this -- protected rather than private because
   * QS4CX_WH_Tensor has the same buffer layout below its extra column sums
   * and fills itself the same way. The caller has to ensure that @a this is
   * initialized, otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy_qs4cx(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QS4CX)
   */
  std::string getStringDataType() const override { return "QS4CX"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid() const override { return true; }

  std::unique_ptr<uint8_t[]> packed_data = nullptr;
};

/**
 * @class   QS4CX_WH_Tensor
 * @brief   QS4CX whose nibbles are already in the HMX WH tile layout.
 *
 * Same values, same per-output-channel scales, same number of nibble bytes --
 * a WH tile is 32x32 i4 in 512 bytes, so the packing costs nothing over
 * row-major. Two things differ:
 *
 *  - the nibbles are arranged as htp_wh_layout.h describes, which is what the
 *    HTP backend registers directly instead of converting and baking on the
 *    DSP (doc 46 section 35);
 *  - a per-output-channel column sum follows the scales, because the matmul
 *    needs it to correct for the activation zero point and recomputing it
 *    from the packed nibbles costs about 20 seconds across this model.
 *
 * So the only layout difference is the extra N floats, which is why getData
 * and getScale are inherited unchanged. Nothing on the CPU can read this
 * tensor: the layout is the accelerator's, and a CPU kernel handed one would
 * compute a wrong answer rather than fail, so loading a model with these
 * weights commits it to the HTP path.
 */
class QS4CX_WH_Tensor : public QS4CX_Tensor {
public:
  /** @brief Basic constructor */
  QS4CX_WH_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW) :
    QS4CX_Tensor(name_, fm) {}

  /**
   * @brief Construct a new QS4CX_WH_Tensor
   *
   * Allocation is deferred out of the base constructor and done here.
   * QS4CX_Tensor::allocate() sizes the buffer with size(), and a virtual call
   * from a base constructor runs the BASE override -- which would quietly
   * allocate N floats too few and leave every column sum reading past the end.
   */
  QS4CX_WH_Tensor(const TensorDim &d, bool alloc_now,
                  Initializer init = Initializer::NONE, std::string name = "") :
    QS4CX_Tensor(d, false, init, name) {
    if (alloc_now)
      allocate();
  }

  /** @brief Construct from a buffer */
  QS4CX_WH_Tensor(const TensorDim &d, const void *buf = nullptr) :
    QS4CX_WH_Tensor(d, true, Initializer::NONE, "") {
    if (d.getDataLen() != 0 && buf != nullptr)
      copy_qs4cx(buf);
  }

  /** @brief Copy constructor from TensorBase */
  QS4CX_WH_Tensor(TensorBase &rhs) : QS4CX_Tensor(rhs) {}

  /**
   * @copydoc Tensor::size()
   * @note The nibble half matches QS4CX exactly; the extra N floats are the
   *       column sums.
   */
  size_t size() const override {
    return QS4CX_Tensor::size() + width() * sizeof(float);
  }

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override { return size() * sizeof(uint8_t); }

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QS4CX_WH)
   */
  std::string getStringDataType() const override { return "QS4CX_WH"; }
};

/**
 * @class QS2CX_WH_Tensor
 * @brief QS4CX_WH at two bits a weight
 *
 * Same tile order, same per-output-channel scale, same column sums -- only
 * the codes are narrower. Each code indexes one of four int4 values in a
 * palette that covers the whole tensor, and the HTP expands the codes back
 * to the int4 lattice in VTCM before the matmul, so everything downstream
 * of the matmul is bit-identical to QS4CX_WH (htp_wh_palette.h).
 *
 * Layout: [K*N/4 codes][4 palette bytes][N f32 scales][N f32 column sums].
 *
 * The palette is per tensor rather than per k-group because doc 54 5.3
 * measured group size against the real checkpoint and it changed the SNR by
 * nothing -- the code histogram has the same shape everywhere, so the fit
 * picks the same four levels. A recipe that makes groups differ (GPTQ would)
 * needs a group axis here; this one does not, and carrying an axis nothing
 * varies along would be carried into the DSP expansion as a table swap.
 */
class QS2CX_WH_Tensor : public QS4CX_WH_Tensor {
public:
  /** @brief Palette entries stored after the codes */
  static constexpr size_t PALETTE_BYTES = 4;

  /** @brief Basic constructor */
  QS2CX_WH_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW) :
    QS4CX_WH_Tensor(name_, fm) {}

  /**
   * @brief Construct a new QS2CX_WH_Tensor
   * @note Allocation is deferred for the reason QS4CX_WH_Tensor documents:
   *       a virtual size() called from a base constructor runs the base
   *       override and under-allocates.
   */
  QS2CX_WH_Tensor(const TensorDim &d, bool alloc_now,
                  Initializer init = Initializer::NONE, std::string name = "") :
    QS4CX_WH_Tensor(d, false, init, name) {
    if (alloc_now)
      allocate();
  }

  /** @brief Construct from a buffer */
  QS2CX_WH_Tensor(const TensorDim &d, const void *buf = nullptr) :
    QS2CX_WH_Tensor(d, true, Initializer::NONE, "") {
    if (d.getDataLen() != 0 && buf != nullptr)
      copy_qs4cx(buf);
  }

  /** @brief Copy constructor from TensorBase */
  QS2CX_WH_Tensor(TensorBase &rhs) : QS4CX_WH_Tensor(rhs) {}

  /**
   * @copydoc Tensor::size()
   */
  size_t size() const override {
    const size_t K = height();
    const size_t N = width();
    return N * ((K + 3) / 4) + PALETTE_BYTES + 2 * width() * sizeof(float);
  }

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override { return size() * sizeof(uint8_t); }

  /**
   * @copydoc QS4CX_Tensor::getScale()
   * @note The palette sits between the codes and the scales, so the base
   *       class's offset would land four bytes short.
   */
  void *getScale() const override {
    if (!data)
      return nullptr;
    data->validate();
    const size_t K = height();
    const size_t N = width();
    return ((int8_t *)getData()) + N * ((K + 3) / 4) + PALETTE_BYTES;
  }

  /** @brief The four int4 codes the 2-bit indices name, ascending */
  const int8_t *getPalette() const {
    if (!data)
      return nullptr;
    data->validate();
    const size_t K = height();
    const size_t N = width();
    return ((const int8_t *)getData()) + N * ((K + 3) / 4);
  }

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QS2CX_WH)
   */
  std::string getStringDataType() const override { return "QS2CX_WH"; }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QS4CX_TENSOR_H__ */

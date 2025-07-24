// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   delegate.h
 * @date   7 Aug 2020
 * @brief  This is Delegate Class for the Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note This class is experimental and subject to major modifications.
 *
 */

#ifndef __DELEGATE_H__
#define __DELEGATE_H__

namespace nntrainer {

/**
 * @class Backend
 * @brief Backend to be used for the operations to use.
 * @note If some operation is not supported on set backend, that operation
 * implemented with default backend will be used.
 *
 */
class Backend {
public:
  /**
   * @brief Backend type enumeration
   */
  enum class BackendType {
    armcl, /**< Arm Compute Library for backend */
    blas,  /**< Libopenblas for backend */
    base   /**< Internally implemented operations for backend */
  };

  /**
   * @brief Construct a new Backend object
   *
   * @param backend Backend of be used. Defaults to base backend
   * @param num_t Number of threads. Defaults to 1 thread
   */
  Backend(BackendType backend = BackendType::base, int num_t = 1) :
    backend(backend),
    num_threads(num_t) {}

  /**
   * @brief Set the number of threads for the backend if supported
   *
   * @param num_threads Number of threads
   */
  void setNumThreads(unsigned int num_threads);

private:
  /**
   * @brief backend type stored
   */
  BackendType backend;

  /**
   * @brief number of threads stored
   */
  unsigned int num_threads;
};

/**
 * @class Device
 * @brief Device to be used for the operations to run.
 * @note The selected delegate device will restrict the supported backend.
 *
 */
class Device {
public:
  /**
   * @brief Device type enumeration
   */
  enum class DeviceType {
    cpu, /**< CPU as the device */
    gpu, /**< GPU as the device */
    npu  /**< NPU as the device */
  };

  /**
   * @brief Construct a new Device object
   *
   * @param device Device delegate to be used for the operations to run. Default
   * device is cpu.
   * @param mem_frac Maximum memory fraction to be used of the set device.
   * @param soft_place To enable soft device placement.
   * @param prec_loss To enable precision loss for faster computation for the
   * device.
   */
  Device(DeviceType device = DeviceType::cpu, float mem_frac = 1.0,
         soft_place = false, prec_loss = false) :
    device(device),
    memory_fraction(mem_frac),
    soft_placement(soft_place),
    precision_loss(prec_loss) {}

  /**
   * @brief Set the Device object
   *
   * @param device The device to be set to
   */
  void setDevice(DeviceType device) { this->device = device; }

  /**
   * @brief Get the Device object
   *
   * @return DeviceType The device type which is set
   */
  DeviceType getDevice() { return device; }

  /**
   * @brief Set the maximum memory fraction which can be used by the framework
   *
   * @param memory_fraction Maximum fraction of memory to be used
   */
  void setMemoryFraction(float memory_fraction) {
    throw std::logic_error("Not yet implemented");
  }

  /**
   * @brief Allow placing the operations on device softly
   * @details This allows framework to use some other device other than the set
   * device for some of the operations for optimization or if an operation is
   * not supported for the selected device. This might incur extra memory copy
   * overhead.
   * @note When set to false, constructing the model can result in error if a
   * required operation is not avaiable for that device.
   *
   * @param soft_placement True to allow soft placement, else false
   */
  void allowSoftPlacement(bool soft_placement) {
    throw std::logic_error("Not yet implemented");
    this->soft_placement = soft_placement;
  }

  /**
   * @brief Allow using low precision version of some of the operations
   * @details This allows framework to use low precision operations (like
   * float16 instead of float32) for some of the operations for higher
   * performance with minimum impact to performance
   *
   * @param precision_loss True to allow loss in precision, else false
   */
  void allowPrecisionLoss(bool precision_loss) {
    throw std::logic_error("Not yet implemented");
    this->precision_loss = precision_loss;
  }

private:
  DeviceType device;     /**< Device type for the object */
  float memory_fraction; /**< Maximum memory fraction which can be used by the
                            framework*/
  bool soft_placement;   /**< Allow placing the operations on device softly */
  bool precision_loss;   /**< Allow using low precision version of some of the
                            operations */
};

/**
 * @class DelegateConfig
 * @brief Configuration for the delegate
 *
 */
class DelegateConfig {
public:
  /**
   * @brief Construct a new Delegate Config object
   *
   * @param backend Backend to be set for this delegate
   * @param device Device to be set for the delegate
   */
  DelegateConfig(Backend backend, Device device) :
    backend(backend),
    device(device) {}

private:
  Backend backend; /**< Backend to be used for the operations to use */
  Device device;   /**< Device to be used for the operations to run. */
};

} // namespace nntrainer

#endif /* __DELEGATE_H__ */

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../../jni/libs/arm64-v8a
NNTRAINER_INCLUDES := $(LOCAL_PATH)/../../../nntrainer/include
endif

NNTRAINER_APPLICATION := $(LOCAL_PATH)/../../

include $(CLEAR_VARS)

ifndef TENSORFLOW_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning TENSORFLOW_ROOT is not defined!)
$(warning TENSORFLOW SRC is going to be downloaded!)

# Currently we are using tensorflow 1.9.0
$(info $(shell ($(LOCAL_PATH)/../../prepare_tflite.sh $(NNTRAINER_APPLICATION))))

TENSORFLOW_ROOT := $(LOCAL_PATH)/../../tensorflow-1.9.0

endif
endif

TF_LITE_DIR=$(TENSORFLOW_ROOT)/tensorflow/contrib/lite

LOCAL_MODULE := tensorflow-lite
TFLITE_SRCS := \
    $(wildcard $(TF_LITE_DIR)/*.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/*.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/*.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/optimized/*.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/reference/*.cc) \
    $(wildcard $(TF_LITE_DIR)/*.c) \
    $(wildcard $(TF_LITE_DIR)/kernels/*.c) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/*.c) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/optimized/*.c) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/reference/*.c) \
    $(wildcard $(TF_LITE_DIR)/downloads/farmhash/src/farmhash.cc) \
    $(wildcard $(TF_LITE_DIR)/downloads/fft2d/fftsg.c)

TFLITE_SRCS := $(sort $(TFLITE_SRCS))

TFLITE_EXCLUDE_SRCS := \
    $(wildcard $(TF_LITE_DIR)/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/test_util.cc) \
    $(wildcard $(TF_LITE_DIR)/examples/minimal/minimal.cc)

TFLITE_SRCS := $(filter-out $(TFLITE_EXCLUDE_SRCS), $(TFLITE_SRCS))
# ANDROID_NDK env should be set before build
TFLITE_INCLUDES := \
    $(ANDROID_NDK)/../ \
    $(TENSORFLOW_ROOT) \
    $(TF_LITE_DIR)/downloads \
    $(TF_LITE_DIR)/downloads/eigen \
    $(TF_LITE_DIR)/downloads/gemmlowp \
    $(TF_LITE_DIR)/downloads/neon_2_sse \
    $(TF_LITE_DIR)/downloads/farmhash/src \
    $(TF_LITE_DIR)/downloads/flatbuffers/include


LOCAL_SRC_FILES := $(TFLITE_SRCS)
LOCAL_C_INCLUDES := $(TFLITE_INCLUDES)

LOCAL_CFLAGS += -O3 -DNDEBUG
LOCAL_CXXFLAGS += -std=c++11 -frtti -fexceptions -O3 -DNDEBUG 

include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++11 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/arm64-v8a/
LOCAL_CXXFLAGS += -std=c++11
LOCAL_CFLAGS += -pthread -fopenmp
LOCAL_LDFLAGS += -fopenmp 
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_training
LOCAL_LDLIBS := -llog

LOCAL_SRC_FILES := main.cpp bitmap_helpers.cpp

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_STATIC_LIBRARIES := tensorflow-lite 

LOCAL_C_INCLUDES += $(TFLITE_INCLUDES) $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)

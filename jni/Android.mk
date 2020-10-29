# ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j2
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ENABLE_TFLITE_BACKBONE := 0

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/..
endif

include $(CLEAR_VARS)

ifndef INIPARSER_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning INIPARSER_ROOT is not defined!)
$(warning INIPARSER SRC is going to be downloaded!)

INIPARSER_ROOT :=./iniparser

endif
endif

$(info $(shell ($(LOCAL_PATH)/prepare_iniparser.sh )))

include $(CLEAR_VARS)

NNTRAINER_JNI_ROOT := $(NNTRAINER_ROOT)/jni

# Build tflite if its backbone is enabled
ifeq ($(ENABLE_TFLITE_BACKBONE),1)
$(warning BUILDING TFLITE BACKBONE !)
TENSORFLOW_VERSION := 1.13.1

ifndef TENSORFLOW_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning TENSORFLOW_ROOT is not defined!)
$(warning TENSORFLOW SRC is going to be downloaded!)

$(info $(shell ($(NNTRAINER_JNI_ROOT)/prepare_tflite.sh $(TENSORFLOW_VERSION) $(NNTRAINER_APPLICATION))))

TENSORFLOW_ROOT := $(NNTRAINER_JNI_ROOT)/tensorflow-$(TENSORFLOW_VERSION)/tensorflow-lite

endif #TENSORFLOW_ROOT
endif #MAKECMDGOALS

LOCAL_MODULE := tensorflow-lite
LOCAL_SRC_FILES := $(TENSORFLOW_ROOT)/lib/arm64/libtensorflow-lite.a
LOCAL_C_INCLUDES := $(TENSORFLOW_ROOT)/include
TFLITE_INCLUDES := $(LOCAL_C_INCLUDES)

include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
endif #ENABLE_TFLITE_BACKBONE

NNTRAINER_SRCS := $(NNTRAINER_ROOT)/nntrainer/src/neuralnet.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/tensor.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/lazy_tensor.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/layer_factory.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/input_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/fc_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/bn_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/loss_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/databuffer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/databuffer_factory.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/databuffer_func.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/databuffer_file.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/util_func.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/optimizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/parse_util.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/tensor_dim.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/conv2d_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/pooling2d_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/activation_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/flatten_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/model_loader.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/addition_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/concat_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/blas_interface.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/weight.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/adam.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/sgd.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/src/optimizer_factory.cpp

# Add tflite backbone building
ifeq ($(ENABLE_TFLITE_BACKBONE),1)
NNTRAINER_SRCS += $(NNTRAINER_ROOT)/nntrainer/src/tflite_layer.cpp
endif #ENABLE_TFLITE_BACKBONE

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/include \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include \
                      $(NNTRAINER_ROOT)/api/capi/include/platform

INIPARSER_SRCS := $(INIPARSER_ROOT)/src/iniparser.c \
                  $(INIPARSER_ROOT)/src/dictionary.c

INIPARSER_INCLUDES := $(INIPARSER_ROOT)/src

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fopenmp -fexceptions
LOCAL_CXXFLAGS      += -std=c++14 -frtti -fexceptions
LOCAL_LDFLAGS       += -fuse-ld=bfd -fopenmp
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog

LOCAL_MODULE        := nntrainer
LOCAL_SRC_FILES     := $(NNTRAINER_SRCS) $(INIPARSER_SRCS)
LOCAL_C_INCLUDES    := $(NNTRAINER_INCLUDES) $(INIPARSER_INCLUDES)

# Add tflite backbone building
ifeq ($(ENABLE_TFLITE_BACKBONE),1)
LOCAL_STATIC_LIBRARIES := tensorflow-lite
LOCAL_C_INCLUDES += $(TFLITE_INCLUDES)
LOCAL_CFLAGS += -DENABLE_TFLITE_BACKBONE=1
endif #ENABLE_TFLITE_BACKBONE

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

CAPI_NNTRAINER_SRCS := $(NNTRAINER_ROOT)/api/capi/src/nntrainer.cpp \
                  $(NNTRAINER_ROOT)/api/capi/src/nntrainer_util.cpp

CAPI_NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/include \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include \
                      $(NNTRAINER_ROOT)/api/capi/include \
                      $(NNTRAINER_ROOT)/api/capi/include/platform

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fopenmp -fexceptions
LOCAL_CXXFLAGS      += -std=c++14 -frtti -fexceptions
LOCAL_LDFLAGS       += -fuse-ld=bfd -fopenmp
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog

LOCAL_MODULE        := capi-nntrainer
LOCAL_SRC_FILES     := $(CAPI_NNTRAINER_SRCS)
LOCAL_C_INCLUDES    := $(CAPI_NNTRAINER_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

CCAPI_NNTRAINER_SRCS := $(NNTRAINER_ROOT)/api/ccapi/src/factory.cpp

CCAPI_NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/include \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include \
                      $(NNTRAINER_ROOT)/api/capi/include/platform

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fopenmp -fexceptions
LOCAL_CXXFLAGS      += -std=c++14 -frtti -fexceptions
LOCAL_LDFLAGS       += -fuse-ld=bfd -fopenmp
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog

LOCAL_MODULE        := ccapi-nntrainer
LOCAL_SRC_FILES     := $(CCAPI_NNTRAINER_SRCS)
LOCAL_C_INCLUDES    := $(CCAPI_NNTRAINER_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

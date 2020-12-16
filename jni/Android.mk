# ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j2
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ENABLE_TFLITE_BACKBONE := 1

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/..
endif

ifndef NDK_LIBS_OUT
NDK_LIBS_OUT := $(NDK_PROJECT_PATH)
endif

include $(CLEAR_VARS)

ifndef INIPARSER_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning INIPARSER_ROOT is not defined!)
$(warning INIPARSER SRC is going to be downloaded!)

INIPARSER_ROOT :=$(NDK_LIBS_OUT)/iniparser

$(info $(shell ($(LOCAL_PATH)/prepare_iniparser.sh $(NDK_LIBS_OUT))))

endif
endif


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

TENSORFLOW_ROOT := $(NDK_LIBS_OUT)/tensorflow-$(TENSORFLOW_VERSION)/tensorflow-lite

$(info $(shell ($(NNTRAINER_JNI_ROOT)/prepare_tflite.sh $(TENSORFLOW_VERSION) $(NDK_LIBS_OUT))))

endif #MAKECMDGOALS
endif #TENSORFLOW_ROOT

LOCAL_MODULE := tensorflow-lite
LOCAL_SRC_FILES := $(TENSORFLOW_ROOT)/lib/arm64/libtensorflow-lite.a
LOCAL_EXPORT_C_INCLUDES := $(TENSORFLOW_ROOT)/include
TFLITE_INCLUDES := $(LOCAL_C_INCLUDES)

include $(PREBUILT_STATIC_LIBRARY)

endif #ENABLE_TFLITE_BACKBONE


ifeq ($(ENABLE_BLAS), 1)
include $(CLEAR_VARS)

## prepare openblas if nothing present
ifndef OPENBLAS_ROOT
ifneq ($(MAKECMDGOALS),clean)

OPENBLAS_ROOT := $(NDK_LIBS_OUT)/openblas
$(info $(shell $(NNTRAINER_JNI_ROOT)/prepare_openblas.sh $(NDK_LIBS_OUT)))

endif #MAKECMDGOALS
endif #OPENBLAS_ROOT

LOCAL_MODULE := openblas
LOCAL_SRC_FILES := $(OPENBLAS_ROOT)/lib/libopenblas.a
LOCAL_EXPORT_C_INCLUDES := $(OPENBLAS_ROOT)/include
LOCAL_EXPORT_CFLAGS += -DUSE_BLAS=1

include $(PREBUILT_STATIC_LIBRARY)
endif #ENABLE_BLAS


include $(CLEAR_VARS)

NNTRAINER_SRCS := $(NNTRAINER_ROOT)/nntrainer/models/neuralnet.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/models/model_loader.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/databuffer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/databuffer_factory.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/databuffer_func.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/databuffer_file.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/tensor.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/lazy_tensor.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/manager.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/var_grad.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/weight.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/tensor_dim.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/blas_interface.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/layer_factory.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/input_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/output_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/fc_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/bn_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/loss_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/conv2d_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/pooling2d_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/activation_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/flatten_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/addition_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/concat_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/graph/network_graph.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/optimizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/adam.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/sgd.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/optimizer_factory.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/utils/util_func.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/utils/parse_util.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/utils/profiler.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/app_context.cpp

# Add tflite backbone building
ifeq ($(ENABLE_TFLITE_BACKBONE),1)
NNTRAINER_SRCS += $(NNTRAINER_ROOT)/nntrainer/layers/tflite_layer.cpp
endif #ENABLE_TFLITE_BACKBONE

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
                      $(NNTRAINER_ROOT)/nntrainer/dataset \
                      $(NNTRAINER_ROOT)/nntrainer/layers \
                      $(NNTRAINER_ROOT)/nntrainer/models \
                      $(NNTRAINER_ROOT)/nntrainer/tensor \
                      $(NNTRAINER_ROOT)/nntrainer/optimizers \
                      $(NNTRAINER_ROOT)/nntrainer/utils \
                      $(NNTRAINER_ROOT)/nntrainer/graph \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include \
                      $(NNTRAINER_ROOT)/api/capi/include/platform

INIPARSER_SRCS := $(INIPARSER_ROOT)/src/iniparser.c \
                  $(INIPARSER_ROOT)/src/dictionary.c

INIPARSER_INCLUDES := $(INIPARSER_ROOT)/src

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fexceptions
LOCAL_CXXFLAGS      += -std=c++14 -frtti -fexceptions
LOCAL_LDFLAGS       += -fuse-ld=bfd
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog -landroid

LOCAL_MODULE        := nntrainer
LOCAL_SRC_FILES     := $(NNTRAINER_SRCS) $(INIPARSER_SRCS)
LOCAL_C_INCLUDES    := $(NNTRAINER_INCLUDES) $(INIPARSER_INCLUDES)

# Add tflite backbone building
ifeq ($(ENABLE_TFLITE_BACKBONE),1)
LOCAL_STATIC_LIBRARIES += tensorflow-lite
LOCAL_C_INCLUDES += $(TFLITE_INCLUDES)
LOCAL_CFLAGS += -DENABLE_TFLITE_BACKBONE=1
endif #ENABLE_TFLITE_BACKBONE

# Enable Profile
ifeq ($(ENABLE_PROFILE), 1)
LOCAL_CFLAGS += -DPROFILE=1
endif #ENABLE_PROFILE

ifeq ($(ENABLE_BLAS), 1)
LOCAL_STATIC_LIBRARIES += openblas
endif #ENABLE_BLAS

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

CAPI_NNTRAINER_SRCS := $(NNTRAINER_ROOT)/api/capi/src/nntrainer.cpp \
                  $(NNTRAINER_ROOT)/api/capi/src/nntrainer_util.cpp

CAPI_NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
                      $(NNTRAINER_ROOT)/nntrainer/dataset \
                      $(NNTRAINER_ROOT)/nntrainer/layers \
                      $(NNTRAINER_ROOT)/nntrainer/models \
                      $(NNTRAINER_ROOT)/nntrainer/graph \
                      $(NNTRAINER_ROOT)/nntrainer/tensor \
                      $(NNTRAINER_ROOT)/nntrainer/optimizers \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include \
                      $(NNTRAINER_ROOT)/api/capi/include \
                      $(NNTRAINER_ROOT)/api/capi/include/platform

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fexceptions
LOCAL_CXXFLAGS      += -std=c++14 -frtti -fexceptions
LOCAL_LDFLAGS       += -fuse-ld=bfd
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog -landroid

LOCAL_MODULE        := capi-nntrainer
LOCAL_SRC_FILES     := $(CAPI_NNTRAINER_SRCS)
LOCAL_C_INCLUDES    := $(CAPI_NNTRAINER_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

CCAPI_NNTRAINER_SRCS := $(NNTRAINER_ROOT)/api/ccapi/src/factory.cpp

CCAPI_NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
                      $(NNTRAINER_ROOT)/nntrainer/dataset \
                      $(NNTRAINER_ROOT)/nntrainer/layers \
                      $(NNTRAINER_ROOT)/nntrainer/models \
                      $(NNTRAINER_ROOT)/nntrainer/tensor \
                      $(NNTRAINER_ROOT)/nntrainer/graph \
                      $(NNTRAINER_ROOT)/nntrainer/optimizers \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include \
                      $(NNTRAINER_ROOT)/api/capi/include/platform

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fexceptions
LOCAL_CXXFLAGS      += -std=c++14 -frtti -fexceptions
LOCAL_LDFLAGS       += -fuse-ld=bfd
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog -landroid

LOCAL_MODULE        := ccapi-nntrainer
LOCAL_SRC_FILES     := $(CCAPI_NNTRAINER_SRCS)
LOCAL_C_INCLUDES    := $(CCAPI_NNTRAINER_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

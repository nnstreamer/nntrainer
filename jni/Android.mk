# ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j2
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ENABLE_TFLITE_BACKBONE := 1
ENABLE_TFLITE_INTERPRETER := 1
ENABLE_BLAS := 1

NEED_TF_LITE := 0

ifeq ($(ENABLE_TFLITE_BACKBONE), 1)
NEED_TF_LITE := 1
else ifeq ($(ENABLE_TFLITE_INTERPRETER), 1)
NEED_TF_LITE := 1
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/..
endif

ifndef NDK_LIBS_OUT
NDK_LIBS_OUT := $(NDK_PROJECT_PATH)
endif

ifndef NDK_INCLUDES_OUT
NDK_INCLUDES_OUT := $(NDK_PROJECT_PATH)
endif

include $(CLEAR_VARS)

ifndef INIPARSER_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning INIPARSER_ROOT is not defined!)
$(warning INIPARSER SRC is going to be downloaded!)

INIPARSER_ROOT :=$(NDK_LIBS_OUT)/iniparser

$(info $(shell ($(LOCAL_PATH)/prepare_iniparser.sh $(NDK_LIBS_OUT))))

endif #MAKECMDGOALS
endif #INIPARSER_ROOT

include $(CLEAR_VARS)

NNTRAINER_JNI_ROOT := $(NNTRAINER_ROOT)/jni

# Build tflite if its backbone is enabled
ifeq ($(NEED_TF_LITE),1)
$(warning BUILDING TFLITE BACKBONE !)
TENSORFLOW_VERSION := 2.3.0

ifndef TENSORFLOW_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning TENSORFLOW_ROOT is not defined!)
$(warning TENSORFLOW SRC is going to be downloaded!)

TENSORFLOW_ROOT := $(NDK_LIBS_OUT)/tensorflow-$(TENSORFLOW_VERSION)/tensorflow-lite

$(info $(shell ($(NNTRAINER_JNI_ROOT)/prepare_tflite.sh $(TENSORFLOW_VERSION) $(NDK_LIBS_OUT))))

$(info $(shell (flatc -c $(NNTRAINER_ROOT)/nntrainer/compiler/tf_schema.fbs)))
$(info $(shell (mv tf_schema_generated.h $(NNTRAINER_ROOT)/nntrainer/compiler)))

endif #MAKECMDGOALS
endif #TENSORFLOW_ROOT

LOCAL_MODULE := tensorflow-lite
LIB_ := arm64

ifeq ($(APP_ABI), armeabi-v7a)
  LIB_ := armv7
endif
LOCAL_SRC_FILES := $(TENSORFLOW_ROOT)/lib/$(LIB_)/libtensorflow-lite.a
LOCAL_EXPORT_C_INCLUDES := $(TENSORFLOW_ROOT)/include
LOCAL_EXPORT_LDLIBS := -lEGL -lGLESv2

include $(PREBUILT_STATIC_LIBRARY)

endif #NEED_TF_LITE

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
include $(CLEAR_VARS)

endif #ENABLE_BLAS

## prepare ml common api if nothing present
ifndef ML_API_COMMON_ROOT
ifneq ($(MAKECMDGOALS),clean)

ML_API_COMMON_ROOT := $(NDK_INCLUDES_OUT)/ml_api_common
$(info $(shell ($(NNTRAINER_JNI_ROOT)/prepare_ml-api.sh $(ML_API_COMMON_ROOT))))

endif #MAKECMDGOALS
endif #ML_API_COMMON_ROOT

ML_API_COMMON_INCLUDES := $(ML_API_COMMON_ROOT)/include

LOCAL_MODULE := ml-api-inference
LOCAL_SRC_FILES := $(ML_API_COMMON_ROOT)/lib/arm64-v8a/libnnstreamer-native.so
LOCAL_EXPORT_C_INCLUDES := $(ML_API_COMMON_ROOT)/include
LOCAL_EXPORT_CFLAGS += -DUSE_BLAS=1

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

NNTRAINER_SRCS := $(NNTRAINER_ROOT)/nntrainer/models/neuralnet.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/models/model_loader.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/models/model_common_properties.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/models/dynamic_training_optimization.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/iteration_queue.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/databuffer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/data_iteration.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/databuffer_factory.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/func_data_producer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/random_data_producers.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/dataset/raw_file_data_producer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/tensor.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/lazy_tensor.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/manager.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/var_grad.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/weight.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/tensor_dim.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/tensor_pool.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/memory_pool.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/basic_planner.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/optimized_v1_planner.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/tensor/blas_interface.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/layer_node.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/layer_context.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/input_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/multiout_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/fc_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/bn_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/loss/loss_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/loss/mse_loss_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/loss/kld_loss_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/loss/cross_entropy_sigmoid_loss_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/loss/cross_entropy_softmax_loss_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/loss/constant_derivative_loss_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/conv2d_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/conv1d_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/pooling2d_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/activation_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/flatten_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/reshape_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/addition_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/attention_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/mol_attention_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/concat_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/preprocess_flip_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/preprocess_translate_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/preprocess_l2norm_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/embedding.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/rnn.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/rnncell.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/lstm.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/lstmcell.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/gru.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/grucell.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/time_dist.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/dropout.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/permute_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/centroid_knn.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/acti_func.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/split_layer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/common_properties.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/layers/layer_impl.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/graph/network_graph.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/graph/graph_core.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/graph/connection.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/optimizer_context.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/optimizer_devel.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/optimizer_impl.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/adam.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/optimizers/sgd.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/utils/util_func.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/utils/ini_wrapper.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/utils/profiler.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/utils/node_exporter.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/utils/base_properties.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/ini_interpreter.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/flatten_realizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/activation_realizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/recurrent_realizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/previous_input_realizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/multiout_realizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/remap_realizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/slice_realizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/input_realizer.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/app_context.cpp

ifeq ($(ENABLE_TFLITE_INTERPRETER), 1)
NNTRAINER_SRCS += $(NNTRAINER_ROOT)/nntrainer/compiler/tflite_opnode.cpp \
                  $(NNTRAINER_ROOT)/nntrainer/compiler/tflite_interpreter.cpp
endif #ENABLE_TFLITE_INTERPRETER

# Add tflite backbone building
ifeq ($(ENABLE_TFLITE_BACKBONE),1)
NNTRAINER_SRCS += $(NNTRAINER_ROOT)/nntrainer/layers/tflite_layer.cpp
endif #ENABLE_TFLITE_BACKBONE

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
                      $(NNTRAINER_ROOT)/nntrainer/dataset \
                      $(NNTRAINER_ROOT)/nntrainer/layers \
                      $(NNTRAINER_ROOT)/nntrainer/layers/loss \
                      $(NNTRAINER_ROOT)/nntrainer/models \
                      $(NNTRAINER_ROOT)/nntrainer/tensor \
                      $(NNTRAINER_ROOT)/nntrainer/optimizers \
                      $(NNTRAINER_ROOT)/nntrainer/utils \
                      $(NNTRAINER_ROOT)/nntrainer/graph \
                      $(NNTRAINER_ROOT)/nntrainer/utils \
                      $(NNTRAINER_ROOT)/nntrainer/compiler \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include

INIPARSER_SRCS := $(INIPARSER_ROOT)/src/iniparser.c \
                  $(INIPARSER_ROOT)/src/dictionary.c

INIPARSER_INCLUDES := $(INIPARSER_ROOT)/src

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fexceptions -fopenmp
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog -landroid -fopenmp

LOCAL_MODULE        := nntrainer
LOCAL_SRC_FILES     := $(NNTRAINER_SRCS) $(INIPARSER_SRCS)
LOCAL_C_INCLUDES    := $(NNTRAINER_INCLUDES) $(INIPARSER_INCLUDES) $(ML_API_COMMON_INCLUDES)

# Add tflite backbone building
ifeq ($(ENABLE_TFLITE_BACKBONE),1)
LOCAL_STATIC_LIBRARIES += tensorflow-lite
LOCAL_CFLAGS += -DENABLE_TFLITE_BACKBONE=1
endif #ENABLE_TFLITE_BACKBONE

ifeq ($(ENABLE_TFLITE_INTERPRETER), 1)
LOCAL_CFLAGS += -DENABLE_TFLITE_INTERPRETER
endif #ENABLE_TFLITE_INTERPRETER

# Enable Profile
ifeq ($(ENABLE_PROFILE), 1)
LOCAL_CFLAGS += -DPROFILE=1
endif #ENABLE_PROFILE

ifeq ($(ENABLE_BLAS), 1)
LOCAL_STATIC_LIBRARIES += openblas
endif #ENABLE_BLAS

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

CCAPI_NNTRAINER_SRCS := $(NNTRAINER_ROOT)/api/ccapi/src/factory.cpp

CCAPI_NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
                      $(NNTRAINER_ROOT)/nntrainer/dataset \
                      $(NNTRAINER_ROOT)/nntrainer/compiler \
                      $(NNTRAINER_ROOT)/nntrainer/layers \
                      $(NNTRAINER_ROOT)/nntrainer/models \
                      $(NNTRAINER_ROOT)/nntrainer/tensor \
                      $(NNTRAINER_ROOT)/nntrainer/graph \
                      $(NNTRAINER_ROOT)/nntrainer/optimizers \
                      $(NNTRAINER_ROOT)/nntrainer/utils \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fexceptions -fopenmp
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog -landroid -fopenmp

LOCAL_MODULE        := ccapi-nntrainer
LOCAL_SRC_FILES     := $(CCAPI_NNTRAINER_SRCS)
LOCAL_C_INCLUDES    := $(CCAPI_NNTRAINER_INCLUDES) $(ML_API_COMMON_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

CAPI_NNTRAINER_SRCS := $(NNTRAINER_ROOT)/api/capi/src/nntrainer.cpp

CAPI_NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
                      $(NNTRAINER_ROOT)/api \
                      $(NNTRAINER_ROOT)/api/ccapi/include \
                      $(NNTRAINER_ROOT)/api/capi/include

LOCAL_SHARED_LIBRARIES := ccapi-nntrainer ml-api-inference nntrainer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fexceptions -fopenmp
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog -landroid -fopenmp

LOCAL_MODULE        := capi-nntrainer
LOCAL_SRC_FILES     := $(CAPI_NNTRAINER_SRCS)
LOCAL_C_INCLUDES    := $(CAPI_NNTRAINER_INCLUDES) $(ML_API_COMMON_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

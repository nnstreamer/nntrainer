LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

ML_API_COMMON_INCLUDES := ${NNTRAINER_ROOT}/ml_api_common/include

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/include \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	${ML_API_COMMON_INCLUDES}

# Prebuilt nntrainer libraries
include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

# Tokenizer library
include $(CLEAR_VARS)
LOCAL_MODULE := tokenizers_c
LOCAL_SRC_FILES := ../lib/libtokenizers_c.a
include $(PREBUILT_STATIC_LIBRARY)

# Build CausalLM executable
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -fopenmp -static-openmp
LOCAL_LDFLAGS += -fexceptions
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_causallm
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp

# Source files
LOCAL_SRC_FILES := ../main.cpp \
    ../causal_lm.cpp \
    ../qwen3_causallm.cpp \
    ../qwen3_moe_causallm.cpp \
    ../huggingface_tokenizer.cpp \
    ../llm_util.cpp \
    ../layers/embedding_layer.cpp \
    ../layers/mha_core.cpp \
    ../layers/qwen_moe_layer.cpp \
    ../layers/reshaped_rms_norm.cpp \
    ../layers/rms_norm.cpp \
    ../layers/swiglu.cpp \
    ../layers/tie_word_embedding.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) \
    $(LOCAL_PATH)/.. \
    $(LOCAL_PATH)/../layers \
    $(NNTRAINER_ROOT)/nntrainer \
    $(NNTRAINER_ROOT)/nntrainer/layers \
    $(NNTRAINER_ROOT)/nntrainer/models \
    $(NNTRAINER_ROOT)/nntrainer/dataset \
    $(NNTRAINER_ROOT)/nntrainer/optimizers \
    $(NNTRAINER_ROOT)/nntrainer/tensor \
    $(NNTRAINER_ROOT)/nntrainer/utils

include $(BUILD_EXECUTABLE)
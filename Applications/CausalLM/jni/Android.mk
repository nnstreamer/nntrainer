LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# Tokenizer library
LOCAL_MODULE := tokenizers_c
LOCAL_SRC_FILES := ../lib/libtokenizers_c.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

# CausalLM library
LOCAL_MODULE := causallm
LOCAL_MODULE_TAGS := optional

# Source files for CausalLM library
LOCAL_SRC_FILES := \
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

# Include directories
LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/.. \
    $(LOCAL_PATH)/../layers \
    $(LOCAL_PATH)/../../../nntrainer \
    $(LOCAL_PATH)/../../../nntrainer/layers \
    $(LOCAL_PATH)/../../../nntrainer/models \
    $(LOCAL_PATH)/../../../nntrainer/dataset \
    $(LOCAL_PATH)/../../../nntrainer/optimizers \
    $(LOCAL_PATH)/../../../nntrainer/tensor \
    $(LOCAL_PATH)/../../../nntrainer/utils \
    $(LOCAL_PATH)/../../../api \
    $(LOCAL_PATH)/../../../api/ccapi/include \
    $(LOCAL_PATH)/../../../api/capi/include/platform

# Compiler flags
LOCAL_CFLAGS := -pthread -fexceptions -fopenmp -static-openmp -DPLUGGABLE
LOCAL_CXXFLAGS := -std=c++17 -frtti -fexceptions -fopenmp -static-openmp -DPLUGGABLE
LOCAL_ARM_NEON := true

# Link flags
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp
LOCAL_LDFLAGS += "-Wl,-z,max-page-size=16384"

# Dependencies
LOCAL_STATIC_LIBRARIES := tokenizers_c
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

# CausalLM executable
LOCAL_MODULE := nntr_causallm
LOCAL_MODULE_TAGS := optional

# Source files
LOCAL_SRC_FILES := ../main.cpp

# Include directories
LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/.. \
    $(LOCAL_PATH)/../layers \
    $(LOCAL_PATH)/../../../nntrainer \
    $(LOCAL_PATH)/../../../api/ccapi/include \
    $(LOCAL_PATH)/../../../api/capi/include/platform

# Compiler flags
LOCAL_CFLAGS := -pthread -fexceptions
LOCAL_CXXFLAGS := -std=c++17 -frtti -fexceptions

# Dependencies
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer causallm
LOCAL_LDLIBS := -llog -landroid

include $(BUILD_EXECUTABLE)
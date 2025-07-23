LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

prepare_tokenizer := $(shell ($(LOCAL_PATH)/prepare_tokenizer.sh))

NNTRAINER_ROOT := $(LOCAL_PATH)/nntrainer

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(LOCAL_PATH)/nntrainer/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/nntrainer/include/nntrainer

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(LOCAL_PATH)/nntrainer/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/nntrainer/include/nntrainer

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := causallm
LOCAL_SRC_FILES := $(LOCAL_PATH)/nntrainer/lib/$(TARGET_ARCH_ABI)/libcausallm.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/nntrainer/include/nntrainer

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := tokenizers_c
LOCAL_SRC_FILES := $(LOCAL_PATH)/lib/$(TARGET_ARCH_ABI)/libtokenizers_c.a

include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/include/nntrainer
CAUSALLM_DIR = .

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53
LOCAL_CXXFLAGS += -std=c++17 -frtti -fexceptions
LOCAL_CFLAGS += -pthread -fexceptions -fopenmp -static-openmp
LOCAL_LDFLAGS += -fexceptions -fopenmp -static-openmp
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := causallm_jni
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp -ljnigraphics

LOCAL_SRC_FILES := causallm_jni.cpp
LOCAL_SHARED_LIBRARIES := ccapi-nntrainer nntrainer causallm
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(CAUSALLM_DIR)

include $(BUILD_SHARED_LIBRARY)
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/include \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/capi/include/platform

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++14 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++14
LOCAL_CFLAGS += -pthread -fopenmp -fexceptions
LOCAL_LDFLAGS += -fopenmp -fexceptions
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_logistic
LOCAL_LDLIBS := -llog

LOCAL_SRC_FILES := main.cpp

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)

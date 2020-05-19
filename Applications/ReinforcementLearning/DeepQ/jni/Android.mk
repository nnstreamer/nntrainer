LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../../../libs/arm64-v8a
NNTRAINER_INCLUDES := $(LOCAL_PATH)/../../../../nntrainer/include
endif

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

ENVDIR=../../Environment
NEURALNET=../../NeuralNet

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++11 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -fexceptions -DUSING_CUSTOM_ENV
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/arm64-v8a/
LOCAL_CXXFLAGS += -std=c++11 -DUSING_CUSTOM_ENV
LOCAL_CFLAGS += -pthread -fopenmp
LOCAL_LDFLAGS += -fopenmp 
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_deepq
LOCAL_LDLIBS := -llog

LOCAL_SRC_FILES := main.cpp $(ENVDIR)/CartPole/cartpole.cpp

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_C_INCLUDES += $(ENVDIR) $(LOCAL_PATH)/include $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)

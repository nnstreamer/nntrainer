LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

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
LOCAL_MODULE := DeepQ

LOCAL_SRC_FILES := main.cpp $(NEURALNET)/matrix.cpp $(NEURALNET)/neuralnet.cpp \
		   $(ENVDIR)/CartPole/cartpole.cpp

LOCAL_C_INCLUDES += $(ENVDIR) $(LOCAL_PATH)/include $(NEURALNET)

include $(BUILD_EXECUTABLE)

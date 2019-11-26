LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

INIPARSER=../../iniparser/src/
LOCAL_MODULE :=iniparser
INIPARSER_SRCS := \
	$(INIPARSER)/iniparser.c \
	$(INIPARSER)/dictionary.c

LOCAL_SRC_FILES :=$(INIPARSER_SRCS)
LOCAL_C_INCLUDES := $(INIPARSER)

LOCAL_CFLAGS += -O3 -DNDEBUG

include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
NEURALNET=../../NeuralNet
LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++11 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/arm64-v8a/
LOCAL_CXXFLAGS += -std=c++11
LOCAL_CFLAGS += -pthread -fopenmp
LOCAL_LDFLAGS += -fopenmp 
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := LogisticRegressoin

LOCAL_SRC_FILES := main.cpp  $(NEURALNET)/matrix.cpp $(NEURALNET)/neuralnet.cpp \
		   $(NEURALNET)/layers.cpp

LOCAL_STATIC_LIBRARIES := iniparser

LOCAL_C_INCLUDES += $(INIPARSER) $(NEURALNET)

include $(BUILD_EXECUTABLE)

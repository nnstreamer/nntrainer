LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../../libs/arm64-v8a
NNTRAINER_INCLUDES := $(LOCAL_PATH)/../../../nntrainer/include
endif

NNTRAINER_APPLICATION := $(LOCAL_PATH)/../..

include $(CLEAR_VARS)

TENSORFLOW_VERSION := 1.13.1

ifndef TENSORFLOW_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning TENSORFLOW_ROOT is not defined!)
$(warning TENSORFLOW SRC is going to be downloaded!)

$(info $(shell ($(NNTRAINER_APPLICATION)/prepare_tflite.sh $(TENSORFLOW_VERSION) $(NNTRAINER_APPLICATION))))

TENSORFLOW_ROOT := $(LOCAL_PATH)/../../tensorflow-$(TENSORFLOW_VERSION)/tensorflow-lite

endif
endif

LOCAL_MODULE := tensorflow-lite
LOCAL_SRC_FILES := $(TENSORFLOW_ROOT)/lib/arm64/libtensorflow-lite.a
LOCAL_EXPORT_C_INCLUDES := $(TENSORFLOW_ROOT)/include

include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++11 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/arm64-v8a/
LOCAL_CXXFLAGS += -std=c++11
LOCAL_CFLAGS += -pthread -fopenmp
LOCAL_LDFLAGS += -fopenmp 
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_classification
LOCAL_LDLIBS := -llog

LOCAL_SRC_FILES := main.cpp bitmap_helpers.cpp

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_STATIC_LIBRARIES := tensorflow-lite

LOCAL_C_INCLUDES += $(TFLITE_INCLUDES) $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++11 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/arm64-v8a/
LOCAL_CXXFLAGS += -std=c++11
LOCAL_CFLAGS += -pthread -fopenmp
LOCAL_LDFLAGS += -fopenmp 
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_classification_func
LOCAL_LDLIBS := -llog

LOCAL_SRC_FILES := main_func.cpp

LOCAL_SHARED_LIBRARIES := nntrainer

LOCAL_STATIC_LIBRARIES := tensorflow-lite

LOCAL_C_INCLUDES += $(TFLITE_INCLUDES) $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)

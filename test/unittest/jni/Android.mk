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
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	${ML_API_COMMON_INCLUDES}

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
GTEST_PATH := tests/googletest

LOCAL_MODULE := googletest_main
LOCAL_CFLAGS := -Itests/googletest/include -Itests/googletest/
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions

LOCAL_SRC_FILES := \
    $(GTEST_PATH)/src/gtest-all.cc

include $(BUILD_STATIC_LIBRARY)


include $(CLEAR_VARS)

LOCAL_MODULE := tensor_unittest
LOCAL_CFLAGS := -Itests/googletest/include -Itests -pthread -fexceptions -fopenmp -static-openmp -DMIN_CPP_VERSION=201703L -DNNTR_NUM_THREADS=1 -D__LOGGING__=1 -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 -march=armv8.2-a+fp16 -O3 -frtti
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS        := -llog -landroid -fopenmp -static-openmp 

LOCAL_SRC_FILES := \
    tests/unittest_nntrainer_tensor.cpp \
    tests/unittest_nntrainer_tensor_fp16.cpp \
    tests/nntrainer_test_util.cpp

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main
include $(BUILD_EXECUTABLE)

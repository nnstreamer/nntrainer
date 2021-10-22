LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer

LOCAL_SRC_FILES := $(LOCAL_PATH)/lib/arm64-v8a/libccapi-nntrainer.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include

include $(PREBUILT_SHARED_LIBRARY)

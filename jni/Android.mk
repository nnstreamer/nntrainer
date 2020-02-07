LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/..
endif

include $(CLEAR_VARS)

INIPARSER_ROOT :=$(NNTRAINER_ROOT)/external/iniparser

NNTRAINER_SRCS := $(NNTRAINER_ROOT)/src/neuralnet.cpp \
                  $(NNTRAINER_ROOT)/src/matrix.cpp \
                  $(NNTRAINER_ROOT)/src/layers.cpp

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)

INIPARSER_SRCS := $(INIPARSER_ROOT)/src/iniparser.c \
                  $(INIPARSER_ROOT)/src/dictionary.c

INIPARSER_INCLUDES := $(INIPARSER_ROOT)/src

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fopenmp
LOCAL_LDFLAGS       += -fuse-ld=bfd
LOCAL_MODULE_TAGS   := optional

LOCAL_MODULE        := nntrainer
LOCAL_SRC_FILES     := $(NNTRAINER_SRCS) $(INIPARSER_SRCS)
LOCAL_C_INCLUDES    += $(NNTRAINER_INCLUDES) $(INIPARSER_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

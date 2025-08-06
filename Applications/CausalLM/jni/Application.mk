APP_ABI := arm64-v8a
APP_PLATFORM := android-35
APP_STL := c++_shared
APP_CPPFLAGS += -fexceptions -frtti -std=c++20
APP_LDFLAGS += -fopenmp
NDK_TOOLCHAIN_VERSION := clang

# resnet - java

This project is to demonstrate how nntrainer is able to work in Java for Android Application.

The training model is Resnet18 in Applications/Resnet and input generator is fake (Random) generator in utils.
Real data generator and test are left as todo.

It takes the pretrained model as pretrained_resnet18.bin, and the output models after trainging are finetuned_resnet18.bin

## How to build
1. Move to jni directory
``` bash
cd ANDROID_PROJECT_ROOT/app/src/main/jni
```

2. Download nntrainer headers and so files by run prepare_android_deps.sh script
``` bash
./prepare_android_deps.sh
```

3. Run ndk-build and set output directory as jniLibs by
``` bash
ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j 8 NDK_LIBS_OUT=../jniLibs
```


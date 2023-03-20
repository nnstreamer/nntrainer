# kotlin example for nntrainer

In order to train Resnet network model, nntrainer is used.
Please be aware, this is simple demo and you need to improve for your own purpose such as error exception and remove hard codded string, parameters, save file format, etc.


### Build
Inorder to build, you need nntrainer setup.
please see https://github.com/nntstreamer/nntrainer/tree/main/docs

Also you need to set NNTRAINER_ROOT parameter.

``` bash
export NNTRAINER_ROOT=$HOME/nntrainer
```

Also it is looking for the libs dir in NNTRAINER_ROOT. So you need to copy the libs dir to $NNTRAINER_ROOT after android build.

In order to build nntrainer for the android, you just go to $NNTRAINER_ROOT/jni and run ndk-build. ndk-build or others need to be set before run ndk-build.

``` bash
cd $NNTRAINER_ROOT/jni
ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j 8
```

After compile, you need to copy the includes and libs in kotlin/app/src/main/jni/nntrainer.

or,

you just run

```bash
$ ./prepare_android.sh

...

[arm64-v8a] Install        : libccapi-nntrainer.so => libs/arm64-v8a/libccapi-nntrainer.so
[arm64-v8a] Install        : libnntrainer.so => libs/arm64-v8a/libnntrainer.so
[arm64-v8a] Compile++      : nntrainer_resnet <= NNTrainer.cpp
[arm64-v8a] Compile++      : nntrainer_resnet <= cifar_dataloader.cpp
[arm64-v8a] SharedLibrary  : libnntrainer_resnet.so
[arm64-v8a] Install        : libnntrainer_resnet.so => libs/arm64-v8a/libnntrainer_resnet.so
[arm64-v8a] Install        : libc++_shared.so => libs/arm64-v8a/libc++_shared.so

```


then, you can build nntrainer_kotlin as in:

``` bash
cd $NNTRAINER_KOTLINE_HOME
./gradlew build
```


---
title: Android NNtrainer Applicaiton Sample
...

# Adnroid NNtrainer Application Sample
This is a pratical demonstration of Android NNTrainer Resnet Application with [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html) data.

## How to run
Build nntrainer with `${NNTRAINER_HOME}/tools/package_android.sh` as in [Document](https://github.com/nnstreamer/nntrainer/blob/main/docs/how-to-run-example-android.md)

```bash
$ ls
api                 CONTRIBUTING.md  index.md  MAINTAINERS.md     nnstreamer        nntrainer.pc.in  RELEASE.md
Applications        debian           jni       meson.build        nntrainer         packaging        test
CODE_OF_CONDUCT.md  docs             LICENSE   meson_options.txt  nntrainer.ini.in  README.md        tools
$
$ ./tools/package_android.sh
$ ls builddir/android_build_result
Android.mk  conf  examples  include  lib
$ ls builddir/android_build_result/libs/arm64-v8a
libcapi-nntrainer.so  libccapi-nntrainer.so  libc++_shared.so  libnnstreamer-native.so  libnntrainer.so
```

Build the jni with `{$APP_HOME}/app/src/main/jni/prepare_android_deps.sh`
```bash
$cd {$APP_HOME}/app/src/main/jni

./prepare_android_deps.sh 
{$APP_HOME}/ResnetJNI/app/src/main/jni/nntrainer
[arm64-v8a] Prebuilt       : libccapi-nntrainer.so <= jni/nntrainer/lib/arm64-v8a/
[arm64-v8a] Install        : libccapi-nntrainer.so => libs/arm64-v8a/libccapi-nntrainer.so
[arm64-v8a] Prebuilt       : libnntrainer.so <= jni/nntrainer/lib/arm64-v8a/
[arm64-v8a] Install        : libnntrainer.so => libs/arm64-v8a/libnntrainer.so
[arm64-v8a] Compile++      : resnet_jni <= resnet.cpp
[arm64-v8a] Compile++      : resnet_jni <= resnet_jni.cpp
[arm64-v8a] Compile++      : resnet_jni <= dataloader.cpp
[arm64-v8a] Compile++      : resnet_jni <= image.cpp
[arm64-v8a] Prebuilt       : libc++_shared.so <= <NDK>/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/
[arm64-v8a] SharedLibrary  : libresnet_jni.so
[arm64-v8a] Install        : libresnet_jni.so => libs/arm64-v8a/libresnet_jni.so
[arm64-v8a] Install        : libc++_shared.so => libs/arm64-v8a/libc++_shared.so
```

Prepare the training data set. you can download the cifar100 and place it into asset directory.

```bash
$cd {$APP_HOME}/app/src/main/asset
$ls
test  train
$ cd train/
$ ls
apple          bridge       cockroach  hamster     motorcycle    plain      seal          table       willow_tree
aquarium_fish  bus          couch      house       mountain      plate      shark         tank        wolf
baby           butterfly    crab       kangaroo    mouse         poppy      shrew         telephone   woman
bear           camel        crocodile  keyboard    mushroom      porcupine  skunk         television  worm
beaver         can          cup        lamp        oak_tree      possum     skyscraper    tiger
bed            castle       dinosaur   lawn_mower  orange        rabbit     snail         tractor
bee            caterpillar  dolphin    leopard     orchid        raccoon    snake         train
beetle         cattle       elephant   lion        otter         ray        spider        trout
bicycle        chair        flatfish   lizard      palm_tree     road       squirrel      tulip
bottle         chimpanzee   forest     lobster     pear          rocket     streetcar     turtle
bowl           clock        fox        man         pickup_truck  rose       sunflower     wardrobe
boy            cloud        girl       maple_tree  pine_tree     sea        sweet_pepper  whale

```


Build Application with gradlew.

``` bash
$cd {$APP_HOME}
$./gradlew build

> Configure project :app

> Task :app:stripDebugDebugSymbols
Unable to strip the following libraries, packaging them as they are: libc++_shared.so, libccapi-nntrainer.so, libnntrainer.so, libresnet_jni.so.

...

BUILD SUCCESSFUL in 10s
83 actionable tasks: 81 executed, 2 up-to-date

```

Install the application and run

``` bash
$adb install {$APP_HOME}/app/build/outputs/apk/debug/app-debug.apk

```

After run the application, you can run the applicaiton.
![Application](/docs/images/app_resnet.jpg?raw=true)

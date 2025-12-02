# NNDetector
NNDetector is a demo application conducting object detection based on nntrainer.
You can detect an object and train with your personal items. Below is a showing of how to detect and train with NNDetector. The method for identifying personal objects is OCL with high order pooling. It can identify with few labeled samples of new classes.

# How To Build
Build nntrainer
~~~
$cd {$NNTRAINER_HOME}
$./tools/package_android.sh
~~~

Build Continual Learning Module
~~~
$cd {$APP_HOME}/app/src/main/jni
$./prepare_android_deps.sh
~~~

Build and Install App
~~~
$./gradlew build
$adb install app/build/outputs/apk/debug/app-debug.apk
~~~
or
~~~
$./gradlew installDebug
~~~

# How To Use
## Inference and data collecting
![](img/step1.jpg)


## Train and Test
### Train and Test with camera
![](img/step2-1.jpg)

### Test with Image file on Device
![](img/step2-2.jpg)


## Inference and Retry
![](img/step3.jpg)


# Troubleshooting
Crash can come out when you touch screen at Test Phase. Then you need to set "appear on top" on the setting of application.
![](img/appear_on_top.jpg)

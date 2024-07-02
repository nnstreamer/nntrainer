---
title: Test JNI
...

# Test


In order to enable the gtest in android, you need to add the googletest source in ndk in this directory.

please do

``` bash
#cp ${ANDROIND_SDK_HOME}/Sdk/ndk/${NDK_VERSION}/sources/third_party/googletest .
```

and to use android builddir/android_build_result, do 
``` bash
#ln -s ../../builddir/android_build_result ../nntrainer

```




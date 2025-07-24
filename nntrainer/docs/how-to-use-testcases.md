---
title: How to run Test Cases
...


# How to run Test Cases

- Use the built library

- Unit Test

For gtest based test cases (common library)

```bash
$ cd build
$ ninja test
...
```

### run test cases on Android

- In order to run unittest on android, please set the environment following [[docs] how-to-run-example-android.md](how-to-run-example-android.md).
- Then, you can run the unittest on Android as follows:

```
(nntrainer) $ ./tools/android_test.sh
(nntrainer) $ adb shell
(adb) $ cd /data/local/tmp/nntr_android_test
(adb) $ export LD_LIBRARY_PATH=.
(adb) $ ./unittest_layers
```

- For more information, please refer to [tools](../tools/README.md)
- [**Note**] Android unittest script builds NNTrainer to support GPU by default.

### How to run
1. you should compile nntrainer with this page --> [Getting Started](https://github.com/nnstreamer/nntrainer/blob/main/docs/getting-started.md)

2. you need to make Dataset(.dat file) with this page --> [Datagen](https://github.com/nnstreamer/nntrainer/blob/main/Applications/utils/datagen/cifar/How-to-gen-data.md)

Once you compile the codes, you can run with
``` bash
$ cd ${nntrainer_dir}
$ meson ${build_dir}
$ cd ${build_dir}
$ ninja
$ sudo ninja install
$ ./Applications/AlexNet/jni/nntrainer_alex ./res/app/AlexNet/alex.ini {.dat files dir}
```

with 1 epoch 128 batch_size you can get this result
``` bash
path: ../Applications/AlexNet/res//alex_trainingSet.dat
data_size: 10000
path: ../Applications/AlexNet/res//alex_valSet.dat
data_size: 2000
#1/1 - Training Loss: 4.59144 >> [ Accuracy: 44.2708% - Validation Loss : 4.58553 ]
```


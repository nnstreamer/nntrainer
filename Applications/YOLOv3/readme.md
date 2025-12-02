# Getting started

## Installation and build method
Please refer to the https://github.com/nnstreamer/nntrainer/blob/main/docs/getting-started.md link for the NNTrainer installation and meson build.

## Setting dataset and code modification
Folders containing training data must include "images" and "annotations" folders.
We supports BMP format for image, and TXT format for annotation.
In addition, image files should be prepared in the same size.

```bash
...
├─── train_dir
│   └─── images
│      └─── 001.bmp
│      └─── 002.bmp
│           ...
│   └─── annotations
│      └─── 001.txt
│      └─── 002.txt
│           ...
...
```

Annotation data format is "label x_pos y_pos width height".

```bash
3 206 168 100 96
0 12 120 10 5
```

Set the input image size using the IMAGE_HEIGHT_SIZE and IMAGE_WIDTH_SIZE parameters in main.cpp, specify the folder path of the input dataset in the TRAIN_DIR_PATH parameter, and execute BUILD to complete the execution preparation.

## Execute training
After building, execute the build_dir/Applications/YOLOv3/jni/nntrainer_yolov3 file to start training.

## Something to know
In both YOLO v2 and v3, an "uncaught error while running! details:  is not allocated" error message occurs when the program is terminated after training. It does not affect the training result, but debugging is still required.

The inference code is not prepared in this example, so please refer to other application examples.
In addition, It does not support the importing pre-trained models from other frameworks.

This is a pratical demonstration of SimpleShot with [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html) data.

# What is a SimpleShot and What Does This Application Do?
SimpleShot is a nearest neighbor based few-shot learner with a simple
tranformation function.

The application takes number of parameters and do the fewshot learning, inferencing

Reference. [Wang, Yan, et al. 2019](https://arxiv.org/abs/1911.04623)

# How to run
Build nntrainer with `-Denable_application=true`

`$./executable model method train_file validation_file app_path`

- model: either `resnet50` or `conv4`
- methods: one of `UN`, `L2N` or `CL2N`
- train_file: [app_path]/tasks/[train_file] is used for training
- validation_file: [app_path]/tasks/[validation_file] is used for validation
- app_path: root path to refer to resources, if not given, path is set current working directory

# Give-It-A-Go
There is a sample ready for a try run

After building, run with

```
${builddir}/Applications/SimpleShot/simpleshot_runner conv4 L2N tractor:turtle:squirrel:willow_tree:sunflower_20shot_seed:456_train.dat tractor:turtle:squirrel:willow_tree:sunflower_seed:456_test.dat ${repodir}/Applications/SimpleShot
```

This will give you 73.3333% accuracy on the validation data

## Backbone
Sample backbone is `conv4_60classes.tflite`
It is a stack of 4 convolutional network blocks suggested in [Vinyals, Oriol, et al. 2019](https://arxiv.org/abs/1606.04080)
Our backbone is trained with 60 classes extracted from cifar100 dataset for 90 epochs.

## Task
Sample task is a 1/5/10/20-shot example of `tractor`, `turtle`, `squirrel`, `willow_tree`, `sunflowe`, extracted randomly from 40 classes.
Those classes **are not exposed** to the model during training.
For brevirty, test data set is shrinked 15 images from each class (75 total)


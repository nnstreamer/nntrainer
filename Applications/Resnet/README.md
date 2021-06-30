# Resnet with cifar100

This application contains a Resnet18 model and a trainer with cifar100.

Reference. [Kaiming He. 2015](https://arxiv.org/abs/1512.03385)
Reference. [Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

## How to run a train epoch

### To simply run with a fake data.

Once you compile, with `meson`, you can run with `meson test app_resnet18`.
Please file an issue if you have a problem running the example.

```bash
$ meson ${build_dir} -Denable-test=true -Denable-long-test=true
$ meson test app_resnet18 -v -c ${build_dir}
```

### To run with a real data.

```bash
$ meson ${build_dir} build
$ ${project_dir}/Applications/Resnet/res/prepare_dataset.sh ${dataset_download_dir} # this is to download raw data of cifar100
$ OPENBLAS_NUM_THREADS=1 ${build_dir}/Applications/Resnet/jni/nntrainer_resnet18 \
    ${dataset_download_dir}/cifar-100-binary \
    ${batch_size} \
    ${data_split} \
    ${epoch}
```

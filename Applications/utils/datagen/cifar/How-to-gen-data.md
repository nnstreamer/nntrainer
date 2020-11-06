# How to generate dataset

This is the how to generate the dataset image from cifar dataset.

### Download cifar2png
First we need to clone cifar2png python code from github.

```bash
$ git clone https://github.com/knjcode/cifar2png.git
```


If you successfully download this, you can see


```bash
$ cd cifar2png
$ ls
cifar2png  common  README.md  requirements.txt  setup.py
```

In order to generate bmp file instead of png, you need to fix common/preporcess.py.

If you want to png image, then skip this.

```bash
sed -i 's/.png/.bmp/g' common/preprocess.py
```

And if you want to change image size, you have to add at line 275.
```bash
image = image.resize((224,224))
```


### Generate 
In order to convert, 


```bash
$ cd cifar2png
$ python3 ./cifar2png cifar10 ${SAVED DIR}
```

Then it automatically download tar.gz file and start convert it and save bmp iamges into ${SAVED DIR}



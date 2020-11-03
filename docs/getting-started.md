# Getting Started

## Prerequisites

The following dependencies are needed to compile/build/run.

* gcc/g++ >=4.9 ( std=c++14 is used )
* meson >= 0.50
* libopenblas-dev and base
* tensorflow-lite >=1.14.0
* libiniparser
* libjsoncpp >=0.6.0 ( if you wand to use open AI )
* libcurl3 >=7.47 ( if you wand to use open AI )
* libgtest ( for testing )

## Install via PPA repository (Debian/Ubuntu)

The NNTrainer releases are available with launchpad PPA repository. In order to install it, use:

```bash
sudo apt-add-repository ppa:nnstreamer
sudo apt update
sudo apt install nntrainer
```

## Clean build with pdebuild (Ubuntu 18.04)

Use the NNStreamer PPA to resolve additional build-dependencies (Tensorflow/Tensorflow-Lite).

Install build tools:

```bash
sudo apt install pbuilder debootstrap devscripts
```

The following example configuration is for Ubuntu 18.04:

```bash
$ cat ~/.pbuilderrc
DISTRIBUTION=bionic
COMPONENTS="main restricted universe multiverse"
OTHERMIRROR="deb http://archive.ubuntu.com/ubuntu ${DISTRIBUTION} main restricted universe multiverse |\
  deb http://archive.ubuntu.com/ubuntu ${DISTRIBUTION}-security main restricted universe multiverse |\
  deb http://archive.ubuntu.com/ubuntu ${DISTRIBUTION}-updates main restricted universe multiverse |\
  deb [trusted=yes] http://ppa.launchpad.net/nnstreamer/ppa/ubuntu ${DISTRIBUTION} main"
$ sudo ln -s  ~/.pbuilderrc /root/.pbuilderrc
$ sudo pbuilder create
```

Run pdebuild to build and get the package.

```bash
$ pdebuild
...
$ ls -al /var/cache/pbuilder/result/*.deb
```

Refer to [PbuilderHowto](https://wiki.ubuntu.com/PbuilderHowto) for more about pdebuild.

## Linux Self-Hosted Build

### Build with Debian/Ubuntu tools

#### Clone the needed repositories

```bash
git clone https://github.com/nnstreamer/nntrainer
```

Alternatively, you may simply download binary packages from PPA:

```bash
sudo apt-add-repository ppa:nnstreamer
sudo apt install tensorflow-lite-dev
```

#### Build .deb package

```bash
cd nntrainer && sudo mk-build-deps --install debian/control && sudo dpkg -i *.deb
```

#### Creating the .deb packages

```bash
export DEB_BUILD_OPTIONS="parallel=$(($(cat /proc/cpuinfo |grep processor|wc -l) + 1))"
cd nntrainer && time debuild -us -uc
```

If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add `-uc -us` options to `debuild`.

#### Install the generated \*.deb files

The files will be there at the parent dir. Eg. at `nnbuilder/..` directory.

In order to install them (should run as root):

```bash
sudo apt install ./tensorflow-lite-dev_*.deb
sudo apt install ./nntrainer_0.1.0*_amd64.deb
```

If you need nntrainer development package:

```bash
sudo apt install ./nntrainer-dev_0.1.0*_amd64.deb
```

### Build with meson

Get meson from <https://mesonbuild.com/Getting-meson.html>

Install the required packages.

```bash
sudo apt install meson ninja-build
```

Build at the git repo root directory, this will install nntrainer and related files.

```bash
meson build
ninja -C build install
```

* Installs libraries to ```{prefix}/{libdir}```
* Installs common header files to ```{prefix}/{includedir}```

## Build on Tizen

Get GBS from <https://source.tizen.org/documentation/reference/git-build-system/usage/gbs-build>

First install the required packages.

```bash
sudo apt install gbs
```

Generates .rpm packages:

```bash
gbs build
```

`gbs build` will execute unit testing as well unlike meson build.


## Troubleshooting

### Error 1:

Applications/TransferLearning/Draw_Classification/jni/meson.build:5:0: ERROR: Dependency "glib-2.0" not found, tried pkgconfig and cmake

### Solution: Please install libglib2.0-dev using the following:

sudo apt-get install libglib2.0-dev

### Error 2:

ERROR: Dependency "gstreamer-1.0" not found, tried pkgconfig and cmake

### Solution: Please install  libgstreamer1.0-dev using the following:

sudo apt-get install libgstreamer1.0-dev

### Error 3:

meson.build:157:3: ERROR: Dependency "nnstreamer" not found, tried pkgconfig and cmake

### Solution: Please install  nnstreamer-dev using the following:

sudo apt install nnstreamer-dev



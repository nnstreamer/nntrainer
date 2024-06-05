# Debugging remotely using lldb-server
By following this tutorial, users can learn how to install and run a cross-compiled unittest executable with the lldb-server debugger attached.

## Install lldb from ndk
- This tutorial assumes that the user has already installed the NDK as a prerequisite. Follow [here](https://github.com/nnstreamer/nntrainer/blob/main/docs/how-to-run-example-android.md) if not installed.
- Then you can get the lldb-server from the NDK. (Skip if already installed)

1. On your terminal, find where your lldb at by:
```bash
find "$ANDROID_NDK/toolchains/llvm/prebuilt" -name lldb-server
```
2. Push your lldb to remote Android device 
> Note : Path of ndk might differ
```bash
adb push "$ANDROID_NDK"/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.6/lib/linux/aarch64/* /data/local/tmp/{$YOUR_WORKING_DIRECTORY}
```

## Prepare debuggable executable

### 1. Android build
```bash
cd ~/nntrainer
./tools/package_android.sh
cd ./test/jni
ndk-build NDK_DEBUG=1;
```

### 2. Collect binaries
You need *.so libraries and executables with debug_info, not stripped for lldb.
Find and push executables by:
```bash
cd ~/nntrainer/test/obj/local/arm64-v8a
adb push unittest_* /data/local/tmp/{$YOUR_WORKING_DIRECTORY}
cd ~/nntrainer/builddir/obj/local/arm64-v8a
adb push *.so /data/local/tmp/{$YOUR_WORKING_DIRECTORY}
```

## How to run

### 1. Port forwarding
Please forward the port in the terminal. You may choose any port number that is convenient for you.
```bash
adb forward tcp:5039 tcp:5039
```
### 2. Start lldb server on adb shell (on android remote host)
1. On your terminal, connect to your device via adb shell
```bash
adb shell
```
2. On adb shell, activate your lldb-server.
```adb
/data/local/tmp/{$YOUR_WORKING_DIRECTORY}/lldb-server platform --listen '*:5039' --server
```
### 3. Start lldb from local host
On your terminal, start lldb. Activated lldb-server will start to establish connection immediately.
```bash
lldb -o 'platform select remote-android' \
-o 'platform connect connect://:5039' \
-o 'platform shell cd /data/local/tmp/{$YOUR_WORKING_DIRECTORY}'
```
### 4. Platform workspace setting
On your lldb shell, set your workspace.
```lldb
platform settings -w /data/local/tmp/{$YOUR_WORKING_DIRECTORY}
```
### 5. Formulate executable target
On your lldb shell, set your target executable to debug
```lldb
target create {$YOUR_EXECUTABLE_BINARY_TO_DEBUG}
```
For example,
```lldb
target create unittest_layers
```
### 7. process launch with ld library path
On your lldb shell, launch your process with LD_LIBRARY_PATH included
This might take a while. Wait patiently after the execution...
```lldb
process launch --environment LD_LIBRARY_PATH=.
```

## Tips while debugging
   1. set breakpoints
      1. refer to [break point commands](https://lldb.llvm.org/use/map.html)
   2. `continue` : if you want to inspect the next
   3. `image lookup -va $pc` :  inspect current image

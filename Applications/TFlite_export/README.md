## Setup

Please Follow install guide : https://github.com/nnstreamer/nntrainer/blob/main/docs/getting-started.md

## Run TFlite export example
in nntrainer root directory

```shell
meson setup -Denable-tflite-interpreter=true build
meson compile -C build     
cd build/Applications/TFlite_export
./nntrainer_tflite_export_example
```

## Result
```
/home/donghak/workspace/nntrainer/buildDir/Applications/TFlite_export/nntrainer_tflite_export_example
Model Create Done
Model Load Done
Model Compile Done
Model Initialize Done
Model Export Done
Input Data : { 0.737751, 0.213513, 0.71443, 0.950324, 0.127887, 0.58553, 0.524543, 0.910263, 0.292329, 0.0426721, 0.151403, 0.579323, } 
NNTrainer Output Data : { 0.142062, 0.197614, 0.0257809, 0.387523, 0.215002, 0.256109, -0.101556, 0.370752, 0.134729, } 
TFLITE Output Data : { 0.142062, 0.197615, 0.0257809, 0.387523, 0.215002, 0.256109, -0.101556, 0.370752, 0.134729, } 
```

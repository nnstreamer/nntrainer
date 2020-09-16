# How to generate TFlite model for feature extractor

In order to use tflite model as an feature extractor, we need to remove some of the layers from pre-trained model and convert to tflite model.

### Check the network configuration with tensorboard
First we need to check and confirm which layer output should be used as an feature. Tensorboard is the best way to visualize whole network and make easier to identify the name of the layers.

Let's assume that there is pre-trained and frozen model, say the ssdlitemobilenetv2, as ```${PWD}/ssdmobilenetv2/frozen.pb```

If you have logdir already, then you simply can run

```
$ tensorboard --logdir=./
```


If you do not have logdir, then you need to create it with ```import_pb_to_tensorboard.py``` from tensorflow.


```
$ python ./import_pb_to_tensorboard.py --model_dir ./frozen.pb --log_dir=./tensorboard/
```


Once it save the data in ```log_dir```, then you can visulaize the newtork with tensorboard,

```
$tensorboard --logdir=./tensorboard/
```

After execute the this command, open ```localhost:6006``` with your prefered brower. Then you have.

> 6006 is the default port for tensorboard. If you set different port number, then use it.

<p align = "center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/docs/images/mobilenetv2.gif" >
</p>

Now you can choose the layer removed from tensorboard. Let's say we want to use output of MobilenetV2/Conv_1/Relu6as an feature.

<p align = "center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/docs/images/SecondFeature.gif" >
</p>


### Convert pb to tflite
In order to convert, we need to use ```toco``` from tensorflow.


```
$ toco --graph_def_file=./frozen.pb --output_file=./mobilenetv2.tflite --inference_type=FLOAT --input_shape=1,300,300,3 --input_array=input --output_arrays=MobilenetV2/Conv_1/Relu6
```


Now, you can get the feature data using this ```mobilnetv2.tflite``` model file.


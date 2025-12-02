# Logistic Regression

This is simple logistic regression example. It has two features and 100 data in ```res/dataset1.txt```.

10 test data is in ```test.txt``` for testing with inference option.

Sample ```test.txt``` is like below.


```
2.4443 1.5438 1.0
7.6316 4.602 1.0
4.165 1.5636 1.0
4.8735 2.6093 1.0
5.5061 2.9052 1.0
1.9383 3.6549 0.0
5.9615 6.4565 0.0
5.1012 7.6009 0.0
2.8745 6.0817 0.0
1.7358 5.4503 0.0
```


This example uses the ```DataBufferFromCallback``` to get the training data and configuration model is in ```res/LogisticRegression.ini```

```
# Model Section : Model
[Model]
Type = Regression	    # Network Type : Regression, KNN, NeuralNetwork
Learning_rate = 0.001 	# Learning Rate
Epochs = 500		    # Epochs
Optimizer = sgd		    # Optimizer : sgd (stochastic gradient decent),
Loss = cross    	    #                       cross ( cross entropy )
Save_Path = "model.bin" # model path to save / read
batch_size = 16		    # batch size

# Layer Section : Name
[inputlayer]
Type = input
Input_Shape = 1:1:2

[outputlayer]
Type = fully_connected
Unit = 1
Bias_initializer = zeros
Activation = sigmoid

```

Once you compile, you can train with

```bash

/data/nntrainer/build/Applications/LogisticRegression/jni/nntrainer_logistic train /data/nntrainer/build/res/app/LogisticRegression/LogisticRegression.ini /data/nntrainer/build/res/app/LogisticRegression/dataset1.txt

export ${res}
$ cd ${build_dir}
$ export res=$(pwd)/res/app/LogisticRegression
$ export app=$(pwd)/Application/LogisticRegression/jni/nntrainer_logistic
$ ${app} train ${res}/LogisticRegression.ini ${res}/dataset1.txt
```

You can see ```logistic_model.bin``` to inference after training and check the accuracy with

```bash
$ ${app} inference ${res}/LogisticRegression.ini ${res}/dataset1.txt
1 : 1
1 : 1
1 : 1
1 : 1
1 : 1
0 : 0
0 : 0
0 : 0
0 : 0
0 : 0
[ Accuracy ] : 100%
```

# Learning with Feature Extractor (Fully Connected Layer)

Unlike using kNN for the classifier at #55 , we could add new layer at the end of feature extractor.
In this toy example, fully connected layer is added to classify as below. The size of the feature is also same with previous example and the training set and test set are also same. ( 3 classes, 5 training set and 2 test set of each )
Only fully connected layer is updatable and Mobilenet ssd v2 is used for feature extractor like previous example and all the testing and training is done on the Galaxy S8.
I wrote some code for the forward and backward propagation during fully connected layer training.

Sigmoid function is used for the activation. Square error loss function and gradient decent is used for the loss function and optimizer. Just 20 neuron is used for the hidden layer and iterate 300 times to make is more simple.

The configuration of the example is below,
<p align = "center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/doc/02a7ee80-f0ce-11e9-97b8-bcc19b7eb222.png" width="400" height="250" > </p>

Training set and test set are below
<p align = "center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/doc/7944ec00-f0ce-11e9-87af-aea730bcd0f5.png" >
</p>

After Iterating 300 times, the change of L2 Norm of the Loss function is below.
<p align = "center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/doc/d42b1300-f0cf-11e9-9b6f-6db30def4684.png" width="500" height="300">
</p>

and the test results for the 8 test cases are below. Step function is used to make more clear.
As you can see, the test result is ok.

<p align ="center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/doc/16555400-f0d2-11e9-959b-f61935fefd5a.png" width ="500" height="180">
</p>


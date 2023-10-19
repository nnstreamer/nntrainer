# How to make your own model using NNTrainer
In this tutorial, we will make a simple neuralnet model called MyApp using NNTrainer.

## Create "MyApp" script file
- Create "MyApp/jni/main.cpp" file on "Applications" directory.
- The structure of the directory is as follows.

```bash
...
├─── api
├─── Applications
│   └─── MyApp
│      └─── jni
│         └─── main.cpp
├─── build
...
```

- Add the following code at the end of the "Applications/meson.build": "subdir('MyApp/jni')".

```bash
...
subdir('MyApp/jni')
```

## Writing scripts for MyApp application
Now let's start implementing a simple neural network in the main.cpp.

### Header files
First, include the header files required for modeling.
```cpp
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <cifar_dataloader.h>
```

### Training configurations
Next, let's set up the configuration for training.
```cpp
const int number_of_db = 16;
const int batch_size = 4;
const int epochs = 10;
const float learning_rate = 0.001;
```

### Modelling
We can create a prototype model using the "createModel" function, then add layers to this object for modelling.

Layers can be created using the "createLayer" function, and the first parameter of the createLayer function indicates the type of layer.

In this example, we will create the model using an input layer and a fully connected layer.

```cpp
std::unique_ptr<ml::train::Model> create_model() {
    std::unique_ptr<ml::train::Model> model = 
        ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

    model->addLayer(
        ml::train::createLayer(
        "input", {"input_shape=1:1:10"})
    );

    model->addLayer(
        ml::train::createLayer("fully_connected", {"unit=5", "activation=softmax"}));

    return model;
}
```

When creating a prototype of the model, you can specify the type of loss as a hyperparameter. Here, it was specified to use mse loss. When creating an input layer, the shape of the input data need to be specified.

Here, 1:1:10 means channel, height, and width in order, respectively, which is equivalent to (10,) shape based on Python's Numpy. Note that channel last will be supported soon.

### Dataset
In this tutorial, we will use the random data generator as a training dataset.

Here, a random data generator is used only for example, so you don't need to understand this part in detail. Howerver, keep in mind that the length of input data is 10 and the length of output data is 1.
```cpp
std::unique_ptr<nntrainer::util::DataLoader> getRandomDataGenerator() {
    std::unique_ptr<nntrainer::util::DataLoader> random_db(
        new nntrainer::util::RandomDataLoader({{batch_size, 1, 1, 10}}, {{batch_size, 1, 1, 1}}, number_of_db));

    return random_db;
}

int dataset_cb(float **input, float **label, bool *last, void *user_data) {
    auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

    data->next(input, label, last);
    return 0;
}
```

### Main logic
Now let's write the main logic to run the actual code. The batch size, number of epochs, and the path to save the trained model can be specified using "setProperty" method.

An optimizer can be created using the createOptimizer function. In this tutorial, we will use the SGD optimizer.

After preparing the model in this way, it needs to be compiled and initialized for actual training. This can be done using the "compile" and "initialize" methods, respectively. Finally, once the dataset for training is set up, all preparations for training are done. Model training can be done using the "train" method. The trained model is automatically saved in the path that was set when creating the model.
```cpp
int main(int argc, char *argv[]) {
    auto model = create_model();

    model->setProperty({"batch_size=" + std::to_string(batch_size),
                        "epochs=" + std::to_string(epochs),
                        "save_path=my_app.bin"});

    auto optimizer = ml::train::createOptimizer("SGD", {"learning_rate=" + std::to_string(learning_rate)});
    model->setOptimizer(std::move(optimizer));

    int status = model->compile();
    status = model->initialize();

    auto random_generator = getRandomDataGenerator();
    auto train_dataset = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, dataset_cb, random_generator.get());

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(train_dataset));

    model->train();

    return status;
}
```

## Build and Run "MyApp" application
To build MyApp application, execute the following command on the NNTrainer directory: "meson build & ninja -C build".
```bash
meson build & ninja -C build
```
Then, it will create the build directory and compile your app to "build/Applications/MyApp/jni".

Finally, you can run your own "MyApp" application with following command on "build/Applications/MyApp/jni" directory.

```bash
./nntrainer_myapp
```

Then you can see the training & validation results like this. In this example, the loss does not decrease because new random data is created each time during training, but if the model is trained using real data, it can be confirmed that the loss converges. 

```bash
#1/10 - Training Loss: 0.199999 
#2/10 - Training Loss: 0.2
#3/10 - Training Loss: 0.196525
#4/10 - Training Loss: 0.199955
#5/10 - Training Loss: 0.19496
#6/10 - Training Loss: 0.196071
#7/10 - Training Loss: 0.193764
#8/10 - Training Loss: 0.2
#9/10 - Training Loss: 0.194124
#10/10 - Training Loss: 0.199409
```

Now, you can check the trained weights binary file from "build/Applications/MyApp/jni" directory. If you want to load the trained weights, you can specify the path with the "load" method in the model object.

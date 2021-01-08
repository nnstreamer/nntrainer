subdir('utils/jni')
subdir('KNN/jni')
subdir('LogisticRegression/jni')
if get_option('enable-ccapi')
  subdir('MNIST/jni')
endif
subdir('VGG/jni')
subdir('ReinforcementLearning/DeepQ/jni')
subdir('TransferLearning/CIFAR_Classification/jni')
subdir('TransferLearning/Draw_Classification/jni')
subdir('Custom/LayerClient/jni')

if get_option('enable-tflite-backbone')
  subdir('SimpleShot')
endif

nntr_app_resdir = nntrainer_resdir / 'app'
run_command('mkdir', '-p', nntr_app_resdir)

subdir('utils')
subdir('LogisticRegression/jni')
if enable_ccapi
  subdir('MNIST/jni')
endif
subdir('VGG/jni')
subdir('Resnet/jni')
subdir('ReinforcementLearning/DeepQ/jni')
subdir('TransferLearning')
subdir('Custom')
subdir('ProductRatings/jni')
subdir('AlexNet/jni')
if get_option('enable-tflite-backbone')
  subdir('KNN/jni')
  subdir('SimpleShot')
endif

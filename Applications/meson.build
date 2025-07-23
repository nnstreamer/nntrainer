fs = import('fs')

nntr_app_resdir = nntrainer_resdir / 'app'

if build_machine.system() == 'windows'
  nntr_app_resdir_win = nntr_app_resdir.replace('/', '\\')
  if not fs.exists (nntr_app_resdir_win)
    run_command([prog_win_cmd, '/C', 'mkdir', nntr_app_resdir_win], check: true)
  endif
else
  run_command(['mkdir', '-p', nntr_app_resdir], check: true)
endif

if get_option('enable-app')
  if get_option('enable-capi')
    subdir('MNIST')
    subdir('LogisticRegression')
    subdir('ReinforcementLearning')
    subdir('SimpleFC')
    subdir('TransferLearning')
    subdir('utils')
    subdir('VGG')
    subdir('AlexNet')
    subdir('Resnet')
    subdir('YOLOv2')
    subdir('YOLOv3')
    subdir('SimpleShot')
    subdir('ProductRatings')
    subdir('Multi_input')
    subdir('MixedPrecision')
    subdir('Layers')
    subdir('KNN')
    if get_option('enable-nnstreamer-backbone')
      subdir('ONNX')
    endif
  endif

  if get_option('enable-ccapi')
    subdir('Tizen_native')
  endif
endif

if host_machine.system() != 'windows'
  subdir('Custom')
  subdir('PicoGPT/jni')
endif

# Always build CausalLM for both desktop and Android
subdir('CausalLM')

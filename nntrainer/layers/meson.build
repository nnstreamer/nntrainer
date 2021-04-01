layer_sources = [
  'activation_layer.cpp',
  'addition_layer.cpp',
  'concat_layer.cpp',
  'output_layer.cpp',
  'bn_layer.cpp',
  'conv2d_layer.cpp',
  'fc_layer.cpp',
  'flatten_layer.cpp',
  'input_layer.cpp',
  'output_layer.cpp',
  'layer.cpp',
  'layer_factory.cpp',
  'loss_layer.cpp',
  'pooling2d_layer.cpp',
  'preprocess_flip_layer.cpp',
  'preprocess_translate_layer.cpp',
  'embedding.cpp',
  'rnn.cpp',
  'acti_func.cpp'
]

layer_headers = [
  'layer_factory.h',
  'layer_internal.h',
  'acti_func.h',
  'loss_layer.h'
]

layer_deps = []

if get_option('enable-nnstreamer-backbone')
  if not nnstreamer_capi_dep.found()
    error('NNStreamer CAPI dependency not found')
  endif
  layer_deps += nnstreamer_capi_dep
  layer_sources += 'nnstreamer_layer.cpp'
endif

if get_option('enable-tflite-backbone')
  if not tflite_dep.found()
    error('Tensorflow-Lite dependency not found')
  endif
  layer_deps += tflite_dep
  layer_sources += 'tflite_layer.cpp'
endif

# dependency for data augmentation, if present
layer_deps += opencv_dep

nntrainer_base_deps += layer_deps

foreach s : layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : layer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

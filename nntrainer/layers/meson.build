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
  'pooling2d_layer.cpp'
]

layer_headers = [
  'layer_factory.h',
  'layer_internal.h'
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

nntrainer_base_deps += layer_deps

foreach s : layer_sources
  nntrainer_sources += join_paths(meson.current_source_dir(), s)
endforeach

foreach h : layer_headers
  nntrainer_headers += join_paths(meson.current_source_dir(), h)
endforeach

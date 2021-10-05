loss_layer_sources = [
  'loss_layer.cpp',
  'mse_loss_layer.cpp',
  'cross_entropy_sigmoid_loss_layer.cpp',
  'cross_entropy_softmax_loss_layer.cpp',
  'constant_derivative_loss_layer.cpp'
]

loss_layer_headers = []

loss_layer_deps = []

nntrainer_base_deps += loss_layer_deps

foreach s : loss_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : loss_layer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


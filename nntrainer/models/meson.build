model_sources = [
  'model_loader.cpp',
  'neuralnet.cpp',
  'model_common_properties.cpp',
  'dynamic_training_optimization.cpp',
]

model_headers = [
  'neuralnet.h',
  'dynamic_training_optimization.h',
  'model_common_properties.h',
]

foreach s : model_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : model_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


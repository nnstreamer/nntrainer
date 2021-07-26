model_sources = [
  'model_loader.cpp',
  'neuralnet.cpp',
  'dynamic_training_optimization.cpp'
]

model_headers = []

foreach s : model_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : model_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


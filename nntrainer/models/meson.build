model_sources = [
  'model_loader.cpp',
  'neuralnet.cpp'
]

model_headers = [
  'neuralnet.h'
]

foreach s : model_sources
  nntrainer_sources += join_paths(meson.current_source_dir(), s)
endforeach

foreach h : model_headers
  nntrainer_headers += join_paths(meson.current_source_dir(), h)
endforeach


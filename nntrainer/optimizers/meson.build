optimizer_sources = [
  'adam.cpp',
  'optimizer.cpp',
  'optimizer_factory.cpp',
  'sgd.cpp'
]

optimizer_headers = [
  'optimizer_factory.h',
  'optimizer_internal.h'
]

foreach s : optimizer_sources
  nntrainer_sources += join_paths(meson.current_source_dir(), s)
endforeach

foreach h : optimizer_headers
  nntrainer_headers += join_paths(meson.current_source_dir(), h)
endforeach


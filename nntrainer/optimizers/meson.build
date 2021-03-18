optimizer_sources = [
  'adam.cpp',
  'optimizer_devel.cpp',
  'optimizer_impl.cpp',
  'optimizer_factory.cpp',
  'sgd.cpp'
]

optimizer_headers = [
  'optimizer_factory.h',
  'optimizer_devel.h',
  'optimizer_impl.h'
]

foreach s : optimizer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : optimizer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


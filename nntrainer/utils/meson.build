util_sources = [
  'util_func.cpp',
  'profiler.cpp',
  'ini_wrapper.cpp',
  'node_exporter.cpp',
  'base_properties.cpp',
  'nntr_threads.cpp'
]

util_headers = [
  'base_properties.h',
  'node_exporter.h',
  'util_func.h',
  'profiler.h',
  'nntr_threads.h'
]

foreach s : util_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : util_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

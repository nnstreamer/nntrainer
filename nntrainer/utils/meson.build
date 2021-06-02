util_sources = [
  'parse_util.cpp',
  'util_func.cpp',
  'profiler.cpp',
  'ini_wrapper.cpp',
  'node_exporter.cpp',
  'base_properties.cpp'
]

util_headers = [
  'profiler.h',
  'base_properties.h',
  'parse_util.h',
  'node_exporter.h'
]

foreach s : util_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : util_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

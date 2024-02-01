util_sources = [
  'util_func.cpp',
  'profiler.cpp',
  'ini_wrapper.cpp',
  'node_exporter.cpp',
  'base_properties.cpp',
  'nntr_threads.cpp',
  'fp16.cpp',
  'util_simd.cpp',
]

util_headers = [
  'base_properties.h',
  'node_exporter.h',
  'util_func.h',
  'profiler.h',
  'nntr_threads.h',
  'fp16.h',
  'util_simd.h',
]

if get_option('enable-trace')
  util_sources += 'tracer.cpp'
  util_headers += 'tracer.h'
endif

if get_option('enable-fp16')
  util_sources += 'util_simd_neon.cpp'
  util_headers += 'util_simd_neon.h'
endif

foreach s : util_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : util_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

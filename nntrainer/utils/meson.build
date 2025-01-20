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
  'dynamic_library_loader.h',
]

if get_option('enable-trace')
  util_sources += 'tracer.cpp'
  util_headers += 'tracer.h'
endif

arch = host_machine.cpu_family()
if get_option('enable-fp16') 
  if arch == 'arm'
    error ('FP16/ARM code (blas_neon.cpp) uses armv8.2 instructions. armv7 is not supported.')
  elif arch == 'aarch64' or get_option('platform') == 'android'
    if get_option('enable-neon')
      util_sources += 'util_simd_neon.cpp'
      util_headers += 'util_simd_neon.h'
    endif
  endif
endif

foreach s : util_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : util_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

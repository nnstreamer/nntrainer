cpu_backend_headers = [
  'cpu_backend.h',
]

subdir('fallback')
nntrainer_inc += include_directories('fallback')
nntrainer_inc_abs += meson.current_source_dir() / 'fallback'

if get_option('enable-blas')
  subdir('cblas_interface')
  nntrainer_inc += include_directories('cblas_interface')
  nntrainer_inc_abs += meson.current_source_dir() / 'cblas_interface'
endif

arch = host_machine.cpu_family()
if arch == 'arm'
  error ('FP16/ARM code (blas_neon.cpp) uses armv8.2 instructions. armv7 is not supported.')
elif arch == 'aarch64' or get_option('platform') == 'android'
  if get_option('enable-neon')
    subdir('aarch64')
    nntrainer_inc += include_directories('aarch64')
    nntrainer_inc_abs += meson.current_source_dir() / 'aarch64'
  endif
elif get_option('enable-avx')
  subdir('x86')
  nntrainer_inc += include_directories('x86')
  nntrainer_inc_abs += meson.current_source_dir() / 'x86'
endif

foreach h : cpu_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

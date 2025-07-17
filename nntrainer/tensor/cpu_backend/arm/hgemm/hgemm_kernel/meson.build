hgemm_kernel_headers = [
  'hgemm_kernel.h',
]


hgemm_kernel_sources = [
    'hgemm_kernel_1x4.cpp',
    'hgemm_kernel_1x8.cpp',
    'hgemm_kernel_4x4.cpp',
    'hgemm_kernel_4x8.cpp',
    'hgemm_kernel_8x8.cpp',
]

if get_option('hgemm-experimental-kernel')
  hgemm_kernel_sources += 'hgemm_kernel_8x16_experimental.cpp'
else
  hgemm_kernel_sources += 'hgemm_kernel_8x16.cpp'
endif

foreach s : hgemm_kernel_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : hgemm_kernel_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


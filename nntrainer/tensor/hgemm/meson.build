hgemm_headers = [
  'hgemm.h',
  'hgemm_util.h',
  'hgemm_kernel_pack.h',
  'hgemm_kernel_4x4.h',
  'hgemm_kernel_4x8.h',
  'hgemm_kernel_8x8.h',
  'hgemm_kernel_8x16.h',
]

hgemm_sources = [
    'hgemm.cpp',
    'hgemm_padding_a.cpp',
    'hgemm_padding_b.cpp',
]

foreach s : hgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : hgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


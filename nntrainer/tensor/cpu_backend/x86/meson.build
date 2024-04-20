simd_interface_x86_headers = [
    'avx_simd.h',
]
simd_interface_x86_sources = [
  'avx_simd.cpp',
]

if get_option('enable-fp16')
    simd_interface_x86_headers += 'blas_avx.h'
    simd_interface_x86_sources += 'blas_avx.cpp'
endif

foreach s : simd_interface_x86_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : simd_interface_x86_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

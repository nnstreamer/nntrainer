matrix_transpose_neon_sources = [
    'matrix_transpose_neon.cpp',
]

matrix_transpose_neon_headers = [
    'matrix_transpose_neon.h',
]

if get_option('enable-fp16') 
    matrix_transpose_neon_sources += 'matrix_transpose_neon_f16.cpp'
    matrix_transpose_neon_headers += 'mask_neon.h'
    matrix_transpose_neon_headers += 'matrix_transpose_kernels_neon.h'
endif

foreach s : matrix_transpose_neon_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matrix_transpose_neon_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

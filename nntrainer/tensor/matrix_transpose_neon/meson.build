matrix_transpose_neon_sources = [
    'matrix_transpose_neon.cpp',
]

matrix_transpose_neon_headers = [
    'matrix_transpose_neon.h',
]

if get_option('enable-fp16') 
  if arch == 'arm'
    error ('FP16/ARM code (blas_neon.cpp) uses armv8.2 instructions. armv7 is not supported.')
  elif arch == 'aarch64' or get_option('platform') == 'android'
    if get_option('enable-neon')
      matrix_transpose_neon_sources += 'matrix_transpose_neon_f16.cpp'
      matrix_transpose_neon_headers += 'mask_neon.h'
      matrix_transpose_neon_headers += 'matrix_transpose_kernels_neon.h'
    endif
  endif
endif

foreach s : matrix_transpose_neon_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matrix_transpose_neon_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

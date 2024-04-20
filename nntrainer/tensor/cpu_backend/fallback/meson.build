fallback_headers = ['fallback_internal.h']
fallback_sources = ['fallback_internal.cpp']

if get_option('enable-avx')
  fallback_headers += 'custom_fallback.h'
  fallback_sources += 'custom_fallback.cpp'
elif get_option('enable-blas')
  fallback_headers += 'custom_fallback.h'
  fallback_sources += 'custom_fallback.cpp'
  fallback_headers += 'tmp_fallback.h'
  fallback_sources += 'tmp_fallback.cpp'
else
  fallback_headers += 'cblas_fallback.h'
  fallback_sources += 'cblas_fallback.cpp'
  fallback_headers += 'custom_fallback.h'
  fallback_sources += 'custom_fallback.cpp'
  fallback_headers += 'tmp_fallback.h'
  fallback_sources += 'tmp_fallback.cpp'
endif

foreach s : fallback_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : fallback_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

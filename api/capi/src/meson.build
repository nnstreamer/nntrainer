capi_src += files('nntrainer.cpp')
capi_src_abs += meson.current_source_dir() / 'nntrainer.cpp'

if get_option('platform') == 'tizen' and get_option('enable-tizen-feature-check')
  capi_src += files('nntrainer-capi-tizen-feature-check.cpp')
  capi_src_abs += meson.current_source_dir() / 'nntrainer-capi-tizen-feature-check.cpp'
endif

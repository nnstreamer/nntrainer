capi_inc = []
capi_inc += include_directories('include')
capi_inc += include_directories('..')
capi_src = []
capi_src += meson.current_source_dir() / 'src' / 'nntrainer.cpp'

if not nnstreamer_capi_dep.found()
  error('nnstreamer capi dependency not found for tizen')
endif

if get_option('platform') == 'tizen' and get_option('enable-tizen-feature-check')
  capi_src += meson.current_source_dir() / 'src' / 'nntrainer-capi-tizen-feature-check.cpp'
endif

capi_headers = []
capi_headers += meson.current_source_dir() / 'include' / 'nntrainer.h'
capi_headers += meson.current_source_dir() / '..' / 'nntrainer-api-common.h'

if get_option('platform') == 'tizen'
  capi_headers += meson.current_source_dir() / 'include' / 'nntrainer-tizen-internal.h'
endif

capi_deps = [
  nntrainer_ccapi_dep,
  nnstreamer_capi_dep,
]

if get_option('platform') == 'tizen'
  message('CAPI is in Tizen mode')

  tizen_deps = [
    dependency('capi-base-common'),
    dependency('capi-system-info'),
    dependency('dlog')
  ]

  capi_deps += tizen_deps
endif

shared_library('capi-nntrainer',
  capi_src,
  dependencies: capi_deps,
  include_directories: capi_inc,
  install: true,
  install_dir: nntrainer_libdir,
)

nntrainer_capi_lib = static_library('capi-nntrainer',
  capi_src,
  dependencies: capi_deps,
  include_directories: capi_inc,
  install: true,
  install_dir: nntrainer_libdir,
)

nntrainer_capi_dep = declare_dependency(link_with: nntrainer_capi_lib,
  dependencies: capi_deps,
  include_directories: capi_inc,
)

configure_file(input: 'capi-ml-training.pc.in', output: 'capi-ml-training.pc',
  install_dir: nntrainer_libdir / 'pkgconfig',
  configuration: nntrainer_conf
)

install_headers(capi_headers,
  subdir: 'nntrainer'
)

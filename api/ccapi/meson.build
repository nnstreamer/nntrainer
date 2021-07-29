ccapi_inc = []
ccapi_inc += include_directories('include')
ccapi_inc += include_directories('..')


ccapi_src = []
ccapi_src += meson.current_source_dir() / 'src' / 'factory.cpp'

ccapi_headers = []
ccapi_headers += meson.current_source_dir() / 'include' / 'dataset.h'
ccapi_headers += meson.current_source_dir() / 'include' / 'layer.h'
ccapi_headers += meson.current_source_dir() / 'include' / 'model.h'
ccapi_headers += meson.current_source_dir() / 'include' / 'optimizer.h'
ccapi_headers += meson.current_source_dir() / 'include' / 'tensor_dim.h'
ccapi_headers += meson.current_source_dir() / '..' / 'nntrainer-api-common.h'

ccapi_deps = [
  nntrainer_dep
]

shared_library('ccapi-nntrainer',
  ccapi_src,
  dependencies: ccapi_deps,
  include_directories: ccapi_inc,
  install: true,
  install_dir: nntrainer_libdir,
)

nntrainer_ccapi_lib = static_library('ccapi-nntrainer',
  ccapi_src,
  dependencies: ccapi_deps,
  include_directories: ccapi_inc,
  install: true,
  install_dir: nntrainer_libdir,
)

nntrainer_ccapi_dep = declare_dependency(link_with: nntrainer_ccapi_lib,
  dependencies: ccapi_deps,
  include_directories: ccapi_inc,
)

configure_file(input: 'ccapi-ml-training.pc.in', output: 'ccapi-ml-training.pc',
  install_dir: nntrainer_libdir / 'pkgconfig',
  configuration: nntrainer_conf
)

install_headers(ccapi_headers,
  subdir: 'nntrainer'
)

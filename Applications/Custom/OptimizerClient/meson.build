optimizer_client_sources = [
  'main.cpp',
  optimizer_example_momentum_src,
]

exe = executable('optimizer_client',
  optimizer_client_sources,
  include_directories: example_inc,
  dependencies: [nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_optimizer_client_api', exe)

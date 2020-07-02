mnist_sources = [
  'main.cpp'
]

executable('nntrainer_mnist',
  mnist_sources,
  dependencies: [iniparser_dep, nntrainer_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

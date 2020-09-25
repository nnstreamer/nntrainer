mnist_sources = [
  'main.cpp'
]
if build_platform == 'tizen'
  if not gtest_dep.found()
    error('Gtest dependency not found for MNIST application')
  endif
endif

executable('nntrainer_mnist',
  mnist_sources,
  dependencies: [iniparser_dep, nntrainer_dep, gtest_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

## @todo Add test after #376

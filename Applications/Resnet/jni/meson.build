resnet_sources = [
  'main.cpp',
  'cifar_dataloader.cpp'
]

e = executable('nntrainer_resnet18',
  resnet_sources,
  include_directories: '.',
  dependencies: [app_utils_dep, iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)


if get_option('enable-long-test')
  testenv = environment()
  testenv.set('OPENBLAS_NUM_THREADS', 1)
  test('app_resnet18', e, args: ['fake', '1', '512'], env: testenv, timeout: 300)
endif

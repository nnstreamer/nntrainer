refinedet_sources = [
  'jni/main.cpp',
  'refinedet_loss.cpp',
  cifar_path / 'cifar_dataloader.cpp'
]

refinedet_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  refinedet_dependencies += [gtest_dep]
endif

e = executable('nntrainer_refinedet',
  refinedet_sources,
  include_directories: [include_directories('.'), include_directories('jni'), cifar_include_dir],
  dependencies: refinedet_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

if get_option('enable-long-test')
  testenv = environment()
  testenv.set('OPENBLAS_NUM_THREADS', '4')
  test('app_refinedet', e, args: ['fake', '1', '128', '1'], env: testenv, timeout: 300)
endif
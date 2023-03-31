# build command for libmomentum_optimizer.so
momentum_optimizer = shared_library('momentum_optimizer',
  optimizer_example_momentum_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: layer_example_inc,
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)

momentum_optimizer_dep = declare_dependency(
  link_with: momentum_optimizer,
  include_directories: optimizer_example_inc
  )

if get_option('enable-test')
  test_target = [
    'optimizer_plugin_common_test.cpp',
    'optimizer_plugin_momentum_test.cpp'
  ]

  exe = executable(
    'app_optimizer_plugin_test', test_target,
    dependencies: [
      gtest_main_dep,
      nntrainer_dep,
      nntrainer_ccapi_dep,
      nntrainer_testutil_dep,
      momentum_optimizer_dep,
    ],
    install: get_option('enable-test'),
    install_dir: application_install_dir,
    include_directories: optimizer_example_inc
  )
  testenv = environment()
  testenv.set('NNTRAINER_PATH', meson.current_build_dir())
  test('app_plugin_test', exe, env: testenv)
endif

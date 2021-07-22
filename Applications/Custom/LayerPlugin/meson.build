# build command for libpow_layer.so
pow_layer = shared_library('pow_layer',
  layer_example_pow_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: layer_example_inc,
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)

# build command for libmae_loss_layer.so
mae_loss_layer  = shared_library('mae_loss_layer',
  layer_example_mae_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: layer_example_inc,
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)

# build command for librnnt_loss_layer.so
rnnt_loss_layer  = shared_library('rnnt_loss_layer',
  layer_example_rnnt_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: layer_example_inc,
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)

pow_layer_dep = declare_dependency(
  link_with: pow_layer,
  include_directories: layer_example_inc
  )

mae_loss_layer_dep = declare_dependency(
  link_with: mae_loss_layer,
  include_directories: layer_example_inc
  )

rnnt_loss_layer_dep = declare_dependency(
  link_with: rnnt_loss_layer,
  include_directories: layer_example_inc
  )

if get_option('enable-test')
  test_target = [
    'layer_plugin_common_test.cpp',
    'layer_plugin_pow_test.cpp',
    'layer_plugin_mae_loss_test.cpp',
    'layer_plugin_rnnt_loss_test.cpp'
  ]

  exe = executable(
  'app_layer_plugin_test', test_target,
  dependencies: [
    gtest_main_dep,
    nntrainer_dep,
    nntrainer_ccapi_dep,
    nntrainer_testutil_dep,
    pow_layer_dep,
    mae_loss_layer_dep,
    rnnt_loss_layer_dep,    
    nntrainer_layer_common_standalone_tests_dep
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir)

  testenv = environment()
  testenv.set('NNTRAINER_PATH', meson.current_build_dir())
  test('app_plugin_test', exe, env: testenv)
endif

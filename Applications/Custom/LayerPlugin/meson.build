pow_layer = shared_library('pow_layer',
  layer_example_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: layer_example_inc,
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)

if get_option('enable-test')
  test_target = [
    'layer_plugin_test.cpp'
  ]

  exe = executable(
  'app_plugin_test', test_target,
  dependencies: [
    gtest_dep_with_main,
    nntrainer_dep,
    nntrainer_ccapi_dep,
    nntrainer_testutil_dep
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

testenv = environment()
testenv.set('RES_PATH', meson.current_build_dir())
test('app_plugin_test', exe, env: testenv)
endif

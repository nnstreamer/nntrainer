test_name = 'unittest_models'

test_target = []

models_targets = [
  'unittest_models_recurrent.cpp',
]

test_target += models_targets
exe = executable(
  test_name,
  test_target,
  dependencies: [
    nntrainer_test_main_deps,
    nntrainer_capi_dep
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test(test_name, exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), test_name)
)

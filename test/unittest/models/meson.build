test_name = 'unittest_models'

test_target = []

models_targets = [
  'models_test_utils.cpp',
  'models_golden_test.cpp',
  'unittest_models_recurrent.cpp',
  'unittest_models_multiout.cpp',
  'unittest_models.cpp',
  # disable temperally
]

test_target += models_targets
exe = executable(
  test_name,
  test_target,
  include_directories: include_directories('.'),
  dependencies: [
    nntrainer_test_main_deps, nntrainer_ccapi_dep
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test(test_name, exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), test_name),
  timeout: test_timeout
)

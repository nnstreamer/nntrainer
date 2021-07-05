test_target = [
  'layers_standalone_common_tests.cpp',
  'layers_dependent_common_tests.cpp',
  'unittest_layers_impl.cpp',
  'unittest_layers_input.cpp',
  'unittest_layers_loss.cpp',
  'unittest_layers_fully_connected.cpp',
  'unittest_layers_batch_normalization.cpp',
]

exe = executable(
  'unittest_layers', test_target,
  dependencies: [
    nntrainer_test_main_deps,
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

# @todo delete v2 once migration is done
test('unittest_layers_v2', exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_layers_v2')
)

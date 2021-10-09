test_name = 'unittest_compiler'

test_target = [
  'compiler_test_util.cpp',
  'unittest_compiler.cpp',
  'unittest_interpreter.cpp',
  'unittest_realizer.cpp'
]

exe = executable(
  test_name,
  test_target,
  dependencies: [
    nntrainer_test_main_deps,
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test(test_name, exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), test_name)
)

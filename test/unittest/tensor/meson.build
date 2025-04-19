tensor_test_inc = include_directories('./')

nntrainer_tensor_tests_dep = declare_dependency(
  include_directories: tensor_test_inc
  )

test_target = [
  'unittest_tensor_feature.cpp'
]

if cxx.get_id() == 'msvc'
  cpp_args_str = []
else
  cpp_args_str = '-Wno-maybe-uninitialized'
endif

# tensor memory unittests
exe = executable(
  'unittest_tensor_feature', test_target,
  dependencies: [
    nntrainer_test_main_deps
  ],
  cpp_args : cpp_args_str,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test('unittest_tensor', exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_tensor'),
  timeout: test_timeout,
  suite: 'unittests'
)

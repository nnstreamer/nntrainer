unittest_ccapi_deps = [
  nntrainer_ccapi_dep,
  nntrainer_test_deps,
]

exec = executable(
  'unittest_ccapi',
  'unittest_ccapi.cpp',
  dependencies: unittest_ccapi_deps,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test('unittest_ccapi', exec, timeout: 120, args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), target))

unittest_ccapi_deps = [
  nntrainer_ccapi_dep,
  nntrainer_test_deps,
]

ccapi_targets = [
  'unittest_ccapi.cpp',
  'unittest_ccapi_tensor.cpp',
]

exec = executable(
  'unittest_ccapi',
  ccapi_targets,
  dependencies: unittest_ccapi_deps,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test('unittest_ccapi', exec,
     timeout: 120,
     args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_ccapi'),
     suite: 'unittests')

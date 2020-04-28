unittest_tizen_deps = [
  nntrainer_capi_dep,
  nntrainer_unittest_deps,
  gtest_dep
]

unittest_tizen_src=[
  'unittest_tizen_capi.cpp',
  '../nntrainer_test_util.cpp'
]

unittest_tizen_capi = executable('unittest_tizen_capi',
  unittest_tizen_src,
  dependencies: [unittest_tizen_deps],
  include_directories: nntrainer_test_inc,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test('unittest_tizen_capi', unittest_tizen_capi)

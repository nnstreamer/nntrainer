unittest_nntrainer_internal_deps = [
  nntrainer_unittest_deps,
  gtest_dep
]

unittest_nntrainer_src=[
  'unittest_nntrainer_internal.cpp',
  '../nntrainer_test_util.cpp'
]

unittest_nntrainer_internal = executable('unittest_nntrainer_internal',
  unittest_nntrainer_src,
  dependencies: [unittest_nntrainer_internal_deps],
  include_directories: nntrainer_test_inc,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('unittest_nntrainer_internal', unittest_nntrainer_internal)


unittest_util_func = executable('unittest_util_func',
  'unittest_util_func.cpp',
  dependencies: [unittest_nntrainer_internal_deps],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('unittest_util_func', unittest_util_func)

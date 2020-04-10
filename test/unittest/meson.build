unittest_nntrainer_internal_deps = [
  nntrainer_unittest_deps,
  gtest_dep
]

unittest_nntrainer_internal = executable('unittest_nntrainer_internal',
  'unittest_nntrainer_internal.cpp',
  dependencies: [unittest_nntrainer_internal_deps],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test('unittest_nntrainer_internal', unittest_nntrainer_internal)

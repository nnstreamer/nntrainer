gtest_dep_with_main = dependency('gtest', main : true, required : false)


test_target = [
  'simpleshot_utils_test.cpp',
  'simpleshot_centering_test.cpp',
  'simpleshot_l2norm_test.cpp',
  'simpleshot_centroid_knn.cpp'
]

exe = executable(
  'simpleshot_tests', test_target,
  dependencies: [gtest_dep_with_main,
      simpleshot_test_dep,
      nntrainer_dep,
      nntrainer_ccapi_dep,
      nntrainer_testutil_dep],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('simpleshot_tests', exe)


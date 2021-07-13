test_target = [
  # 'simpleshot_utils_test.cpp',
  # 'simpleshot_centering_test.cpp',
  # 'simpleshot_l2norm_test.cpp',
  # 'simpleshot_centroid_knn.cpp',
  'simpleshot_layer_common_tests.cpp'
]

exe = executable(
  'simpleshot_tests', test_target,
  dependencies: [gtest_main_dep,
      simpleshot_test_dep,
      nntrainer_dep,
      nntrainer_ccapi_dep,
      nntrainer_testutil_dep,
      nntrainer_layer_common_standalone_tests_dep
      ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('simpleshot_tests', exe, args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'simpleshot_tests'))


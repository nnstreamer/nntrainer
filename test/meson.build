gtest_dep = dependency('gtest', required: false)

nntrainer_test_inc = include_directories('./include')

if gtest_dep.found()
  nntrainer_unittest_deps = [
    nntrainer_dep,
    gtest_dep
  ]
endif

if get_option('enable-capi')
  subdir('tizen_capi')
  subdir('unittest')
endif

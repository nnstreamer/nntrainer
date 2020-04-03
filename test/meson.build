gtest_dep = dependency('gtest', required: false)
if gtest_dep.found()
  nntrainer_unittest_deps = [
    nntrainer_dep,
    gtest_dep
  ]
endif

if get_option('enable-capi')
  subdir('tizen_capi')
endif

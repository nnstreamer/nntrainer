nntrainer_test_inc = include_directories('./include')

nntrainer_test_deps = [
  nntrainer_dep #gtest is linked in nntrainer_testutil_lib
]

# build test util when gtest is found
if gtest_dep.found()
  nntrainer_testutil_lib = static_library(
    'nntrainer_test_util',
    'nntrainer_test_util.cpp',
    dependencies: [nntrainer_test_deps, gtest_dep],
    include_directories: nntrainer_test_inc
  )
  nntrainer_testutil_dep = declare_dependency(
    link_with: nntrainer_testutil_lib,
    include_directories: nntrainer_test_inc
  )
  nntrainer_test_deps += nntrainer_testutil_dep
endif

if get_option('enable-capi')
  subdir('tizen_capi')
  subdir('unittest')
endif

if get_option('enable-ccapi')
  subdir('ccapi')
endif

if get_option('enable-nnstreamer-tensor-filter')
  install_subdir('test_models', install_dir: application_install_dir)
endif

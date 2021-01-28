nntrainer_test_inc = include_directories('./include')

nntrainer_test_deps = [
  nntrainer_dep
]

nntrainer_testutil_lib = static_library(
  'nntrainer_test_util',
  'nntrainer_test_util.cpp',
  dependencies: [nntrainer_test_deps],
  include_directories: nntrainer_test_inc
)

nntrainer_testutil_dep = declare_dependency(
  link_with: nntrainer_testutil_lib,
  include_directories: nntrainer_test_inc
)

nntrainer_test_deps += nntrainer_testutil_dep
nntrainer_test_deps += gtest_dep

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

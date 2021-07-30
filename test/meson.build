nntrainer_test_resdir = nntrainer_resdir / 'test'
run_command('mkdir', '-p', nntrainer_test_resdir)

nntrainer_test_inc = include_directories('./include')


nntrainer_testutil_lib = static_library(
  'nntrainer_test_util',
  'nntrainer_test_util.cpp',
  dependencies: [nntrainer_dep],
  include_directories: nntrainer_test_inc
)

nntrainer_testutil_dep = declare_dependency(
  link_with: nntrainer_testutil_lib,
  include_directories: nntrainer_test_inc
)

nntrainer_test_deps = [
  nntrainer_dep,
  nntrainer_testutil_dep,
  gtest_dep
]

# this is depedency is for the gtest with main included
nntrainer_test_main_deps = [
  nntrainer_dep,
  nntrainer_testutil_dep,
  gtest_main_dep
]

if get_option('enable-capi')
  subdir('tizen_capi')
  subdir('unittest')
endif

if get_option('enable-ccapi')
  subdir('ccapi')
endif

if get_option('enable-nnstreamer-tensor-filter')
  run_command('cp', '-lr',
    meson.current_source_dir() / 'test_models/',
    nntrainer_test_resdir
  )
endif

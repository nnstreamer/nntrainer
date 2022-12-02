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
  gmock_dep,
  gtest_dep,
  nntrainer_dep,
  nntrainer_testutil_dep
]

# this is depedency is for the gtest with main included
nntrainer_test_main_deps = [
  gmock_dep,
  gtest_main_dep,
  nntrainer_dep,
  nntrainer_testutil_dep
]

if enable_capi
  subdir('tizen_capi')
endif

if enable_ccapi
  subdir('ccapi')
  subdir('unittest')
endif

nnstreamer_test_dep = dependency('nnstreamer-test-devel', required: false)

if get_option('enable-nnstreamer-tensor-filter').enabled() and nnstreamer_test_dep.found()
  subdir('nnstreamer')
endif

if get_option('enable-tflite-interpreter') or get_option('enable-nnstreamer-tensor-filter').enabled()
  run_command('cp','-lr',
    meson.current_source_dir() / 'test_models/',
    nntrainer_test_resdir
  )
endif

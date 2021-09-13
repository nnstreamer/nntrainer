memory_test_inc = include_directories('./')

nntrainer_memory_planner_tests_lib = shared_library(
  'nntrainer_memory_planner_validation',
  'memory_planner_validate.cpp',
  dependencies: [nntrainer_dep, gtest_dep], # nntrainer_devel_dep
  include_directories: memory_test_inc
)

nntrainer_memory_tests_dep = declare_dependency(
  link_with: nntrainer_memory_planner_tests_lib,
  include_directories: memory_test_inc
  )

test_target = [
  'memory_planner_validate.cpp',
  'unittest_memory_planner.cpp',
]

exe = executable(
  'unittest_memory', test_target,
  dependencies: [
    nntrainer_test_main_deps
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test('unittest_memory', exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_memory')
)

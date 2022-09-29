memory_test_inc = include_directories('./')

nntrainer_memory_tests_dep = declare_dependency(
  include_directories: memory_test_inc
  )

test_target = [
  'unittest_memory_planner.cpp',
  'unittest_memory_pool.cpp',
  'unittest_cache_pool.cpp'
]

# memory unittests
exe = executable(
  'unittest_memory', test_target,
  dependencies: [
    nntrainer_test_main_deps
  ],
  cpp_args : '-Wno-maybe-uninitialized',
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test('unittest_memory', exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_memory')
)

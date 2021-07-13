test_name = 'unittest_datasets'

test_target = []

producer_targets = [
  'data_producer_common_tests.cpp',
  'unittest_random_data_producers.cpp',
  'unittest_func_data_producer.cpp',
  'unittest_raw_file_data_producer.cpp',
  'unittest_batch_queue.cpp'
]

test_target += producer_targets
exe = executable(
  test_name,
  test_target,
  dependencies: [
    nntrainer_test_main_deps,
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test(test_name, exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), test_name)
)

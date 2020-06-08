unittest_nntrainer_deps = [nntrainer_test_deps] # if unittest-wide dep is added, this is the place to add

unzip_target = ['trainset.tar.gz', 'valset.tar.gz', 'testset.tar.gz']
src_path = join_paths(meson.source_root(), 'packaging')
dest_path = meson.build_root()

foreach target: unzip_target
  _src_path = join_paths(src_path, target)
  run_command(['tar', 'xzf', _src_path, '-C', dest_path])
endforeach

run_command(['cp', join_paths(src_path, 'label.dat'), join_paths(dest_path, 'label.dat')])

unittest_nntrainer_internal = executable('unittest_nntrainer_internal',
  'unittest_nntrainer_internal.cpp',
  dependencies: unittest_nntrainer_deps,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('unittest_nntrainer_internal', unittest_nntrainer_internal)

unittest_nntrainer_layers = executable('unittest_nntrainer_layers',
  'unittest_nntrainer_layers.cpp',
  dependencies: unittest_nntrainer_deps,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('unittest_nntrainer_layers', unittest_nntrainer_layers)

unittest_nntrainer_tensor = executable('unittest_nntrainer_tensor',
  'unittest_nntrainer_tensor.cpp',
  dependencies: unittest_nntrainer_deps,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('unittest_nntrainer_tensor', unittest_nntrainer_tensor)

unittest_util_func = executable('unittest_util_func',
  'unittest_util_func.cpp',
  dependencies: unittest_nntrainer_deps,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('unittest_util_func', unittest_util_func)

unittest_databuffer_file = executable('unittest_databuffer_file',
  ['../nntrainer_test_util.cpp', 'unittest_databuffer_file.cpp'],
  dependencies: [unittest_nntrainer_deps],
  include_directories: nntrainer_test_inc,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('unittest_databuffer_file', unittest_databuffer_file)

unittest_nntrainer_lazy_tensor = executable('unittest_nntrainer_lazy_tensor',
  ['../nntrainer_test_util.cpp', 'unittest_nntrainer_lazy_tensor.cpp'],
  dependencies: [unittest_nntrainer_deps],
  include_directories: nntrainer_test_inc,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)
test('unittest_nntrainer_lazy_tensor', unittest_nntrainer_lazy_tensor)

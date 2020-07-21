unittest_nntrainer_deps = [nntrainer_test_deps] # if unittest-wide dep is added, this is the place to add

# test material need to be unzipped from "(project_home)/packaging/"
unzip_target = ['trainset.tar.gz', 'valset.tar.gz', 'testset.tar.gz', 'unittest_layers.tar.gz']

src_path = join_paths(meson.source_root(), 'packaging')
dest_path = meson.build_root()

foreach target: unzip_target
  _src_path = join_paths(src_path, target)
  run_command(['tar', 'xzf', _src_path, '-C', dest_path])
endforeach

run_command(['cp', join_paths(src_path, 'label.dat'), join_paths(dest_path, 'label.dat')])

test_target = [
  'unittest_nntrainer_activations',
  'unittest_nntrainer_internal',
  'unittest_nntrainer_layers',
  'unittest_nntrainer_lazy_tensor',
  'unittest_nntrainer_tensor',
  'unittest_util_func',
  'unittest_databuffer_file',
  'unittest_nntrainer_modelfile'
]

foreach target: test_target
  exe = executable(
    target,
    target + '.cpp',
    dependencies: unittest_nntrainer_deps,
    install_dir: application_install_dir,
    install: get_option('enable-test')
  )
  test(target, exe)
endforeach


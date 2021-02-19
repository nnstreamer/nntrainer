unittest_nntrainer_deps = [
  nntrainer_test_deps,
  nntrainer_capi_dep, # so that feature_state can be overridden
] # if unittest-wide dep is added, this is the place to add

# test material need to be unzipped from "(project_home)/packaging/"
unzip_target = [
  ['trainset.tar.gz', ''],
  ['valset.tar.gz', ''],
  ['testset.tar.gz', ''],
  ['unittest_layers.tar.gz', 'unittest_layers'],
  ['unittest_models.tar.gz', 'unittest_models'],
]

src_path = meson.source_root() / 'packaging'
dest_path = nntrainer_test_resdir

foreach target: unzip_target
  _src_path = src_path / target[0]
  run_command('mkdir', '-p', dest_path / target[1])
  run_command(['tar', 'xzf', _src_path, '-C', dest_path / target[1]])
endforeach

run_command(['cp', '-l', src_path / 'label.dat', dest_path / 'label.dat'])

test_target = [
  'unittest_nntrainer_activations',
  'unittest_nntrainer_internal',
  'unittest_nntrainer_layers',
  'unittest_nntrainer_lazy_tensor',
  'unittest_nntrainer_tensor',
  'unittest_util_func',
  'unittest_databuffer_file',
  'unittest_nntrainer_modelfile',
  'unittest_nntrainer_models',
  'unittest_nntrainer_graph',
  'unittest_nntrainer_appcontext'
]

if get_option('enable-profile')
  test_target += 'unittest_nntrainer_profiler'
endif

foreach target: test_target
  exe = executable(
    target,
    target + '.cpp',
    dependencies: unittest_nntrainer_deps,
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )
  test(target, exe)
endforeach


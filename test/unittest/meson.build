unittest_nntrainer_deps = [
  nntrainer_test_deps,
  nntrainer_ccapi_dep
] # if unittest-wide dep is added, this is the place to add

if get_option('platform') == 'tizen'
  unittest_nntrainer_deps += [ nntrainer_capi_dep ] # so that feature_state can be overridden
endif

# test material need to be unzipped from "(project_home)/packaging/"
unzip_target = [
  ['trainset.tar.gz', ''],
  ['valset.tar.gz', ''],
  ['testset.tar.gz', ''],
  ['unittest_layers.tar.gz', 'unittest_layers'],
  ['unittest_layers_w16a16.tar.gz', 'unittest_layers'],
  ['unittest_layers_embedding.tar.gz', 'unittest_layers'],
  ['unittest_models.tar.gz', 'unittest_models'],
  ['unittest_models_v2.tar.gz', 'unittest_models'],
  ['unittest_models_v3.tar.gz', 'unittest_models'],
  ['unittest_models_multiout.tar.gz', 'unittest_models'],
  ['unittest_models_exeorder.tar.gz', 'unittest_models'],
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
  ['unittest_nntrainer_activations', []],
  ['unittest_nntrainer_exe_order', []],
  ['unittest_nntrainer_internal', []],
  ['unittest_nntrainer_lazy_tensor', []],
  ['unittest_nntrainer_tensor', []],
  ['unittest_nntrainer_tensor_nhwc', []],
  ['unittest_util_func', []],
  ['unittest_nntrainer_modelfile', []],
  ['unittest_nntrainer_models', [
    'models' / 'models_test_utils.cpp', 'models' / 'models_golden_test.cpp'
  ]],
  ['unittest_nntrainer_graph', []],
  ['unittest_nntrainer_appcontext', []],
  ['unittest_base_properties', []],
  ['unittest_common_properties', []],
  ['unittest_nntrainer_tensor_pool', []],
  ['unittest_nntrainer_lr_scheduler', []],
  ['unittest_nntrainer_task', []],
]

if get_option('enable-fp16')
  test_target += [['unittest_nntrainer_tensor_fp16', []]]
  test_target += [['unittest_nntrainer_tensor_pool_fp16', []]]
endif

if get_option('enable-profile')
  if gmock_dep.version().version_compare('>=1.10.0')
    test_target += [['unittest_nntrainer_profiler', []]]
  endif
endif

foreach target: test_target
  exe = executable(
    target[0],
    [target[0] + '.cpp'] + target[1],
    # below is temporary measure, we will eventually remove unittest_nntrainer_models
    include_directories: include_directories('models'),
    dependencies: unittest_nntrainer_deps,
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )
  test(target[0], exe,
    args: ['--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), target[0])],
    timeout: test_timeout)
endforeach

unittest_inc = include_directories('.')

subdir('memory')
subdir('compiler')
subdir('layers')
subdir('datasets')
subdir('models')
subdir('integration_tests')

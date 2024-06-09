layer_common_test_inc = [ include_directories('./', '../../include') ]
layer_common_test_standalone_files = files('layers_standalone_common_tests.cpp')
layer_common_test_dependent_files = files('layers_dependent_common_tests.cpp')

nntrainer_layer_common_standalone_tests_lib = shared_library(
  'nntrainer_layer_common_standalone_tests',
  'layers_standalone_common_tests.cpp',
  cpp_args : '-Wno-maybe-uninitialized',
  dependencies: [nntrainer_dep, gtest_dep], # nntrainer_devel_dep
  include_directories: layer_common_test_inc
)

nntrainer_layer_common_standalone_tests_dep = declare_dependency(
  link_with: nntrainer_layer_common_standalone_tests_lib,
  include_directories: layer_common_test_inc
  )

nntrainer_layer_common_dependent_tests_lib = shared_library(
  'nntrainer_layer_common_dependent_tests',
  'layers_dependent_common_tests.cpp',
  cpp_args : '-Wno-maybe-uninitialized',
  dependencies: [nntrainer_dep, gtest_dep], # nntrainer_devel_dep
  include_directories: layer_common_test_inc
)

nntrainer_layer_common_dependent_tests_dep = declare_dependency(
  link_with: nntrainer_layer_common_dependent_tests_lib,
  include_directories: layer_common_test_inc
  )

test_target = [
  layer_common_test_dependent_files,
  layer_common_test_standalone_files,
  'layers_golden_tests.cpp',
  'unittest_layer_node.cpp',
  'unittest_layers.cpp',
  'unittest_layers_impl.cpp',
  'unittest_layers_input.cpp',
  'unittest_layers_loss.cpp',
  'unittest_layers_fully_connected.cpp',
  'unittest_layers_batch_normalization.cpp',
  'unittest_layers_layer_normalization.cpp',
  'unittest_layers_convolution2d.cpp',
  'unittest_layers_convolution1d.cpp',
  'unittest_layers_pooling2d.cpp',
  'unittest_layers_flatten.cpp',
  'unittest_layers_activation.cpp',
  'unittest_layers_addition.cpp',
  'unittest_layers_multiout.cpp',
  'unittest_layers_rnn.cpp',
  'unittest_layers_rnncell.cpp',
  'unittest_layers_lstm.cpp',
  'unittest_layers_lstmcell.cpp',
  'unittest_layers_gru.cpp',
  'unittest_layers_grucell.cpp',
  'unittest_layers_preprocess_flip.cpp',
  'unittest_layers_split.cpp',
  'unittest_layers_embedding.cpp',
  'unittest_layers_concat.cpp',
  'unittest_layers_permute.cpp',
  'unittest_layers_attention.cpp',
  'unittest_layers_dropout.cpp',
  'unittest_layers_reshape.cpp',
  # 'unittest_layers_mol_attention.cpp',
  'unittest_layers_multi_head_attention.cpp',
  'unittest_layers_positional_encoding.cpp',
  'unittest_layers_upsample2d.cpp'
]

if get_option('enable-tflite-backbone')
  test_target += 'unittest_layers_tflite.cpp'
endif

if opencv_dep.found()
  test_target += 'unittest_layers_preprocess_translate.cpp'
endif

if get_option('enable-nnstreamer-backbone')
  if get_option('platform') != 'tizen'
    # ml singleshot api cannot be tested inside tizen because of feature issue
    test_target += 'unittest_layers_nnstreamer.cpp'
  endif
endif

exe = executable(
  'unittest_layers', test_target,
  dependencies: [
    nntrainer_test_main_deps,
  ],
  cpp_args : '-Wno-maybe-uninitialized',
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

# @todo delete v2 once migration is done
test('unittest_layers', exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_layers'),
  timeout: test_timeout
)

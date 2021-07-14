layer_common_test_inc = include_directories('./')

nntrainer_layer_common_standalone_tests_lib = shared_library(
  'nntrainer_layer_common_standalone_tests',
  'layers_standalone_common_tests.cpp',
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
  dependencies: [nntrainer_dep, gtest_dep, nntrainer_layer_common_standalone_tests_dep], # nntrainer_devel_dep
  include_directories: layer_common_test_inc
)

nntrainer_layer_common_dependent_tests_dep = declare_dependency(
  link_with: nntrainer_layer_common_dependent_tests_lib,
  include_directories: layer_common_test_inc
  )

test_target = [
  'unittest_layers_impl.cpp',
  'unittest_layers_input.cpp',
  'unittest_layers_loss.cpp',
  'unittest_layers_fully_connected.cpp',
  'unittest_layers_batch_normalization.cpp',
  'unittest_layers_convolution2d.cpp',
  'unittest_layers_pooling2d.cpp',
  'unittest_layers_flatten.cpp',
  'unittest_layers_activation.cpp',
  'unittest_layers_addition.cpp',
  'unittest_layers_multiout.cpp',
  'unittest_layers_rnn.cpp',
  'unittest_layers_lstm.cpp',
  'unittest_layers_gru.cpp',
  'unittest_layers_preprocess.cpp',
  'unittest_layers_split.cpp',
  'unittest_layers_embedding.cpp',
  'unittest_layers_concat.cpp',
  'unittest_layers_tflite.cpp',
  'unittest_layers_permute.cpp',
  'unittest_layers_nnstreamer.cpp',
]

exe = executable(
  'unittest_layers', test_target,
  dependencies: [
    nntrainer_test_main_deps,
    nntrainer_layer_common_standalone_tests_dep,
    nntrainer_layer_common_dependent_tests_dep
  ],
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

# @todo delete v2 once migration is done
test('unittest_layers_v2', exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_layers_v2')
)

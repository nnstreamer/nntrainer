subdir('loss')
nntrainer_inc += include_directories('loss')
nntrainer_inc_abs += meson.current_source_dir() / 'loss'

layer_sources = [
  'activation_layer.cpp',
  'addition_layer.cpp',
  'attention_layer.cpp',
  'mol_attention_layer.cpp',
  'multi_head_attention_layer.cpp',
  'concat_layer.cpp',
  'bn_layer.cpp',
  'layer_normalization_layer.cpp',
  'conv2d_layer.cpp',
  'conv1d_layer.cpp',
  'fc_layer.cpp',
  'flatten_layer.cpp',
  'input_layer.cpp',
  'multiout_layer.cpp',
  'layer_node.cpp',
  'pooling2d_layer.cpp',
  'preprocess_flip_layer.cpp',
  'preprocess_translate_layer.cpp',
  'preprocess_l2norm_layer.cpp',
  'embedding.cpp',
  'rnn.cpp',
  'rnncell.cpp',
  'lstm.cpp',
  'lstmcell.cpp',
  'lstmcell_core.cpp',
  'zoneout_lstmcell.cpp',
  'time_dist.cpp',
  'common_properties.cpp',
  'split_layer.cpp',
  'permute_layer.cpp',
  'layer_impl.cpp',
  'gru.cpp',
  'grucell.cpp',
  'dropout.cpp',
  'centroid_knn.cpp',
  'layer_context.cpp',
  'reshape_layer.cpp',
  'reduce_mean_layer.cpp',
  'positional_encoding_layer.cpp',
  'identity_layer.cpp',
  'upsample2d_layer.cpp'
]

layer_headers = [
  'layer_context.h',
  'layer_devel.h',
  'layer_impl.h',
  'common_properties.h',
]

layer_deps = []

# for android, it is disabled until we have support for the nnstreamer package
if get_option('enable-nnstreamer-backbone') and get_option('platform') != 'android'
  if not nnstreamer_capi_dep.found()
    error('NNStreamer CAPI dependency not found')
  endif
  layer_deps += nnstreamer_capi_dep
  layer_sources += 'nnstreamer_layer.cpp'
endif

if get_option('enable-tflite-backbone')
  if not tflite_dep.found()
    error('Tensorflow-Lite dependency not found')
  endif
  layer_deps += tflite_dep
  layer_sources += 'tflite_layer.cpp'
endif

# dependency for data augmentation, if present
layer_deps += opencv_dep

nntrainer_base_deps += layer_deps

foreach s : layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : layer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

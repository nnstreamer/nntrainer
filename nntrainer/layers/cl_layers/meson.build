cl_layer_sources = [
    'fc_layer_cl.cpp',
  # 'addition_layer_cl.cpp',
   'swiglu_cl.cpp',
   'reshape_cl.cpp',
   'rmsnorm_layer_cl.cpp',
   'concat_cl.cpp',
]

if get_option('enable-fp16')
  cl_layer_sources += [
    'swiglu_cl_fp16.cpp',
    'reshape_cl_fp16.cpp',
    'rmsnorm_layer_cl_fp16.cpp',
    'concat_cl_fp16.cpp',
  ]
endif

foreach s : cl_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

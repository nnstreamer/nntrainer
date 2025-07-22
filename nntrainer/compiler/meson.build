compiler_sources = [
  'ini_interpreter.cpp',
  'activation_realizer.cpp',
  'flatten_realizer.cpp',
  'recurrent_realizer.cpp',
  'remap_realizer.cpp',
  'slice_realizer.cpp',
  'input_realizer.cpp',
  'previous_input_realizer.cpp',
  'multiout_realizer.cpp',
  'bn_realizer.cpp',
  'loss_realizer.cpp',
]

compiler_headers = [
  'compiler_fwd.h'
]

if get_option('enable-tflite-interpreter')
  if not tflite_dep.found()
    error('Tensorflow2-Lite dependency not found')
  endif
  if not flatc_prog.found()
    error('flatc executable not found')
  endif

  flat_header = custom_target('tflite-schema',
                               input: 'tf_schema.fbs',
                               output: 'tf_schema_generated.h',
                               command: [flatc_prog, '-o', '@OUTDIR@', '-c', '@INPUT@'])

  nntrainer_inc_abs += meson.current_build_dir()
  nntrainer_inc += include_directories('.')

  flat_header_dep = declare_dependency(sources : flat_header)

  nntrainer_base_deps += flat_header_dep
  compiler_sources += [
    'tflite_interpreter.cpp',
    'tflite_opnode.cpp'
  ]

if not flatc_prog.found()
  error('flatc executable not found')
endif
endif

foreach s : compiler_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : compiler_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

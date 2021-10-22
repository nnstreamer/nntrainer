compiler_sources = [
  'ini_interpreter.cpp',
  'flatten_realizer.cpp',
  'recurrent_realizer.cpp',
  'remap_realizer.cpp',
  'slice_realizer.cpp',
  'input_realizer.cpp'
]

compiler_headers = []

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
                               command: [flatc_prog, '-o', '@CURRENT_SOURCE_DIR@', '-c', '@INPUT@'])

  nntrainer_inc_abs += meson.current_build_dir()

  flat_header_dep = declare_dependency(sources : flat_header)

  nntrainer_base_deps += flat_header_dep
  compiler_sources += [
    'tflite_interpreter.cpp',
    'tflite_opnode.cpp'
  ]
endif

foreach s : compiler_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : compiler_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

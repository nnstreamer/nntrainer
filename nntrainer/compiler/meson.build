compiler_sources = [
  'ini_interpreter.cpp'
]

compiler_headers = []

if get_option('enable-tflite-interpreter')
  flatc = find_program('flatc', required: true)
  add_project_arguments('-DENABLE_TFLITE_INTERPRETER=1', language:['c','cpp'])

  flat_header = custom_target('tflite-schema',
                               input: 'tf_schema.fbs',
                               output: 'tf_schema_generated.h',
                               command: [flatc, '-c', '@INPUT@'])

  nntrainer_sources += flat_header
endif

foreach s : compiler_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : compiler_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


nntrainer_schema_dir = meson.build_root() / 'nntrainer' / 'schema'

if get_option('enable-onnx-interpreter')
  protobuf_dep = dependency('protobuf', required: true)
  nntrainer_base_deps += protobuf_dep

  protoc = find_program('protoc', required: true)  
  onnx_proto = custom_target('onnx_proto',
                              input: 'onnx.proto',
                              output: ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
                              command: [
                                protoc,
                                '--cpp_out=' + nntrainer_schema_dir,
                                '-I=' + meson.current_source_dir(),
                                meson.current_source_dir() + '/onnx.proto'
                              ])

  nntrainer_inc_abs += meson.current_build_dir()
  nntrainer_inc += include_directories('.')
  onnx_proto_dep = declare_dependency(sources : onnx_proto)
  nntrainer_base_deps += onnx_proto_dep
endif

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
                               command: [flatc_prog, '-o', nntrainer_schema_dir, '-c', '@INPUT@'])

  nntrainer_inc_abs += meson.current_build_dir()
  nntrainer_inc += include_directories('.')

  flat_header_dep = declare_dependency(sources : flat_header)

  nntrainer_base_deps += flat_header_dep

if not flatc_prog.found()
  error('flatc executable not found')
endif
  flat_header2 = custom_target('nntrainer-schema',
                               input: 'nntrainer_schema.fbs',
                               output: 'nntrainer_schema_generated.h',
                               command: [flatc_prog, '-o', nntrainer_schema_dir, '-c', '@INPUT@'])

  nntrainer_inc_abs += meson.current_build_dir()
  nntrainer_inc += include_directories('.')

  flat_header2_dep = declare_dependency(sources : flat_header2)

  nntrainer_base_deps += flat_header2_dep

endif

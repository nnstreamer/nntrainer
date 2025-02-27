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

foreach s : compiler_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : compiler_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

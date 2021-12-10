optimizer_sources = [
  'adam.cpp',
  'optimizer_devel.cpp',
  'optimizer_impl.cpp',
  'sgd.cpp',
  'optimizer_context.cpp',
  'lr_scheduler_constant.cpp',
  'lr_scheduler_exponential.cpp',
  'optimizer_wrapped.cpp'
]

optimizer_headers = [
  'optimizer_devel.h',
  'optimizer_impl.h',
  'optimizer_context.h',
]

foreach s : optimizer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : optimizer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


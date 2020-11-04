dataset_sources = [
  'databuffer.cpp',
  'databuffer_factory.cpp',
  'databuffer_file.cpp',
  'databuffer_func.cpp'
]

dataset_headers = [
  'databuffer.h',
  'databuffer_factory.h'
]

foreach s : dataset_sources
  nntrainer_sources += join_paths(meson.current_source_dir(), s)
endforeach

foreach h : dataset_headers
  nntrainer_headers += join_paths(meson.current_source_dir(), h)
endforeach


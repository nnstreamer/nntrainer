dataset_sources = [
  'batch_queue.cpp',
  'databuffer.cpp',
  'data_iteration.cpp',
  'databuffer_factory.cpp',
  'random_data_producers.cpp',
  'func_data_producer.cpp',
  'raw_file_data_producer.cpp'
]

dataset_headers = [
  'databuffer.h',
  'databuffer_factory.h'
]

foreach s : dataset_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : dataset_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


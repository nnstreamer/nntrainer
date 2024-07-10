hgemm_padding_headers = [
  'hgemm_padding.h',
  'hgemm_padding_a.h',
  'hgemm_padding_b.h',
]


hgemm_padding_sources = [
  'hgemm_padding_a.cpp',
  'hgemm_padding_b.cpp',
]

foreach s : hgemm_padding_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : hgemm_padding_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


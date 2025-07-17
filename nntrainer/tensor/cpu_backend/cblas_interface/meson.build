cblas_interface_headers = [
    'cblas_interface.h',
]
cblas_interface_sources = [
    'cblas_interface.cpp',
]

foreach s : cblas_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : cblas_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

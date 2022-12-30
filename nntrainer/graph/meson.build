graph_sources = [
  'network_graph.cpp',
  'graph_core.cpp',
  'connection.cpp'
]

graph_headers = ['connection.h']

foreach s : graph_sources
  nntrainer_sources += files(s)
  nntrainer_sources_abs += meson.current_source_dir() / s
endforeach

foreach h : graph_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

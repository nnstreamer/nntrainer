graph_sources = [
  'network_graph.cpp',
  'graph_core.cpp'
]

graph_headers = []

foreach s : graph_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : graph_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

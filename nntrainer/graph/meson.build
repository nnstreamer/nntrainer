graph_sources = [
  'network_graph.cpp',
  'graph_core.cpp',
  'subgraph_base.cpp',
  'connection.cpp'
]

graph_headers = [
  'connection.h',
  'graph_node.h',
  'network_graph.h',
  'graph_core.h',
  'subgraph_base.h',
  'graph_node.h',
]

foreach s : graph_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : graph_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

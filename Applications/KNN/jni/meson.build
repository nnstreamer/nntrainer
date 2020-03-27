knn_sources = [
  'main.cpp',
  'bitmap_helpers.cpp'
]

knn_inc = include_directories('.')

executable('nntrainer_knn',
  knn_sources,
  dependencies: [iniparser_dep, nntrainer_dep, tflite_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

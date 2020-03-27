classification_sources = [
  'main.cpp',
  'bitmap_helpers.cpp'
]

executable('nntrainer_classification',
  classification_sources,
  dependencies: [iniparser_dep, nntrainer_dep, tflite_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

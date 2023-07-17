llama_sources = [
  'main.cpp'
]

llama_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

e = executable('nntrainer_llama',
  llama_sources,
  include_directories: [include_directories('.')],
  dependencies: llama_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

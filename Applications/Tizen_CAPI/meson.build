tizen_capi_file_sources = [
  'main.c'
]

executable('nntrainer_tizen_capi_file',
  tizen_capi_file_sources,
  dependencies: [iniparser_dep, nntrainer_dep, nntrainer_capi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

executable('nntrainer_classification_capi',
  'c_api.c',
  dependencies: [nntrainer_capi_dep, nntrainer_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

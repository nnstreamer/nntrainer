tizen_capi_file_sources = [
  'main.c'
]

executable('nntrainer_tizen_capi_file',
  tizen_capi_file_sources,
  dependencies: [iniparser_dep, nntrainer_dep, nntrainer_capi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

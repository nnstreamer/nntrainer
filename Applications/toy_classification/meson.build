build_root = meson.build_root()

classification_sources = [
  'main.cpp',
]

e = executable('toy_classification',
  classification_sources,
  dependencies: [iniparser_dep, nntrainer_dep, nntrainer_capi_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

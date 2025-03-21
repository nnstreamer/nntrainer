app_sources = [
  'main.cpp',
]

app_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  app_dependencies += [gtest_dep]
endif

e = executable('nntrainer_save_quant',
  app_sources,
  include_directories: [include_directories('.')],
  dependencies: app_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

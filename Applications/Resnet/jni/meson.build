resnet_sources = [
  'main.cpp',
]

e = executable('nntrainer_resnet18',
  resnet_sources,
  dependencies: [app_utils_dep, iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_resnet18', e)

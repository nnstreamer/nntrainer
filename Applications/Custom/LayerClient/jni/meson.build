res_path = meson.current_source_dir() / '..' / 'res'

layer_client_sources = [
  'main.cpp'
]

layer_client_inc = include_directories('.')

ini_in_path = res_path / 'custom_layer_client.ini'
ini_out_path = meson.build_root() / 'custom_layer_client.ini'

run_command('cp', ini_in_path, ini_out_path)

e = executable('layer_client',
  layer_client_sources,
  dependencies: [app_utils_dep, iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

res_path = meson.current_source_dir() / '..' / 'res'

layer_client_sources = [
  'main.cpp',
  layer_example_src
]

ini_in_path = res_path / 'custom_layer_client.ini'
ini_out_path = meson.build_root() / 'custom_layer_client.ini'

run_command('cp', '-lr', res_path, nntr_app_resdir / 'LayerClient')

e = executable('layer_client',
  layer_client_sources,
  include_directories: layer_example_inc,
  dependencies: [app_utils_dep, iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

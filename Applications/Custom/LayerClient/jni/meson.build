res_path = meson.current_source_dir() / '..' / 'res'

layer_client_sources = [
  'main.cpp',
  layer_example_pow_src,
  layer_example_mae_src
]

run_command('cp', '-lr', res_path, nntr_app_resdir / 'LayerClient')

e = executable('layer_client',
  layer_client_sources,
  include_directories: layer_example_inc,
  dependencies: [app_utils_dep, iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_layer_client_api', exe, args: 'model')
test('app_layer_client_ini', exe, args: res_path / 'custom_layer_client.ini')

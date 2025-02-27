fs = import('fs')
app_res_dir = fs.parent(meson.current_source_dir()) / 'res'
build_app_res_dir = nntr_app_resdir / 'LayerClient'

if host_machine.system() == 'windows'
  app_res_dir_win = app_res_dir.replace('/', '\\')
  build_app_res_dir_win = build_app_res_dir.replace('/', '\\')
  run_command('cmd.exe', '/C', 'mkdir', build_app_res_dir_win)
  run_command('cmd.exe', '/C', 'xcopy',app_res_dir_win, build_app_res_dir_win)
else
  run_command('cp', '-lr', app_res_dir, build_app_res_dir)
endif

layer_client_sources = [
  'main.cpp',
  layer_example_pow_src,
  layer_example_mae_src
]

exe = executable('layer_client',
  layer_client_sources,
  include_directories: layer_example_inc,
  dependencies: [app_utils_dep, iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_layer_client_api', exe, args: 'model')
test('app_layer_client_ini', exe, args: build_app_res_dir / 'custom_layer_client.ini')

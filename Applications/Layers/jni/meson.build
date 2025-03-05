fs = import('fs')
app_res_dir = fs.parent(meson.current_source_dir()) / 'res'
build_app_res_dir = nntr_app_resdir / 'Layers'

if host_machine.system() == 'windows'
  app_res_dir_win = app_res_dir.replace('/', '\\')
  build_app_res_dir_win = build_app_res_dir.replace('/', '\\')
  run_command('cmd.exe', '/C', 'mkdir', build_app_res_dir_win)
  run_command('cmd.exe', '/C', 'xcopy',app_res_dir_win, build_app_res_dir_win)
else
  run_command('cp', '-lr', app_res_dir, build_app_res_dir)
endif

Layers_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp'
]

executable('nntrainer_Layers',
  Layers_sources,
  dependencies: [iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep, app_utils_dep],
  include_directories: [include_directories('.'), cifar_include_dir],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

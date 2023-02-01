build_root = meson.build_root()
res_path = meson.current_source_dir() / '..' / 'res'

nntr_Layers_resdir = nntr_app_resdir / 'Layers'
run_command('cp', '-lr', res_path, nntr_Layers_resdir)

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

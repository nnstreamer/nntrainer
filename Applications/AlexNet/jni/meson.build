build_root = meson.build_root()
res_path = meson.current_source_dir() / '..' / 'res'

nntr_alex_resdir = nntr_app_resdir / 'AlexNet'
run_command('cp', '-lr', res_path, nntr_alex_resdir)

alex_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp'
]

executable('nntrainer_alexnet',
  alex_sources,
  dependencies: [iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep, app_utils_dep],
  include_directories: [include_directories('.'), cifar_include_dir],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

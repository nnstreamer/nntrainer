ini_in = join_paths(meson.current_source_dir(), 'Tizen_CAPI_config.ini')
ini_out = join_paths(meson.build_root(), 'Tizen_CAPI_config.ini')


e = executable('nntrainer_tizen_capi_file',
  'main.c',
  dependencies: [iniparser_dep, nntrainer_dep, nntrainer_capi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

run_command('cp', ini_in, ini_out)
test('app_classification_capi_ini', e, timeout: 60)


executable('nntrainer_classification_capi_file',
  'capi_file.c',
  dependencies: [nntrainer_capi_dep, nntrainer_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)
test('app_classification_capi_file', e, timeout: 60)

e = executable('nntrainer_classification_capi_func',
  'capi_func.c',
  dependencies: [nntrainer_capi_dep, nntrainer_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_classification_capi_func', e, timeout: 60)

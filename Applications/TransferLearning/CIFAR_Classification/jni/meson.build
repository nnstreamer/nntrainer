build_root = meson.build_root()
res_path = meson.current_source_dir() / '..' / 'res'

classification_sources = [
  'main.cpp'
]

e = executable('nntrainer_classification',
  classification_sources,
  dependencies: [app_utils_dep, iniparser_dep, nntrainer_dep, tflite_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

ini_in_path = res_path / 'Classification.ini'
ini_out_path = build_root / 'classification.ini'

# change epochs to 5
run_command('cp', ini_in_path, ini_out_path)
run_command(['sed', '-i', 's/Epochs\ \=\ 30000/Epochs\ \=\ 1/', ini_out_path])

test('app_classification', e, args: [ini_out_path, build_root + '/'])

classification_func_sources = [
  'main_func.cpp'
]

e = executable('nntrainer_classification_func',
  classification_func_sources,
  dependencies: [iniparser_dep, nntrainer_ccapi_dep, app_utils_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

ini_in_path = res_path / 'Classification_func.ini'
ini_out_path = build_root / 'classification_func.ini'

# change epochs to 5
run_command('cp', ini_in_path, ini_out_path)
run_command(['sed', '-i', 's/Epochs\ \=\ 30000/Epochs\ \=\ 1/', ini_out_path])

test('app_classification_func', e, args: [ini_out_path], timeout: 60)

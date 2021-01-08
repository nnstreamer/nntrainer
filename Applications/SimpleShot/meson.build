simpleshot_sources = [
  'task_runner.cpp'
]

simpleshot_inc = include_directories('.')

e = executable('simpleshot_runner',
  simpleshot_sources,
  dependencies: [app_utils_dep, nntrainer_dep, tflite_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

if get_option('enable-test')
  subdir('test')
endif

simpleshot_sources = [
  'simpleshot_utils.cpp'
]

simpleshot_inc = include_directories('.')


e = executable('simpleshot_runner',
  ['task_runner.cpp'] + simpleshot_sources,
  dependencies: [app_utils_dep, nntrainer_dep, tflite_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

if get_option('enable-test')
  simpleshot_test_dep = declare_dependency(
    sources: simpleshot_sources,
    include_directories: simpleshot_inc,
  )
  subdir('test')
endif

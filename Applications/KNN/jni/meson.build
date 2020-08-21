res_path = join_paths(meson.current_source_dir(), '..', 'res')

knn_sources = [
  'main_sample.cpp',
  'bitmap_helpers.cpp'
]

knn_inc = include_directories('.')

e = executable('knn_sample',
  knn_sources,
  dependencies: [iniparser_dep, nntrainer_dep, tflite_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_knn', e, args: [res_path + '/'])

res_path = join_paths(meson.current_source_dir(), '..', 'res')
e = executable('nntrainer_logistic',
  'main.cpp',
  dependencies: [iniparser_dep, nntrainer_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_logistic', e, args: [join_paths(res_path, 'LogisticRegression.ini'), join_paths(res_path, 'dataset1.txt')])

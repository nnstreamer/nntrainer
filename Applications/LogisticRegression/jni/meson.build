res_path = meson.current_source_dir() / '..' / 'res'
run_command('cp', '-lr', res_path, nntr_app_resdir / 'LogisticRegression')

e = executable('nntrainer_logistic',
  'main.cpp',
  dependencies: [iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_logistic', e, args: ['train',
  nntr_app_resdir / 'LogisticRegression' / 'LogisticRegression.ini',
  nntr_app_resdir / 'LogisticRegression' / 'dataset1.txt']
)

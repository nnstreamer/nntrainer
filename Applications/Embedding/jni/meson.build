res_path = meson.current_source_dir() / '..' / 'res'
run_command('cp', '-lr', res_path, nntr_app_resdir / 'Embedding')

e = executable('nntrainer_embedding',
  'main.cpp',
  dependencies: [iniparser_dep, nntrainer_ccapi_dep, gtest_dep, nntrainer_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_embedding', e, args: ['train',
  nntr_app_resdir / 'Embedding' / 'Embedding.ini',
  nntr_app_resdir / 'Embedding' / 'embedding_input.txt']
)

res_path = meson.current_source_dir() / '..' / 'res'
run_command('cp', '-lr', res_path, nntr_app_resdir / 'ProductRatings')

e = executable('nntrainer_product_ratings',
  'main.cpp',
  dependencies: [iniparser_dep, nntrainer_ccapi_dep, gtest_dep, nntrainer_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)

# test split example
test('nntrainer_product_ratings', e, args: ['train',
  nntr_app_resdir / 'ProductRatings' / 'product_ratings_model.ini',
  nntr_app_resdir / 'ProductRatings' / 'sample_product_ratings.txt']
)

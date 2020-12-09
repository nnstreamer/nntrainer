build_root = meson.build_root()
res_path = meson.current_source_dir() / '..' / 'res'
mnist_sources = [
  'main.cpp'
]
if build_platform == 'tizen'
  if not gtest_dep.found()
    error('Gtest dependency not found for VGG application')
  endif
endif

executable('nntrainer_vgg',
  mnist_sources,
  dependencies: [iniparser_dep, nntrainer_dep, app_utils_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

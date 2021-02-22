res_path = meson.current_source_dir() / '..' / 'res'
nntr_mnist_resdir = nntr_app_resdir / 'MNIST'

run_command('cp', '-lr', res_path, nntr_mnist_resdir)

mnist_sources = [
  'main.cpp'
]

if not gtest_dep.found()
  error('Gtest dependency not found for MNIST application')
endif

e = executable('nntrainer_mnist',
  mnist_sources,
  dependencies: [iniparser_dep, nntrainer_ccapi_dep, gtest_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_mnist', e, args: [
  nntr_mnist_resdir / 'mnist.ini',
  nntr_mnist_resdir / 'mnist_trainingSet.dat']
)


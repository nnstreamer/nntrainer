env_dir='../../Environment'

deepq_sources = [
  'main.cpp',
  'gym_binding.cpp',
  join_paths(env_dir, 'CartPole/cartpole.cpp')
]

deepq_inc = [
  env_dir,
  './'
]

executable('nntrainer_deepq',
  deepq_sources,
  dependencies: [iniparser_dep, nntrainer_dep, tflite_dep, jsoncpp_dep, libcurl_dep],
  include_directories: include_directories(env_dir),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

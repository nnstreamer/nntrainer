res_path = meson.current_source_dir()
env_dir='../../Environment'

deepq_sources = [
  'main.cpp',
  join_paths(env_dir, 'CartPole/cartpole.cpp')
]

deepq_inc = [
  env_dir
]

deepq_deps=[
  iniparser_dep,
  nntrainer_dep,
  tflite_dep,
  jsoncpp_dep,
  libcurl_dep
]

if get_option('use_gym')
   message('[Applications/ReinforcementLearning] Eabling OpenAI/Gym using gym-http-api')
   deepq_deps += dependency('gym-http-api')
   deepq_deps += dependency('boost')
endif

e = executable('nntrainer_deepq',
  deepq_sources,
  dependencies: deepq_deps,
  include_directories: include_directories(env_dir),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_DeepQ', e, args: [join_paths(res_path, 'DeepQ.ini')], timeout: 60)

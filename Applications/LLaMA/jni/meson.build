rms_norm_src = files('rms_norm.cpp')
rms_norm_layer = shared_library('rms_norm',
  rms_norm_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)
rms_norm_dep = declare_dependency(
  link_with: rms_norm_layer,
  include_directories: include_directories('./')
)

swiglu_src = files('swiglu.cpp')
swiglu_layer = shared_library('swiglu',
  swiglu_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)
swiglu_dep = declare_dependency(
  link_with: swiglu_layer,
  include_directories: include_directories('./')
)

llama_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp',
  rms_norm_src,
  swiglu_src
]

llama_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep,
  rms_norm_dep,
  swiglu_dep
]

e = executable('nntrainer_llama',
  llama_sources,
  include_directories: [include_directories('.'), cifar_include_dir],
  dependencies: llama_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

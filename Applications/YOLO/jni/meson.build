layer_reorg_src = files('reorg_layer.cpp')

# build command for lib_reorg_layer.so
reorg_layer = shared_library('reorg_layer',
  layer_reorg_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)

reorg_layer_dep = declare_dependency(
  link_with: reorg_layer,
  include_directories: include_directories('./')
)

yolo_sources = [
  'main.cpp',
  cifar_path / 'cifar_dataloader.cpp',
  layer_reorg_src
]

yolo_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep,
  reorg_layer_dep
]

e = executable('nntrainer_yolov2',
  yolo_sources,
  include_directories: [include_directories('.'), cifar_include_dir],
  dependencies: yolo_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

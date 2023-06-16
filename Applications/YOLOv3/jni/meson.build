upsample_layer_src = files('upsample_layer.cpp')

upsample_layer = shared_library('upsample_layer',
  upsample_layer_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: application_install_dir,
  cpp_args: '-DPLUGGABLE'
)

upsample_layer_dep = declare_dependency(
  link_with: upsample_layer,
  include_directories: include_directories('./')
)

yolo_sources = [
  'main.cpp',
  'det_dataloader.cpp',
  'yolo_v3_loss.cpp',
  upsample_layer_src
]

yolo_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep,
  upsample_layer_dep
]

e = executable('nntrainer_yolov3',
  yolo_sources,
  include_directories: [include_directories('.')],
  dependencies: yolo_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

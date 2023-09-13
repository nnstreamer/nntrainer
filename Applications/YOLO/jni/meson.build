# build command for lib_yolov2_loss_layer.so
yolov2_loss_src = files('yolo_v2_loss.cpp')
yolov2_loss_layer = shared_library('yolov2_loss_layer',
  yolov2_loss_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: nntrainer_libdir/'nntrainer'/'layers',
  cpp_args: '-DPLUGGABLE'
)
yolov2_loss_layer_dep = declare_dependency(
  link_with: yolov2_loss_layer,
  include_directories: include_directories('./')
)

# build command for lib_reorg_layer.so
layer_reorg_src = files('reorg_layer.cpp')
reorg_layer = shared_library('reorg_layer',
  layer_reorg_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: include_directories('./'),
  install: true,
  install_dir: nntrainer_libdir/'nntrainer'/'layers',
  cpp_args: '-DPLUGGABLE'
)
reorg_layer_dep = declare_dependency(
  link_with: reorg_layer,
  include_directories: include_directories('./')
)

yolo_sources = [
  'main.cpp',
  'det_dataloader.cpp',
  'yolo_v2_loss.cpp',
  'reorg_layer.cpp',
]

yolo_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep,
  yolov2_loss_layer_dep,
  reorg_layer_dep
]

e = executable('nntrainer_yolov2',
  yolo_sources,
  include_directories: [include_directories('.')],
  dependencies: yolo_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

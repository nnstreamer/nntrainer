yolo_sources = [
  'main.cpp',
  'det_dataloader.cpp'
]

yolo_dependencies = [app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

e = executable('nntrainer_yolov2',
  yolo_sources,
  include_directories: [include_directories('.')],
  dependencies: yolo_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

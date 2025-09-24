cpp_args = []

if get_option('enable-pluggable')
    cpp_args += [ '-DPLUGGABLE' ]
endif

upsample_layer_src = files('upsample_layer.cpp')

if get_option('default_library') == 'static'
  upsample_layer = static_library('upsample_layer',
    upsample_layer_src,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    include_directories: include_directories('./'),
    install: true,
    install_dir: application_install_dir,
    cpp_args: cpp_args
  )
else
  upsample_layer = shared_library('upsample_layer',
    upsample_layer_src,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    include_directories: include_directories('./'),
    install: true,
    install_dir: application_install_dir,
    cpp_args: cpp_args
  )
endif

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

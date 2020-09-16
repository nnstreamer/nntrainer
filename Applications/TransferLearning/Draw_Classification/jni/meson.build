build_root = meson.build_root()
res_path = join_paths(meson.current_source_dir(), '..', 'res')

gst_api_version = '1.0'
glib_dep = dependency('glib-2.0')
gst_dep = dependency('gstreamer-'+gst_api_version)
nnstreamer_capi_dep = dependency('', required: false)
if build_platform == 'tizen'
  nnstreamer_capi_dep = dependency('capi-nnstreamer')
endif

training_sources = [
  'main.cpp',
  'bitmap_helpers.cpp'
]

e = executable('nntrainer_training',
  training_sources,
  dependencies: [iniparser_dep, nntrainer_capi_dep, nnstreamer_capi_dep, tflite_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_training', e, args: [join_paths(res_path, 'Training.ini'), res_path + '/'], timeout: 60)

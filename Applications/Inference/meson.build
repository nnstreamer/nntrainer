sources = [
  'main.cpp'
]

glib_dep = dependency('glib-2.0')
gst_dep = dependency('gstreamer-1.0')

e = executable('nntrainer_inference',
  sources,
  dependencies: [glib_dep, gst_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test('app_nntrainer_inference', e)

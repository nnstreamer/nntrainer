filter_sub_nntrainer_sources = ['tensor_filter_nntrainer.cc']

nnstreamer_filter_nntrainer_sources = []
foreach s : filter_sub_nntrainer_sources
  nnstreamer_filter_nntrainer_sources += meson.current_source_dir() / s
endforeach

# TODO: remove gstreamer dependency by updating nnstreamer_plugin_api.h
gst_api_version = '1.0'
glib_dep = dependency('glib-2.0')
gmodule_dep = dependency('gmodule-2.0')
gst_dep = dependency('gstreamer-'+gst_api_version)

nntrainer_prefix = get_option('prefix')

nnstreamer_filter_nntrainer_deps = [glib_dep, gmodule_dep, gst_dep, nntrainer_ccapi_dep, nnstreamer_filter_dep]

nnstreamer_libdir = nntrainer_prefix / get_option('libdir')
subplugin_install_prefix = get_option('nnstreamer-subplugin-install-path')
filter_subplugin_install_dir = subplugin_install_prefix / 'filters'

shared_library('nnstreamer_filter_nntrainer',
  nnstreamer_filter_nntrainer_sources,
  dependencies: nnstreamer_filter_nntrainer_deps,
  include_directories: [nntrainer_inc, '.'], # '.' shouldn't be installed
  install: true,
  install_dir: filter_subplugin_install_dir
)

static_library('nnstreamer_filter_nntrainer',
  nnstreamer_filter_nntrainer_sources,
  dependencies: nnstreamer_filter_nntrainer_deps,
  include_directories: [nntrainer_inc, '.'], # '.' shouldn't be installed
  install: true,
  install_dir: nnstreamer_libdir
)

trainer_subplugin_sources = ['tensor_trainer_nntrainer.cc']

extra_defines += '-DSUPPORT_NNSREAMER_PLUGIN_TRAINER=1'

nnstreamer_trainer_nntrainer_sources = []
foreach s : trainer_subplugin_sources
  nnstreamer_trainer_nntrainer_sources += meson.current_source_dir() / s
endforeach

# TODO: remove gstreamer dependency by updating nnstreamer_plugin_api.h
gst_api_version = '1.0'
glib_dep = dependency('glib-2.0')
gmodule_dep = dependency('gmodule-2.0')
gst_dep = dependency('gstreamer-'+gst_api_version)

nntrainer_prefix = get_option('prefix')

nnstreamer_trainer_nntrainer_deps = [glib_dep, gmodule_dep, gst_dep, nntrainer_ccapi_dep, nnstreamer_dep]

nnstreamer_libdir = nntrainer_prefix / get_option('libdir')
subplugin_install_prefix = get_option('nnstreamer-subplugin-install-path')
trainer_subplugin_install_dir = subplugin_install_prefix / 'trainers'

shared_library('nnstreamer_trainer_nntrainer',
  nnstreamer_trainer_nntrainer_sources,
  dependencies: nnstreamer_trainer_nntrainer_deps,
  include_directories: [nntrainer_inc, '.'], # '.' shouldn't be installed
  install: true,
  install_dir: trainer_subplugin_install_dir
)

static_library('nnstreamer_trainer_nntrainer',
  nnstreamer_trainer_nntrainer_sources,
  dependencies: nnstreamer_trainer_nntrainer_deps,
  include_directories: [nntrainer_inc, '.'], # '.' shouldn't be installed
  install: true,
  install_dir: nnstreamer_libdir
)

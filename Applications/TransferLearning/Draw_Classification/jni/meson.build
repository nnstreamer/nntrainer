build_root = meson.build_root()
res_path = meson.current_source_dir() / '..' / 'res'
nntr_draw_resdir = nntr_app_resdir / 'Draw_Classification'
run_command('cp', '-lr', res_path, nntr_draw_resdir)

gst_api_version = '1.0'
glib_dep = dependency('glib-2.0')
gst_dep = dependency('gstreamer-'+gst_api_version)

if not nnstreamer_capi_dep.found()
  error('nnstreamer capi dependency not found for tizen')
endif
if not gtest_dep.found()
  error('Gtest dependency not found for tizen')
endif

training_sources = [
  'main.cpp'
]

e = executable('nntrainer_training',
  training_sources,
  dependencies: [app_utils_dep, iniparser_dep, nntrainer_capi_dep, nnstreamer_capi_dep, tflite_dep, gtest_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

nntrainer_filter_env = environment()
nntrainer_filter_env.set('GTEST_OUTPUT', 'xml:@0@/@1@.xml'.format(meson.build_root(), 'app_draw_classification'))
nntrainer_filter_env.set('NNSTREAMER_FILTERS', meson.build_root() / 'nnstreamer' / 'tensor_filter')

# We need to revisit this later
# if get_option('enable-nnstreamer-tensor-filter').enabled()
#    test('app_draw_classification', e, env: nntrainer_filter_env,
#      args: [nntr_draw_resdir / 'Training.ini', nntr_draw_resdir], timeout: 150)
# endif

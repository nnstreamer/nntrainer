# CausalLM JNI build for Android

# Copy resources for Android
res_path = meson.current_source_dir() / '..' / 'res'
nntr_causallm_resdir = nntr_app_resdir / 'CausalLM'

# Create resource directory and copy resources if they exist
if fs.exists(res_path)
  run_command(['cp', '-lr', res_path, nntr_causallm_resdir], check: true)
endif

causallm_jni_sources = [
  'main.cpp',
  'causallm_jni.cpp',
]

causallm_jni_dependencies = [
  app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

# Add test dependencies if enabled
if get_option('enable-test')
  causallm_jni_dependencies += [gtest_dep]
endif

# Build the JNI executable for Android
e = executable('nntrainer_causallm_jni',
  causallm_jni_sources,
  include_directories: [include_directories('.')],
  dependencies: causallm_jni_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir,
  cpp_args: ['-DANDROID_BUILD']
)
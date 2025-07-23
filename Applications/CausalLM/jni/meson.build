# CausalLM JNI build for Android
# Following reference patch style: ae24db6e9c018a819841f5884defb2c9c1fc3a14

res_path = meson.current_source_dir() / '..' / 'res'
nntr_causallm_resdir = nntr_app_resdir / 'CausalLM'
run_command(['cp', '-lr', res_path, nntr_causallm_resdir], check: true)

causallm_jni_sources = [
  'main.cpp',
]

causallm_jni_dependencies = [
  app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  causallm_jni_dependencies += [gtest_dep]
endif

e = executable('nntrainer_causallm',
  causallm_jni_sources,
  include_directories: [include_directories('.')],
  dependencies: causallm_jni_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
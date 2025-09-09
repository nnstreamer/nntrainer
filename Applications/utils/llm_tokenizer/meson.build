llm_tokenizer_path = meson.current_source_dir()

llm_tokenizer_inc = include_directories('.')

llm_tokenizer_src_abs = [
  meson.current_source_dir() / 'huggingface_tokenizer.cpp',
  meson.current_source_dir() / 'nntr_tokenizer_util.cpp',
]

if (get_option('platform') == 'android')
  llm_tokenizer_lib_path = meson.current_source_dir() / 'libs' / 'libtokenizers_android_c.a'
elif (get_option('platform') == 'windows')
  llm_tokenizer_lib_path = meson.current_source_dir() / 'libs' / 'libtokenizers_c.lib'
else #Ubuntu
  llm_tokenizer_lib_path = meson.current_source_dir() / 'libs' / 'libtokenizers_c.a'
endif

llm_tokenizer = shared_library(
  'llm_tokenizer',
  llm_tokenizer_src_abs,
  include_directories: llm_tokenizer_inc,
  dependencies: [],
  link_args: [llm_tokenizer_lib_path],
)

llm_tokenizer_dep = declare_dependency(
  link_with: llm_tokenizer,
  include_directories: llm_tokenizer_inc
)

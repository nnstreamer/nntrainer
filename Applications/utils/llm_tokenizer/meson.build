llm_tokenizer_path = meson.current_source_dir()

llm_tokenizer_inc = include_directories('.')

llm_tokenizer_src_abs = [
  meson.current_source_dir() / 'huggingface_tokenizer.cpp',
  meson.current_source_dir() / 'nntr_tokenizer_util.cpp',
]

llm_toeknizer_dep = []

if (get_option('platform') == 'android')
  llm_tokenizer_lib_path = meson.current_source_dir() / 'libs' / 'libtokenizers_android_c.a'
elif (get_option('platform') == 'windows') and (build_machine.system() == 'windows')
  ws2_dep = cc.find_library('ws2_32', required: true)
  ntdll_dep = cc.find_library('ntdll', required: true)
  bcrypt_dep = cc.find_library('bcrypt', required: true)
  userenv_dep = cc.find_library('userenv', required: true)
  llm_toeknizer_dep += ws2_dep
  llm_toeknizer_dep += ntdll_dep
  llm_toeknizer_dep += bcrypt_dep
  llm_toeknizer_dep += userenv_dep
  llm_tokenizer_lib_path = meson.current_source_dir() / 'libs' / 'libtokenizers_c.lib'
else #Ubuntu
  llm_tokenizer_lib_path = meson.current_source_dir() / 'libs' / 'libtokenizers_c.a'
endif

llm_tokenizer = shared_library(
  'llm_tokenizer',
  llm_tokenizer_src_abs,
  include_directories: llm_tokenizer_inc,
  dependencies: [llm_toeknizer_dep],
  link_args: [llm_tokenizer_lib_path],
)

llm_tokenizer_dep = declare_dependency(
  link_with: llm_tokenizer,
  include_directories: llm_tokenizer_inc
)

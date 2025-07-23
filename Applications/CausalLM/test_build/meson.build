# CausalLM Android Build Configuration
# This file builds CausalLM as an executable for Android

project('causallm-android', 'cpp',
  version : '1.0.0',
  default_options : [
    'warning_level=1',
    'werror=false',
    'cpp_std=c++14'
  ]
)

# Find nntrainer dependency (if available)
nntrainer_dep = dependency('nntrainer', required: false)

# Source files for the executable
executable_src = [
  'main.cpp'
]

# CausalLM library sources (when available)
causallm_src = [
  # 'causal_lm.cpp',
  # 'huggingface_tokenizer.cpp',
  # 'qwen3_causallm.cpp',
  # 'qwen3_moe_causallm.cpp',
]

# Include directories
causallm_inc = include_directories('.')
causallm_inc += include_directories('layers')

# Check if CausalLM source files exist
fs = import('fs')
actual_causallm_src = []
foreach src_file : causallm_src
  if fs.exists(src_file)
    actual_causallm_src += src_file
  endif
endforeach

# Custom layer sources (if they exist)
layer_sources = []
layer_files = [
  'layers/tie_word_embedding.cpp',
  'layers/swiglu.cpp', 
  'layers/mha_core.cpp',
  'layers/embedding_layer.cpp',
  'layers/reshaped_rms_norm.cpp',
  'layers/qwen_moe_layer.cpp'
]

foreach layer_file : layer_files
  if fs.exists(layer_file)
    layer_sources += layer_file
  endif
endforeach

# Combine all sources
all_sources = executable_src + actual_causallm_src + layer_sources

# Dependencies
deps = []
if nntrainer_dep.found()
  deps += nntrainer_dep
endif

# Tokenizer library (Android-specific path)
tokenizer_lib_path = meson.current_source_dir() / 'lib' / 'android' / 'libtokenizers_c.a'
tokenizer_link_args = []
if fs.exists(tokenizer_lib_path)
  tokenizer_link_args = [tokenizer_lib_path]
else
  message('Android tokenizer library not found: ' + tokenizer_lib_path)
endif

# Android-specific compiler flags
android_cpp_args = [
  '-DANDROID',
  '-fPIC'
]

# Build executable for Android
causallm_android_exe = executable('nntr_causallm_android',
  sources: all_sources,
  include_directories: causallm_inc,
  dependencies: deps,
  link_args: tokenizer_link_args,
  cpp_args: android_cpp_args,
  install: true,
  install_dir: 'bin'
)
# CausalLM Application Build Configuration

# Main executable source
executable_src = [
  'main.cpp'
]

# CausalLM library sources - check if files exist
causallm_src = []
causallm_files = [
  'causal_lm.cpp',
  'huggingface_tokenizer.cpp', 
  'qwen3_causallm.cpp',
  'qwen3_moe_causallm.cpp',
]

# Check if CausalLM source files exist
fs = import('fs')
foreach src_file : causallm_files
  if fs.exists(src_file)
    causallm_src += src_file
  endif
endforeach

causallm_inc = include_directories('.')
causallm_inc += include_directories('layers')

# Custom layer dependencies - check if files exist before adding
causallm_layer_dependencies = []

layer_files = [
  'layers/tie_word_embedding.cpp',
  'layers/swiglu.cpp',
  'layers/mha_core.cpp',
  'layers/embedding_layer.cpp',
  'layers/reshaped_rms_norm.cpp',
  'layers/qwen_moe_layer.cpp'
]

layer_sources = []
foreach layer_file : layer_files
  if fs.exists(layer_file)
    layer_sources += layer_file
  endif
endforeach

# If we have layer sources, create dependencies
if layer_sources.length() > 0
  causallm_layer_dependencies += declare_dependency(
    sources: layer_sources,
    include_directories: causallm_inc
  )
endif

# Tokenizer dependencies
tokenizer_dependencies = []
tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'
if fs.exists(tokenizer_path)
  tokenizer_dependencies += declare_dependency(link_args: [tokenizer_path])
endif

# Build executable
e = executable('nntr_causallm',
    executable_src + causallm_src + layer_sources,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, tokenizer_dependencies],
    install: get_option('install-app'),
    install_dir: application_install_dir
)
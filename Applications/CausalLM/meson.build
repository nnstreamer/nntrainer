# CausalLM Application Build Configuration

causallm_src = [
  'causal_lm.cpp',
  'huggingface_tokenizer.cpp', 
  'qwen3_causallm.cpp',
  'qwen3_moe_causallm.cpp',
]

executable_src = [
  'main.cpp'
]

causallm_inc = include_directories('.')
causallm_inc += include_directories('layers')

# Custom layer dependencies
causallm_tie_word_embedding_dep = declare_dependency(
  sources: ['layers/tie_word_embedding.cpp'],
  include_directories: causallm_inc
)

causallm_swiglu_dep = declare_dependency(
  sources: ['layers/swiglu.cpp'],
  include_directories: causallm_inc
)

causallm_mha_core_dep = declare_dependency(
  sources: ['layers/mha_core.cpp'],
  include_directories: causallm_inc
)

causallm_embedding_layer_dep = declare_dependency(
  sources: ['layers/embedding_layer.cpp'],
  include_directories: causallm_inc
)

causallm_reshaped_rms_norm_dep = declare_dependency(
  sources: ['layers/reshaped_rms_norm.cpp'],
  include_directories: causallm_inc
)

causallm_moe_layer_dep = declare_dependency(
  sources: ['layers/qwen_moe_layer.cpp'],
  include_directories: causallm_inc
)

causallm_layer_dependencies = [
    causallm_tie_word_embedding_dep,
    causallm_swiglu_dep,
    causallm_mha_core_dep,
    causallm_embedding_layer_dep,
    causallm_reshaped_rms_norm_dep,
    causallm_moe_layer_dep,
]

tokenizer_dependencies = []
tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'

causallm = shared_library('causallm',
    causallm_src,
    dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        causallm_layer_dependencies, 
        tokenizer_dependencies,
        openmp_dep
    ],
    include_directories: causallm_inc,
    install: true,
    install_dir: application_install_dir,
    link_args: [tokenizer_path],
    cpp_args: '-DPLUGGABLE'
)

causallm_dep = declare_dependency(
    link_with: [causallm],
    include_directories: causallm_inc
)

e = executable('nntr_causallm',
    executable_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
)
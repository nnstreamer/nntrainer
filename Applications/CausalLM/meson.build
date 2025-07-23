# CausalLM Application Meson Build Configuration
# This file defines how to build CausalLM application and its dependencies

causallm_src = [
    'causal_lm.cpp',
    'llm_util.cpp',
    'huggingface_tokenizer.cpp',
]

causallm_model_src = [
    'qwen3_causallm.cpp',
    'qwen3_moe_causallm.cpp'
]

causallm_layer_src = [
    'layers/embedding_layer.cpp',
    'layers/rms_norm.cpp',
    'layers/swiglu.cpp',
    'layers/mha_core.cpp',
    'layers/tie_word_embedding.cpp',
    'layers/reshaped_rms_norm.cpp',
    'layers/qwen_moe_layer.cpp',
]

causallm_main_src = ['main.cpp']

# Include directories
causallm_inc = include_directories('.')
causallm_layers_inc = include_directories('layers')

# Collect all CausalLM sources
all_causallm_src = causallm_src + causallm_model_src + causallm_layer_src

# Check for tokenizer library
tokenizer_lib_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'
tokenizer_dep = []

if build_machine.system() != 'windows'
    # Try to find tokenizer library
    if fs.exists(tokenizer_lib_path)
        tokenizer_dep = declare_dependency(
            link_args: [tokenizer_lib_path]
        )
    endif
endif

# CausalLM layer dependencies (from the original meson.build structure)
causallm_layer_deps = []

# Build CausalLM as a shared library
causallm_lib = shared_library('causallm',
    all_causallm_src,
    dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        tokenizer_dep,
        openmp_dep
    ],
    include_directories: [causallm_inc, causallm_layers_inc],
    install: get_option('install-app'),
    install_dir: application_install_dir,
    cpp_args: ['-DPLUGGABLE'],
    version: meson.project_version(),
    soversion: nntrainer_so_version
)

# Declare CausalLM dependency for other targets
causallm_dep = declare_dependency(
    link_with: causallm_lib,
    include_directories: [causallm_inc, causallm_layers_inc]
)

# Build CausalLM executable
if get_option('enable-app') or get_option('enable-causallm-app')
    causallm_exe = executable('nntr_causallm',
        causallm_main_src,
        dependencies: [
            nntrainer_dep,
            nntrainer_ccapi_dep,
            causallm_dep,
            tokenizer_dep
        ],
        include_directories: [causallm_inc, causallm_layers_inc],
        install: get_option('install-app'),
        install_dir: application_install_dir,
        cpp_args: ['-DPLUGGABLE']
    )
endif

# Android-specific configuration
if get_option('platform') == 'android'
    # Install additional files needed for Android
    if get_option('install-app')
        # Install headers for JNI if needed
        install_headers([
            'causal_lm.h',
            'factory.h',
            'llm_util.hpp'
        ], 
        subdir: 'nntrainer/causallm'
        )
    endif
endif
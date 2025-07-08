project('qwen3_moe_app', 'cpp',
  version: '1.0.0',
  license: 'Apache-2.0',
  meson_version: '>=0.50.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=1',
    'werror=false',
  ]
)

# Dependencies
nntrainer_dep = dependency('nntrainer', required: true)

# Sources
qwen3_moe_sources = [
  'main.cpp',
  'silu_layer.cpp',
  'rms_norm_layer.cpp', 
  'qwen3_moe_mlp_layer.cpp',
]

# Headers (for installation if needed)
qwen3_moe_headers = [
  'silu_layer.h',
  'rms_norm_layer.h',
  'qwen3_moe_mlp_layer.h',
]

# Include directories
inc_dir = include_directories('.')

# Executable
qwen3_moe_exe = executable('nntrainer_qwen3_moe',
  qwen3_moe_sources,
  dependencies: [nntrainer_dep],
  include_directories: inc_dir,
  install: true,
  install_dir: get_option('bindir')
)

# Custom target for testing (if test data is available)
test_target = custom_target('qwen3_moe_test',
  command: [qwen3_moe_exe, '--help'],
  output: 'test_output',
  build_by_default: false
)
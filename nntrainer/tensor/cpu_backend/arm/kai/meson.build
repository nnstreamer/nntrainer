kai_headers = [
    'kai_common.h'
]

subdir('matmul_clamp_f32_qai8dxp_qsi4cxp')
nntrainer_inc += include_directories('matmul_clamp_f32_qai8dxp_qsi4cxp')
nntrainer_inc_abs += meson.current_source_dir() / 'matmul_clamp_f32_qai8dxp_qsi4cxp'


foreach h : kai_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

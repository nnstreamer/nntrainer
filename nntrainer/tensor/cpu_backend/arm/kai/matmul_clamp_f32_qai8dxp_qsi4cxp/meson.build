matmul_clamp_f32_qai8dxp_qsi4cxp_headers = [
    'kai_lhs_quant_pack_qai8dxp_f32.h',
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h',
    'kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h',
    'kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h'
]

matmul_clamp_f32_qai8dxp_qsi4cxp_sources = [
    'kai_lhs_quant_pack_qai8dxp_f32.c',
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod.c',
    'kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.c',
    'kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.c',
]


foreach s : matmul_clamp_f32_qai8dxp_qsi4cxp_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matmul_clamp_f32_qai8dxp_qsi4cxp_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

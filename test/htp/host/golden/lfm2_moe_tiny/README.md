# Golden dump of the tiny LFM2-MoE fixture through the in-process HTP build

Produced by `test/htp/host/run_inproc_e2e.sh` with `NNTR_INPROC_GOLDEN=update`:
the fixture `test/unittest/models/causallm_reference/lfm2_moe_tiny` (hidden 64,
MoE intermediate 32, 4 experts top-2, layers conv/attention/conv with one dense
layer), stream-quantized with `--fc_dtype Q4_0 --moe_dtype QS4CX_WH` on the
host ISA, run with prompt 16 (`ids[i] = 1 + 7i mod 30`) and 8 greedy tokens,
`moe_engine: htp`, default flags (`NNTR_MOE_HTP_M1_GEMV` unset).

* `manifest.txt`: one line per MoE call in call order,
  `name entry M K inter N_out kind` (`entry` is `moe_layer` or `forward`).
* `moe_<call>_in.f32` (M x K) and `_out.f32` (M x N_out): the bytes the ARM
  side handed the call and got back.
* `logits_<step>.f32`: the driver's per-step logits (step 0 = prefill).

`run_inproc_e2e.sh` requires a byte-identical match (`E2E eval golden ...
bit_identical=1`). Regenerate only deliberately, like `reference_logits.json`,
and say why in the commit.

What a mismatch means. The HTP side runs scalar stand-ins
(`test/htp/host/standin/hvx_scalar.c`): the u8 activation quantizer's
rounding and the int32 -> f32 epilogue order there have no `_det` spec yet,
so this golden is a regression gate on the stand-ins, not the device's
bits (plan 84 section 3.1). The layers before each call run nntrainer's CPU
path; `first_diff=moe_00000_in.f32` (an input file, on the very first call)
means that CPU path differs on this machine (a different SIMD level), not
the HTP path -- an `_out` file with its `_in` identical is the HTP path.

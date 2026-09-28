# Golden dump of the hd64 LFM2-MoE fixture through the in-process HTP build

Produced by `test/htp/host/run_inproc_e2e.sh` with `NNTR_INPROC_GOLDEN=update`,
like `../lfm2_moe_tiny`: the fixture
`test/unittest/models/causallm_reference/lfm2_moe_tiny_hd64` (hidden 128, 2 q
heads, 1 kv head, head_dim 64, max_position_embeddings 32, MoE intermediate 32,
4 experts top-2, layers conv/attention/conv with one dense layer; the same
generator at `--dim 128 --n-heads 2 --n-kv-heads 1 --head-dim 64 --max-pos 32`,
plan 130 section 3.4), stream-quantized with `--fc_dtype Q4_0 --moe_dtype
QS4CX_WH` on the host ISA, run with prompt 16, 8 greedy tokens, `--max-seq 32`,
`moe_engine: htp`, the per-token switch **off** (`NNTR_HTP_FORWARD` unset).

The file layout and what a mismatch means are the tiny golden's
(`../lfm2_moe_tiny/README.md`). This fixture exists because the tiny one's
head_dim 8 cannot run the m=1 attention kinds (QK_NORM / ROPE / ATTN_M1 need
head_dim 64 and a multiple-of-32 max_seq): the harness compares the switch-on
run against the switch-off run of this fixture under an SNR floor and the
token policy, and this golden pins the switch-off run itself.

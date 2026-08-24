# CED — nntrainer Inference Port

Port of [CED](https://huggingface.co/mispeech/ced-tiny) (Consistent Ensemble
Distillation), an AudioSet tagging model, driven entirely from the upstream
HuggingFace `config.json`.

## Architecture

CED is a ViT over mel-spectrogram patches, so the encoder trunk is the one
`TimmViTTransformer` already builds. `CedTransformer` only supplies what
differs:

| | timm ViT (image) | CED (audio) |
|---|---|---|
| input | 3 × 224 × 224 | **1 × n_mels × target_length** (1 × 64 × 1012) |
| input norm | — | **`init_bn`**: BatchNorm over the frequency axis |
| pos embed | one `[1, N, D]` table | **separable** `time_pos_embed` + `freq_pos_embed` |
| head | none (feature output) | **mean-pool → LayerNorm → Linear → sigmoid** |
| trunk | 12 pre-norm blocks | identical (config only: D=192, heads=3, mlp=768) |

Two details are easy to get wrong:

- **LayerNorm epsilons differ.** The encoder is built with
  `partial(nn.LayerNorm, eps=1e-6)`, but the head's `outputlayer.0` is a plain
  `nn.LayerNorm`, so it uses PyTorch's `1e-5` default.
- **The positional embeddings are pre-summed offline.** `modeling_ced.py` adds
  `time_pos_embed [1,D,1,T]` and `freq_pos_embed [1,D,F,1]` in NCHW *before*
  flattening. Both are constants, so `weight_converter.py` folds their
  broadcast sum into the single `[1,1,F*T,D]` table the graph adds after
  flattening. `flatten(2,3)` is frequency-major, which is also how a
  `[D,F,T]` NCHW tensor reshapes to `[D,F*T]`, so the orderings agree.

## Input boundary

The model input is `input_values`, the mel-dB spectrogram — the same tensor
HuggingFace's `CedModel.forward()` takes. The STFT / mel / dB front-end lives
outside the model in both implementations, so the runner reads a raw FP32
`[n_mels, frames]` blob.

## Weight conversion

```bash
python weight_converter.py \
  --input  /path/to/ced-tiny \
  --output /path/to/ced-tiny/nntr_ced_tiny_fp32.bin
```

The blob order matches the graph's layer creation order; see the header comment
in `weight_converter.py`.

## Reference extraction and verification

```bash
# PyTorch reference: input, intermediates, and final 527-class logits
python extract_reference.py --out-dir /path/to/ced-tiny/nntr_ref

# optionally from a real 16 kHz wav instead of a deterministic signal
python extract_reference.py --out-dir <dir> --wav sample.wav
```

Then run with `CED_REF_BIN` pointing at the reference logits:

```bash
CED_REF_BIN=/path/to/ced-tiny/nntr_ref/ref_logits.bin \
  ./builddir/Applications/CausalLM/nntr_causallm /path/to/ced-tiny
```

`CED_OUT_BIN=<path>` additionally dumps the raw 527-float output.

## Verified results (x86, FP32)

`ced-tiny`, all 527 classes compared against HuggingFace
`CedForAudioClassification`:

| input | max_abs_diff | rms_diff | NaN | top-1 |
|---|---|---|---|---|
| white noise (seed 1234) | 2.01e-05 | 9.86e-07 | 0 | `White noise` 0.4694 (ref 0.4694) |
| chirp 200→3500 Hz + clicks | 1.76e-05 | 1.01e-06 | 0 | `Sine wave` 0.5435 (ref 0.5435) |

Top-5 labels and their order match the reference exactly for both inputs.

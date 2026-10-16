from pathlib import Path
D=Path(__file__).resolve().parent
rel='Applications/CausalLM/layers/conv_block_layer.cpp'
s=(D/'before'/rel).read_text()
s=s.replace('unsigned int rows, unsigned int channels) {\n#if defined(__aarch64__)\n  if (rows < 16 || channels != 2048)', '''unsigned int rows, unsigned int channels,
                                   unsigned int first = 0,
                                   unsigned int last = 0) {
#if defined(__aarch64__)
  if (last == 0)
    last = rows;
  if (rows < 16 || channels != 2048 || first >= last || last > rows)''',1)
s=s.replace('overlaps(projection, 3u * bytes) ||','overlaps(projection, static_cast<size_t>(last - first) *\n                             3u * channels * sizeof(float)) ||',1)
s=s.replace('    0, static_cast<size_t>(rows), [&](size_t r) {','    static_cast<size_t>(first), static_cast<size_t>(last), [&](size_t r) {',1)
s=s.replace('const float *gate = projection + r * 3u * channels + channels;', 'const float *gate =\n        projection + (r - first) * 3u * channels + channels;',1)
s=s.replace('  (void)channels;\n  return false;', '  (void)channels;\n  (void)first;\n  (void)last;\n  return false;',1)
needle='ConvBlockLayer::ConvBlockLayer() :\n'
helper='''#if defined(__aarch64__)
/** Consume each existing FC row chunk before the next engine call can reuse
 *  its ION view. Global gated rows retain the two-row convolution history.
 *  The 1920+128 requests match invokeFc at K=2048; no staging capacity grows. */
static bool lfmConvChunkedPrefill(
  nntrainer::ComputeOps *ops, void *in_weight, float *in_scale, float *input,
  float *projection, float *gated, const float *weights, float *output,
  float *state, unsigned int rows, unsigned int channels) {
  if (rows != 2048 || channels != 2048 || !ops || !in_weight || !in_scale ||
      !input || !projection || !gated || !weights || !output || !state ||
      !ops->supports_gemm_q4_0_batch_norm_fp32())
    return false;
  const size_t bytes = static_cast<size_t>(rows) * channels * sizeof(float);
  const size_t weight_bytes = 3u * channels * sizeof(float);
  const size_t state_bytes = 2u * channels * sizeof(float);
  const auto disjoint = [](const void *a, size_t a_bytes, const void *b,
                          size_t b_bytes) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a);
    const uintptr_t y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x >= a_bytes : x - y >= b_bytes;
  };
  // Interleaving CPU writes with projection calls is safe only when they
  // cannot change a later chunk's input or any still-live projection/history.
  struct Range {
    const void *data;
    size_t bytes;
  };
  const Range writes[] = {{projection, 3u * bytes}, {gated, bytes},
                          {output, bytes}, {state, state_bytes}};
  for (size_t i = 0; i < 4; ++i) {
    if (!disjoint(writes[i].data, writes[i].bytes, input, bytes) ||
        !disjoint(writes[i].data, writes[i].bytes, weights, weight_bytes))
      return false;
    for (size_t j = 0; j < i; ++j)
      if (!disjoint(writes[i].data, writes[i].bytes, writes[j].data,
                    writes[j].bytes))
        return false;
  }
  constexpr unsigned int step = 1920;
  for (unsigned int first = 0; first < rows; first += step) {
    const unsigned int count = std::min(step, rows - first);
    const unsigned int last = first + count;
    const size_t chunk_bytes =
      static_cast<size_t>(count) * 3u * channels * sizeof(float);
    float *host = projection + static_cast<size_t>(first) * 3u * channels;
    ops->gemm_q4_0_batch_norm_fp32(
      {in_weight}, {in_scale}, input + static_cast<size_t>(first) * channels,
      {host}, count, {3u * channels}, channels, nullptr, {0u}, nullptr, 0.0f,
      nullptr, 0u, 0u, false);
    const auto *view =
      static_cast<const float *>(ops->host_view(host, chunk_bytes));
    if (!view || !disjoint(view, chunk_bytes, input, bytes) ||
        !disjoint(view, chunk_bytes, gated, bytes) ||
        !disjoint(view, chunk_bytes, output, bytes) ||
        !disjoint(view, chunk_bytes, state, state_bytes) ||
        !disjoint(view, chunk_bytes, weights, weight_bytes)) {
      ops->sync_to_host(host, chunk_bytes);
      view = host;
    }
    nntrainer::ThreadManager::Global().parallel_for(
      static_cast<size_t>(first), static_cast<size_t>(last), [&](size_t r) {
        const float *a = view + (r - first) * 3u * channels;
        const float *c = a + 2u * channels;
        float *dst = gated + r * channels;
        for (unsigned int j = 0; j < channels; ++j)
          dst[j] = a[j] * c[j];
      });
    // All geometry/overlap guards were checked before the first RPC and
    // for this view. Keep failure explicit instead of reading future rows.
    if (!lfmConvPostGatePrefill(gated, weights, view, output, rows, channels,
                               first, last))
      throw std::runtime_error("conv_block: invalid retained chunk view");
  }
  std::memcpy(state, gated + static_cast<size_t>(rows - 2u) * channels,
              channels * sizeof(float));
  std::memcpy(state + channels,
              gated + static_cast<size_t>(rows - 1u) * channels,
              channels * sizeof(float));
  return true;
}
#endif

'''
assert s.count(needle)==1;s=s.replace(needle,helper+needle)
needle='  // Match invokeFc\'s K=2048 row cap: larger calls would grow its output\n'
early='''#if defined(__aarch64__)
  // Keep both original RPC shapes at 2048 rows, consuming their retained
  // views separately. Other prompts and unsupported/overlapping tensors
  // continue through the existing projection and convolution paths below.
  if (lfm_prefill && rows == 2048 && C == 2048 &&
      in_step.getDataType() == fp32 && in_step.getContiguous() &&
      !(htp_owns_state_ && from > 0) &&
      lfmConvChunkedPrefill(ops, in_w.getData<char>(), in_w.getScale<float>(),
                           in_step.getData<float>(), proj.getData<float>(),
                           g, w_ptr, y, state, rows, C)) {
    conv_out.dot(out_w, out_step, false, false);
    return;
  }
#endif
'''
assert s.count(needle)==1;s=s.replace(needle,early+needle)
(D/'candidate'/rel).write_text(s)

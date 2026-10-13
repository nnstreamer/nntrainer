// SPDX-License-Identifier: Apache-2.0
// The generated .inc files contain unchanged production control flow. Legacy
// numerical kernels are observable callbacks, not numerical approximations.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#define ENABLE_HEXKL 1
#include "htp_rope_prefill.h"
#include "htp_rope_prefill_shape.h"

static unsigned assertions;
#define CHECK(x)                                                               \
  do {                                                                         \
    ++assertions;                                                              \
    if (!(x))                                                                  \
      throw std::runtime_error(std::string("check ") + #x + " at " +           \
                               std::to_string(__LINE__));                      \
  } while (0)

template <typename F> static void mustThrow(F fn) {
  bool caught = false;
  try {
    fn();
  } catch (const std::runtime_error &) {
    caught = true;
  }
  CHECK(caught);
}

using uint32 = uint32_t;
using uint16 = uint16_t;
using remote_handle64 = uintptr_t;
enum { AEE_SUCCESS, AEE_EUNSUPPORTED, AEE_EINVALIDFORMAT, AEE_ENOMEMORY };
enum { HEXKL_KV_Q8, HEXKL_KV_Q4 };
struct hexkl_kv_q {
  unsigned kind = HEXKL_KV_Q8, fixed = 0, n_head_kv = 8, head_dim = 64;
  unsigned max_rows = 2048;
};
struct nntr_hvx_session {
  bool hmx_locked = true;
  int kv_q = 0;
};
static hexkl_kv_q kv;
static nntr_hvx_session session;
static bool kv_present = true;
static const hexkl_kv_q *hexkl_kv_q_get(int *, uint32 handle) {
  return kv_present && handle == 7u ? &kv : nullptr;
}
struct Observe {
  int prep_rc = 0, attn_rc = 0, outer_rc = 0;
  int force_phase = -1, force_prepare = -1, force_attention = -1;
  bool skip_dispatch = false;
  unsigned prep = 0, attn = 0, rpc = 0, fp16 = 0, released = 0;
  unsigned staged = 0, cached_append = 0;
  const float *q = nullptr, *k = nullptr, *v = nullptr;
  const uint16_t *ko = nullptr, *vo = nullptr;
} obs;
static constexpr size_t NQ = 1024u * 2048u, NK = 1024u * 512u;
static constexpr size_t NC = 1024u * 128u;

static int nntr_hvx_rope_qkv_prefill_f32(
  remote_handle64, uint32 m, uint32 qw, uint32 kw, uint32 hd, const float *cs,
  int cslen, const float *q, int qlen, const float *k, int klen, const float *v,
  int vlen, float *qo, int qolen, uint16 *ko, int kolen, uint16 *vo, int volen,
  uint32 *elapsed, int elapsed_len) {
  ++obs.prep;
  CHECK(obs.attn == 0 && m == 1024 && qw == 2048 && kw == 512 && hd == 64);
  CHECK(cslen == int(NC) && qlen == int(NQ) && qolen == qlen && q == qo);
  CHECK(klen == int(NK) && vlen == klen && kolen == klen && volen == klen);
  CHECK(elapsed_len == 1 && cs && elapsed);
  obs.q = q;
  obs.k = k;
  obs.v = v;
  obs.ko = ko;
  obs.vo = vo;
  if (obs.prep_rc)
    return obs.prep_rc;
  qo[0] += 2.0f;
  qo[NQ - 1] += 3.0f;
  ko[0] = 0x3412;
  ko[NK - 1] = 0x5678;
  vo[0] = 0x9abc;
  vo[NK - 1] = 0xdef0;
  *elapsed = 41;
  return 0;
}

static int nntr_hvx_attn_q8_npu_step(
  remote_handle64, uint32 handle, uint32 row0, const uint16 *k, int klen,
  const uint16 *v, int vlen, uint32 nq, uint32 from, uint32 to, uint32 heads,
  uint32 window, float softcap, const float *q, int qlen, const float *sinks,
  int sinkslen, float *out, int outlen, uint32 *stats, int statslen) {
  ++obs.attn;
  CHECK(obs.prep == 1 && handle == 7 && row0 == 0 && from == 0 && to == 1024);
  CHECK(k == obs.ko && v == obs.vo && q == obs.q);
  CHECK(klen == int(NK) && vlen == klen && qlen == int(NQ) && outlen == qlen);
  CHECK(nq == 1024 && heads == 32 && window == 0 && softcap == 0);
  CHECK(!sinks && sinkslen == 0 && statslen == 20);
  CHECK(q[0] == 3.0f && q[NQ - 1] == 4.0f);
  CHECK(k[0] == 0x3412 && k[NK - 1] == 0x5678);
  CHECK(v[0] == 0x9abc && v[NK - 1] == 0xdef0);
  ++obs.cached_append; // A failure may follow a partial resident-cache append.
  out[0] = 8.0f;
  out[NQ - 1] = 9.0f;
  for (int i = 0; i < 20; ++i)
    stats[i] = unsigned(100 + i);
  return obs.attn_rc;
}

#include "dsp.inc"

static int nntr_hvx_attn_q8_prefill_prepare(
  remote_handle64 handle, uint32 kh, uint32 m, const float *cs, int cslen,
  float *q, int qlen, const float *k, int klen, const float *v, int vlen,
  uint16 *ko, int kolen, uint16 *vo, int volen, float *out, int outlen,
  uint32 *status, int statuslen) {
  ++obs.rpc;
  if (obs.skip_dispatch)
    return obs.outer_rc;
  const int rc =
    fixture_dsp_composite(handle, kh, m, cs, cslen, q, qlen, k, klen, v, vlen,
                          ko, kolen, vo, volen, out, outlen, status, statuslen);
  if (obs.force_phase >= 0)
    status[0] = unsigned(obs.force_phase);
  if (obs.force_prepare >= 0)
    status[1] = unsigned(obs.force_prepare);
  if (obs.force_attention >= 0)
    status[2] = unsigned(obs.force_attention);
  return obs.outer_rc ? obs.outer_rc : rc;
}

template <typename... Args> static void fixtureLog(const char *, Args...) {}
#define ml_logi(...) fixtureLog(__VA_ARGS__)
#define ml_logw(...) fixtureLog(__VA_ARGS__)
namespace nntrainer {
struct HtpBackend {
  bool active = true;
  std::vector<std::pair<const void *, size_t>> ranges;
  static HtpBackend &global() {
    static HtpBackend backend;
    return backend;
  }
  bool enabled() const { return active; }
  remote_handle64 handle() const {
    return reinterpret_cast<uintptr_t>(&session);
  }
  bool isSharedRange(const void *p, size_t bytes) const {
    for (const auto &range : ranges)
      if (p == range.first && bytes <= range.second)
        return true;
    return false;
  }
};
struct Buffer {
  std::vector<uint8_t> bytes;
  void *data() { return bytes.data(); }
};
class HtpComputeOps {
public:
  bool supported = true, supported_q2 = false;
  std::mutex invoke_mutex_, attn_kind_mutex_;
  std::map<int, unsigned> attn_cache_kinds_{{7, 0}};
  Buffer rope_prefill_pool_[6];
  uint64_t prefill_q8_composite_calls_ = 0;
  Buffer &stage(Buffer &buffer, size_t bytes) {
    buffer.bytes.resize(bytes);
    return buffer;
  }
  void stagedMemcpy(void *dst, const void *src, size_t bytes) {
    ++obs.staged;
    std::memcpy(dst, src, bytes);
  }
  bool attn_trace_enabled() const { return false; }
  int64_t now_us() const { return 0; }
  bool supports_kv_cache_q() const { return supported; }
  bool supports_kv_cache_q2() const { return supported_q2; }
  void kv_cache_q_release(int h) {
    CHECK(h >= 0);
    ++obs.released;
  }
#include "arm.inc"
};
static HtpComputeOps ops;
static HtpComputeOps *get_htp_ops() { return &ops; }
#include "bridge.inc"
} // namespace nntrainer

namespace ml::train {
struct TensorDim {
  enum class DataType { FP32, FP16, UINT16, UINT8 };
  unsigned b = 1, c = 1, h = 1024, w = 512;
  unsigned batch() const { return b; }
  unsigned channel() const { return c; }
  unsigned height() const { return h; }
  unsigned width() const { return w; }
  void height(unsigned n) { h = n; }
  size_t getFeatureLen() const { return size_t(c) * h * w; }
};
} // namespace ml::train
namespace nntrainer {
struct Tensor {
  void *p;
  ml::train::TensorDim d;
  ml::train::TensorDim::DataType type;
  unsigned batch() const { return d.b; }
  unsigned channel() const { return d.c; }
  unsigned height() const { return d.h; }
  unsigned width() const { return d.w; }
  auto getDataType() const { return type; }
  template <typename T> T *getData() const { return static_cast<T *>(p); }
  Tensor getSharedDataTensor(ml::train::TensorDim dim, size_t off, bool) const {
    return {static_cast<uint8_t *>(p) +
              off * (type == ml::train::TensorDim::DataType::FP32 ? 4 : 2),
            dim, type};
  }
  void copyData(const Tensor &) {
    throw std::runtime_error("unexpected FP16 Q");
  }
};
namespace props {
struct PrefillRopeFused {
  bool value = false;
  bool get() const { return value; }
};
struct MaxTimestep {
  unsigned value = 2048;
  unsigned get() const { return value; }
};
} // namespace props
} // namespace nntrainer
namespace props = nntrainer::props;
static bool strict_enabled = true;
static bool htpStrict() { return strict_enabled; }
static nntrainer::HtpComputeOps *htpStrictOps() { return &nntrainer::ops; }
struct Thrower {
  template <typename T> void operator<<(const T &) {
    throw std::runtime_error("unexpected branch");
  }
};
#define NNTR_THROW_IF(condition, type)                                         \
  if (condition)                                                               \
  Thrower()

class MHACoreLayer {
public:
  bool use_rope = true, skip_prefill = false, is_causal = true;
  unsigned cache_index = 0, head_dim = 64, num_heads_Q = 32, num_heads_KV = 8;
  std::tuple<props::PrefillRopeFused, props::MaxTimestep> mha_core_props;
  int kv_cache_quant_kind = 0;
  bool q_cache_failed = false, accel_logged_ = false;
  nntrainer::HtpComputeOps *compute_ops_ = &nntrainer::ops;
  float attn_logit_softcapping = 0.0f, theta = 10000.0f;
  unsigned local_window_size = 0, max_position_embeddings = 2048;
  std::vector<int> q_cache_handles{7};
  std::vector<unsigned> q_cache_synced{0}, q2_scales_set{0};
  bool q2_calibrated = false;
  struct Frequencies {
    std::vector<std::vector<float>> cos, sin;
  };
  std::unique_ptr<Frequencies> freqs_fp32;
  std::mutex rope_init_mtx;
  void precompute_freqs(unsigned hd, unsigned maximum, float, bool) {
    freqs_fp32 = std::make_unique<Frequencies>();
    freqs_fp32->cos.assign(maximum, std::vector<float>(hd, 1.0f));
    freqs_fp32->sin.assign(maximum, std::vector<float>(hd, 0.0f));
  }
  void release_quantized_cache();
  bool try_prefill_q8_composite(unsigned, unsigned, unsigned,
                                nntrainer::Tensor &, nntrainer::Tensor &,
                                nntrainer::Tensor &, nntrainer::Tensor &,
                                nntrainer::Tensor &, nntrainer::Tensor &,
                                const ml::train::TensorDim &, bool &);
  void one_batch_incremental_forwarding(
    unsigned, unsigned, unsigned, unsigned, nntrainer::Tensor &,
    nntrainer::Tensor &, nntrainer::Tensor &, nntrainer::Tensor &,
    nntrainer::Tensor &, nntrainer::Tensor &, ml::train::TensorDim &,
    ml::train::TensorDim &, ml::train::TensorDim &, ml::train::TensorDim &);
  bool try_fused_rope_qkv(nntrainer::Tensor &q, nntrainer::Tensor &k,
                          nntrainer::Tensor &v, nntrainer::Tensor &kc,
                          nntrainer::Tensor &vc, unsigned) {
    std::vector<float> phase(NC);
    uint32 elapsed;
    CHECK(nntr_hvx_rope_qkv_prefill_f32(
            0, 1024, 2048, 512, 64, phase.data(), NC, q.getData<float>(), NQ,
            k.getData<float>(), NK, v.getData<float>(), NK, q.getData<float>(),
            NQ, kc.getData<uint16>(), NK, vc.getData<uint16>(), NK, &elapsed,
            1) == 0);
    return true;
  }
  void apply_rotary_emb_tensor_v2(nntrainer::Tensor &, nntrainer::Tensor &,
                                  unsigned, unsigned, bool) {
    throw std::runtime_error("preparation replay");
  }
  bool try_quantized_attention(unsigned, nntrainer::Tensor &q,
                               nntrainer::Tensor &kc, nntrainer::Tensor &vc,
                               const ml::train::TensorDim &,
                               nntrainer::Tensor &out, unsigned, unsigned,
                               const nntrainer::Tensor *) {
    if (q_cache_failed)
      return false;
    uint32 stats[20];
    CHECK(nntr_hvx_attn_q8_npu_step(0, 7, 0, kc.getData<uint16>(), NK,
                                    vc.getData<uint16>(), NK, 1024, 0, 1024, 32,
                                    0, 0.0f, q.getData<float>(), NQ, nullptr, 0,
                                    out.getData<float>(), NQ, stats, 20) == 0);
    q_cache_synced[0] = 1024;
    return true;
  }
  bool try_accelerated_attention(nntrainer::Tensor &q, nntrainer::Tensor &kc,
                                 nntrainer::Tensor &vc, nntrainer::Tensor &out,
                                 unsigned, unsigned,
                                 const nntrainer::Tensor *) {
    CHECK(q_cache_failed && q_cache_handles.empty() && q_cache_synced.empty());
    CHECK(obs.prep == 1 && obs.attn == 1 && obs.released >= 1);
    CHECK(q.getData<float>()[0] == 3.0f && kc.getData<uint16>()[0] == 0x3412);
    CHECK(vc.getData<uint16>()[0] == 0x9abc);
    ++obs.fp16;
    out.getData<float>()[0] = 11.0f;
    return true;
  }
};
#include "mha.inc"

struct Fixture {
  std::vector<float> cs = std::vector<float>(NC, 0.5f);
  std::vector<float> q = std::vector<float>(NQ, 1.0f);
  std::vector<float> k = std::vector<float>(NK, 2.0f);
  std::vector<float> v = std::vector<float>(NK, 3.0f);
  std::vector<float> out = std::vector<float>(NQ, -1.0f);
  std::vector<uint16_t> ko = std::vector<uint16_t>(NK, 0xeeee);
  std::vector<uint16_t> vo = std::vector<uint16_t>(NK, 0xffff);
  uint32 status[24];
  void reset() {
    obs = {};
    kv = {};
    session = {};
    kv_present = true;
    strict_enabled = true;
    nntrainer::ops.supported = true;
    nntrainer::ops.supported_q2 = false;
    nntrainer::ops.attn_cache_kinds_ = {{7, 0}};
    auto &backend = nntrainer::HtpBackend::global();
    backend.active = true;
    backend.ranges = {{q.data(), NQ * 4},  {k.data(), NK * 4},
                      {v.data(), NK * 4},  {ko.data(), NK * 2},
                      {vo.data(), NK * 2}, {out.data(), NQ * 4}};
    std::fill(q.begin(), q.end(), 1.0f);
    std::fill(out.begin(), out.end(), -1.0f);
    std::fill(ko.begin(), ko.end(), 0xeeee);
    std::fill(vo.begin(), vo.end(), 0xffff);
    std::fill(std::begin(status), std::end(status), 0xdddddddd);
    for (const char *name :
         {"NNTR_HTP_PREFILL_ATTN_COMPOSITE", "NNTR_HTP_PREFILL_QKV_PREPARE",
          "NNTR_HTP_ATTN_Q8_WIRE", "NNTR_HTP_ATTN_Q8_NPU"})
      CHECK(setenv(name, "1", 1) == 0);
  }
  int dsp(uint32 *record = nullptr, int record_len = 24, int qlen = NQ) {
    return fixture_dsp_composite(
      reinterpret_cast<uintptr_t>(&session), 7, 1024, cs.data(), NC, q.data(),
      qlen, k.data(), NK, v.data(), NK, ko.data(), NK, vo.data(), NK,
      out.data(), NQ, record ? record : status, record_len);
  }
  auto arm(unsigned rows = 1024) {
    return nntrainer::htp_rope_q8_attention_prefill(
      7, rows, cs.data(), q.data(), k.data(), v.data(), ko.data(), vo.data(),
      out.data());
  }
  void untouched() {
    CHECK(obs.prep == 0 && obs.attn == 0 && obs.cached_append == 0);
    CHECK(q.front() == 1 && q.back() == 1 && out.front() == -1);
    CHECK(ko.front() == 0xeeee && vo.front() == 0xffff);
  }
  struct Graph {
    using DT = ml::train::TensorDim::DataType;
    ml::train::TensorDim cd{1, 1, 1024, 512}, qd{1, 1, 1024, 2048};
    nntrainer::Tensor q, k, v, ko, vo, out;
    explicit Graph(Fixture &f) :
      q{f.q.data(), qd, DT::FP32},
      k{f.k.data(), cd, DT::FP32},
      v{f.v.data(), cd, DT::FP32},
      ko{f.ko.data(), cd, DT::FP16},
      vo{f.vo.data(), cd, DT::FP16},
      out{f.out.data(), qd, DT::FP32} {}
    bool helper(MHACoreLayer &m, bool &prepared, unsigned from = 0,
                unsigned to = 1024, unsigned batch = 0) {
      return m.try_prefill_q8_composite(batch, from, to, q, k, v, ko, vo, out,
                                        cd, prepared);
    }
    void caller(MHACoreLayer &m) {
      m.one_batch_incremental_forwarding(0, 0, 0, 1024, q, k, v, out, ko, vo,
                                         cd, cd, cd, cd);
    }
  };
};

int main() {
  using Result = nntrainer::HtpPrefillAttentionResult;
  Fixture f;
  f.reset();
  CHECK(f.dsp() == 0 && f.status[0] == 2 && f.status[1] == 0 &&
        f.status[2] == 0);
  CHECK(f.status[3] == 41);
  for (unsigned i = 0; i < 20; ++i)
    CHECK(f.status[4 + i] == 100 + i);
  const auto q_reference = f.q, out_reference = f.out;
  const auto ko_reference = f.ko, vo_reference = f.vo;
  f.reset();
  CHECK(nntr_hvx_rope_qkv_prefill_f32(
          0, 1024, 2048, 512, 64, f.cs.data(), NC, f.q.data(), NQ, f.k.data(),
          NK, f.v.data(), NK, f.q.data(), NQ, f.ko.data(), NK, f.vo.data(), NK,
          f.status, 1) == 0);
  CHECK(nntr_hvx_attn_q8_npu_step(
          0, 7, 0, f.ko.data(), NK, f.vo.data(), NK, 1024, 0, 1024, 32, 0, 0,
          f.q.data(), NQ, nullptr, 0, f.out.data(), NQ, f.status, 20) == 0);
  CHECK(f.q == q_reference && f.out == out_reference && f.ko == ko_reference &&
        f.vo == vo_reference);
  CHECK(std::all_of(f.k.begin(), f.k.end(), [](float x) { return x == 2.0f; }));
  CHECK(std::all_of(f.v.begin(), f.v.end(), [](float x) { return x == 3.0f; }));
  for (int rc : {AEE_EUNSUPPORTED, AEE_ENOMEMORY}) {
    f.reset();
    obs.prep_rc = rc;
    CHECK(f.dsp() == 0 && f.status[0] == 0 && f.status[1] == unsigned(rc));
    CHECK(obs.prep == 1 && obs.attn == 0 && f.q[0] == 1);
    f.reset();
    obs.attn_rc = rc;
    CHECK(f.dsp() == 0 && f.status[0] == 1 && f.status[2] == unsigned(rc));
    CHECK(obs.prep == 1 && obs.attn == 1 && obs.cached_append == 1);
  }
  for (unsigned i = 0; i < 7; ++i) {
    f.reset();
    void *addresses[] = {f.q.data(),  f.k.data(),  f.v.data(),  f.cs.data(),
                         f.ko.data(), f.vo.data(), f.out.data()};
    CHECK(f.dsp(static_cast<uint32 *>(addresses[i])) == AEE_EINVALIDFORMAT);
    f.untouched();
  }
  for (unsigned i = 0; i < 6; ++i) {
    f.reset();
    void *addresses[] = {f.q.data(),  f.k.data(),  f.v.data(),
                         f.cs.data(), f.ko.data(), f.vo.data()};
    auto *alias = static_cast<float *>(addresses[i]);
    CHECK(nntrainer::htp_rope_q8_attention_prefill(
            7, 1024, f.cs.data(), f.q.data(), f.k.data(), f.v.data(),
            f.ko.data(), f.vo.data(), alias) == Result::Unsupported);
    CHECK(obs.rpc == 0);
    f.untouched();
    CHECK(fixture_dsp_composite(
            reinterpret_cast<uintptr_t>(&session), 7, 1024, f.cs.data(), NC,
            f.q.data(), NQ, f.k.data(), NK, f.v.data(), NK, f.ko.data(), NK,
            f.vo.data(), NK, alias, NQ, f.status, 24) == AEE_EINVALIDFORMAT);
    f.untouched();
  }
  f.reset();
  CHECK(fixture_dsp_composite(reinterpret_cast<uintptr_t>(&session), 7, 1024,
                              f.cs.data(), NC, f.q.data(), NQ, f.k.data(), NK,
                              f.v.data(), NK, f.ko.data(), NK, f.vo.data(), NK,
                              f.out.data(), NQ, nullptr,
                              24) == AEE_EINVALIDFORMAT);
  f.untouched();
  f.reset();
  CHECK(f.dsp(reinterpret_cast<uint32 *>(f.q.data() + NQ - 1)) ==
        AEE_EINVALIDFORMAT);
  f.untouched();
  f.reset();
  CHECK(f.dsp(nullptr, 23) == AEE_EINVALIDFORMAT);
  f.untouched();
  f.reset();
  CHECK(f.dsp(nullptr, 24, NQ - 1) == AEE_EINVALIDFORMAT);
  f.untouched();
  for (unsigned which = 0; which < 7; ++which) {
    f.reset();
    if (which == 0)
      session.hmx_locked = false;
    if (which == 1)
      kv_present = false;
    if (which == 2)
      kv.kind = HEXKL_KV_Q4;
    if (which == 3)
      kv.fixed = 1;
    if (which == 4)
      kv.n_head_kv = 4;
    if (which == 5)
      kv.head_dim = 128;
    if (which == 6)
      kv.max_rows = 1023;
    CHECK(f.arm() == Result::Unsupported);
    f.untouched();
  }
  f.reset();
  CHECK(f.arm() == Result::Complete && obs.staged == 1);
  f.reset();
  nntrainer::HtpBackend::global().ranges.push_back({f.cs.data(), NC * 4});
  CHECK(f.arm() == Result::Complete && obs.staged == 0);
  for (const char *name :
       {"NNTR_HTP_PREFILL_ATTN_COMPOSITE", "NNTR_HTP_PREFILL_QKV_PREPARE",
        "NNTR_HTP_ATTN_Q8_WIRE", "NNTR_HTP_ATTN_Q8_NPU"}) {
    f.reset();
    CHECK(setenv(name, "0", 1) == 0);
    CHECK(f.arm() == Result::Unsupported && obs.rpc == 0);
    f.untouched();
  }
  for (unsigned which = 0; which < 10; ++which) {
    f.reset();
    if (which < 6)
      nntrainer::HtpBackend::global().ranges.erase(
        nntrainer::HtpBackend::global().ranges.begin() + which);
    if (which == 6)
      nntrainer::HtpBackend::global().active = false;
    if (which == 7)
      nntrainer::ops.attn_cache_kinds_.clear();
    if (which == 8)
      nntrainer::ops.attn_cache_kinds_[7] = 1;
    CHECK(f.arm(which == 9 ? 1023 : 1024) == Result::Unsupported &&
          obs.rpc == 0);
    f.untouched();
  }
  for (int rc : {AEE_EUNSUPPORTED, AEE_ENOMEMORY}) {
    f.reset();
    obs.prep_rc = rc;
    mustThrow([&] { f.arm(); });
    CHECK(obs.prep == 1 && obs.attn == 0);
    f.reset();
    obs.attn_rc = rc;
    CHECK(f.arm() == Result::Prepared);
    CHECK(obs.prep == 1 && obs.attn == 1);
    f.reset();
    obs.outer_rc = rc;
    mustThrow([&] { f.arm(); });
    CHECK(obs.prep == 1 && obs.attn == 1 && f.q[0] == 3);
    f.reset();
    obs.outer_rc = rc;
    obs.skip_dispatch = true;
    mustThrow([&] { f.arm(); });
    f.untouched();
  }
  f.reset();
  obs.skip_dispatch = true;
  mustThrow([&] { f.arm(); });
  for (int phase : {0, 1, 3, 4, 99}) {
    f.reset();
    obs.force_phase = phase;
    mustThrow([&] { f.arm(); });
  }
  f.reset();
  obs.force_phase = 3;
  obs.force_prepare = 9;
  mustThrow([&] { f.arm(); });

  f.reset();
  {
    Fixture::Graph g(f);
    MHACoreLayer m;
    m.q_cache_synced[0] = 1800;
    g.caller(m);
    CHECK(obs.rpc == 1 && obs.prep == 1 && obs.attn == 1 && obs.fp16 == 0);
    CHECK(m.q_cache_synced[0] == 1024 && !m.q_cache_failed && m.accel_logged_);
  }
  f.reset();
  {
    Fixture::Graph g(f);
    MHACoreLayer m;
    session.hmx_locked = false;
    g.caller(m);
    CHECK(obs.rpc == 1 && obs.prep == 1 && obs.attn == 1 && obs.fp16 == 0);
    CHECK(!m.q_cache_failed && m.q_cache_handles.size() == 1);
  }
  for (int rc : {AEE_EUNSUPPORTED, AEE_ENOMEMORY}) {
    f.reset();
    Fixture::Graph g(f);
    MHACoreLayer m;
    m.q_cache_handles = {7, 8, -1};
    m.q_cache_synced = {0, 0, 0};
    obs.attn_rc = rc;
    g.caller(m);
    CHECK(obs.prep == 1 && obs.attn == 1 && obs.fp16 == 1 && obs.released == 2);
    CHECK(m.q_cache_failed && m.q_cache_handles.empty() && f.q[0] == 3);
    CHECK(f.out[0] == 11 && f.ko[0] == 0x3412 && f.vo[0] == 0x9abc);
  }
  for (int mode = 0; mode < 5; ++mode) {
    f.reset();
    Fixture::Graph g(f);
    MHACoreLayer m;
    if (mode == 0)
      obs.prep_rc = AEE_ENOMEMORY;
    if (mode == 1)
      obs.outer_rc = AEE_ENOMEMORY;
    if (mode == 2)
      obs.outer_rc = AEE_EUNSUPPORTED;
    if (mode == 3)
      obs.force_phase = 99;
    if (mode == 4)
      obs.skip_dispatch = true;
    mustThrow([&] { g.caller(m); });
    CHECK(obs.prep <= 1 && obs.attn <= 1 && obs.fp16 == 0);
  }
  for (unsigned which = 0; which < 30; ++which) {
    f.reset();
    Fixture::Graph g(f);
    MHACoreLayer m;
    unsigned from = 0, to = 1024, batch = 0;
    switch (which) {
    case 0:
      m.use_rope = false;
      break;
    case 1:
      m.skip_prefill = true;
      break;
    case 2:
      m.is_causal = false;
      break;
    case 3:
      m.cache_index = 1;
      break;
    case 4:
      m.head_dim = 128;
      break;
    case 5:
      m.num_heads_Q = 16;
      break;
    case 6:
      m.num_heads_KV = 4;
      break;
    case 7:
      std::get<props::PrefillRopeFused>(m.mha_core_props).value = true;
      break;
    case 8:
      m.kv_cache_quant_kind = 1;
      break;
    case 9:
      m.q_cache_failed = true;
      break;
    case 10:
      m.compute_ops_ = nullptr;
      break;
    case 11:
      nntrainer::ops.supported = false;
      break;
    case 12:
      nntrainer::ops.supported_q2 = true;
      break;
    case 13:
      m.attn_logit_softcapping = 1;
      break;
    case 14:
      m.local_window_size = 512;
      break;
    case 15:
      m.q_cache_handles.clear();
      break;
    case 16:
      m.q_cache_synced.clear();
      break;
    case 17:
      m.q_cache_handles[0] = -1;
      break;
    case 18:
      g.cd.h = 1023;
      break;
    case 19:
      g.cd.w = 1024;
      break;
    case 20:
      g.q.d.h = 1023;
      break;
    case 21:
      g.q.type = Fixture::Graph::DT::FP16;
      break;
    case 22:
      g.ko.type = Fixture::Graph::DT::FP32;
      break;
    case 23:
      g.out.d.c = 2;
      break;
    case 24:
      m.max_position_embeddings = 1023;
      break;
    case 25:
      strict_enabled = false;
      break;
    case 26:
      from = 1;
      break;
    case 27:
      to = 1023;
      break;
    case 28:
      batch = 1;
      break;
    case 29:
      CHECK(setenv("NNTR_HTP_PREFILL_ATTN_COMPOSITE", "0", 1) == 0);
      break;
    }
    bool prepared = true;
    CHECK(!g.helper(m, prepared, from, to, batch) && !prepared && obs.rpc == 0);
    f.untouched();
  }
  std::printf("PASS: attention composite actual control flow (%u assertions)\n",
              assertions);
}
#else
int main() {
  std::puts("SKIP: environment-injection fixture requires Linux/Android");
}
#endif

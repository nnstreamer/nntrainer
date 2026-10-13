#!/usr/bin/env python3
"""mkcfg.py <name> <doc.txt> "<question>" [max_seq_len] [C] [more.txt]

Writes three demo configs for one document (push with `demo.sh cfg`):
  demo_<name>       the document as system_prompt.head_prompt with a KV cache file:
                    the first run saves the cache and exits, later runs load it
  demo_<name>_full  document + question as one prompt (no cache: the whole prefill)
  demo_<name>_q     the question alone (no document)
max_seq_len: default 2304 (a 2k document); 23552 for a 22-23k one (64-multiple,
  >= document tokens + question + 60).  C: NNTR_MOE_CACHE_EXPERTS of the cached and
  the question-only run, default 32, 24 at 23552 (32 does not map there).
  A run that prefills the whole 22-23k document (the save, _full) uses C=8: its
  2048-row chunks do not fit the DSP heap beside a larger pool at that length.
"""
import json, os, sys
name, doc, q = sys.argv[1], open(sys.argv[2]).read().strip('\n'), sys.argv[3]
mx = int(sys.argv[4]) if len(sys.argv) > 4 else 2304
C = sys.argv[5] if len(sys.argv) > 5 else ('32' if mx <= 4608 else '24')
# more.txt: text prefilled after the cached document, before the question (cache + N more tokens)
more = open(sys.argv[6]).read().strip('\n') if len(sys.argv) > 6 else None
W = os.path.dirname(os.path.abspath(__file__)); D = '/data/local/tmp/nntrainer/causallm'
MD = D + '/model/gemma4-26b-a4b'
base = json.load(open(W + '/base_nntr_config.json'))
base.update(model_file_name=MD + '/nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin', tokenizer_file=MD + '/tokenizer.json',
            num_to_generate=60, max_seq_len=mx, init_seq_len=2048)
head = '<bos><|turn>user\nRead the provided <Text> carefully.\n\n<Text>\n' + doc + '\n</Text>\n'
tail = '<turn|>\n<|turn>model\n'
if more:  # the document continues past the cached part: <Text> closes after it
    head = head[:-len('</Text>\n')] + '\n'
    q = more + '\n</Text>\n' + q
kv = dict(base, sample_input=q, system_prompt={
    'head_prompt': head, 'tail_prompt': tail,
    'kvcache': {'pre_computed_cache_path': '%s/demo_%s/kv.bin' % (D, name)}})
full = dict(base, sample_input=head + q + tail)
qonly = dict(base, sample_input='<bos><|turn>user\n' + q + tail)
for n, c in ((name, kv), (name + '_full', full), (name + '_q', qonly)):
    d = W + '/demo_' + n; os.makedirs(d, exist_ok=True)
    json.dump(c, open(d + '/nntr_config.json', 'w'), indent=2, ensure_ascii=False)
    Cp = C if mx <= 4608 else '8'  # the whole-document prefill's pool
    open(d + '/env', 'w').write('NNTR_MOE_CACHE_EXPERTS=' + (Cp if n.endswith('_full') else C) + '\n')
    open(d + '/env_save', 'w').write('NNTR_MOE_CACHE_EXPERTS=' + Cp + '\n')
print('demo_%s{,_full,_q}: %d chars of document, max_seq_len %d, C=%s' % (name, len(doc), mx, C))

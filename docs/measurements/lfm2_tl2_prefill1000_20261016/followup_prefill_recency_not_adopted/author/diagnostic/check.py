#!/usr/bin/env python3
"""Actual marker formatter/guard extraction plus strict parser negatives.

Metadata and counters are stand-ins here. Assignment and LRU numeric tests are
separate; this is not a model, DSP, routing or performance test.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

D = Path(__file__).resolve().parent
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('parse', D/'parse.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = (D/'candidate/Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp').read_text()
a = source.index('#if defined(NNTR_LFM2_RECENCY_TRACE)')
b = source.index('#endif', a) + len('#endif\n')
marker = source[a:b]
(out/'marker.inc').write_text(marker)
code = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <tuple>
namespace nntrainer { namespace props { struct Unit {unsigned get() const {return 1536;}};}}
struct Input {unsigned batch() const {return 1;} unsigned channel() const {return 1;}};
struct Context {unsigned getNumInputs() const {return 1;}};
struct Hint {size_t n; size_t size() const {return n;}};
static void emit(unsigned trace_layer, unsigned total_tokens, size_t count, bool compact_recency) {
  Input input_; Context context; Hint extra_top_k{count};
  unsigned from=0,hidden_size=2048,num_experts=64,topk=4;
  constexpr unsigned EXTRA_TOPK=5;
  bool experts_virtual=true,in_norm=true,router_norm=false,out_norm=false;
  bool out_add=false,softmax_router=false,gelu_act=false;
  std::tuple<nntrainer::props::Unit> moe_props;
#include "marker.inc"
}
int main(int argc,char**argv) {
  if(argc==1) {emit(UINT32_MAX,2048,SIZE_MAX,true); return 0;}
  const unsigned rows=std::strtoul(argv[1],nullptr,10), fast=std::strtoul(argv[2],nullptr,10);
  for(unsigned layer=0;layer<38;++layer) emit(layer,rows,fast?64-layer%6:rows*9,fast);
  emit(0,1,9,false); emit(0,64,9,false); emit(0,512,9,false);
}
'''
(out/'marker.cpp').write_text(code)
cmd=['g++','-std=c++17','-O2','-Wall','-Wextra','-DNNTR_LFM2_RECENCY_TRACE=1',str(out/'marker.cpp'),'-o',str(out/'marker')]
subprocess.run(cmd,check=True)
positives=[]
for rows in (1024,2048):
    for enabled in (0,1):
        p=subprocess.run([str(out/'marker'),str(rows),str(enabled)],capture_output=True,text=True,check=True)
        result=module.parse(p.stderr,rows,enabled)
        (out/f'{rows}-{enabled}.log').write_text(p.stderr)
        positives.append({k:v for k,v in result.items() if k!='records'})
p=subprocess.run([str(out/'marker')],capture_output=True,text=True,check=True)
assert len(p.stderr.strip())<128
good=(out/'1024-1.log').read_text()
lines=good.splitlines()
bad=[('empty',''),('missing','\n'.join(lines[:-1])),('duplicate',good+lines[0]),
     ('reorder','\n'.join([lines[1],lines[0]]+lines[2:])),
     ('wrong_rows',good.replace('M=1024','M=2048',1)),
     ('wrong_raw',good.replace('raw=9216','raw=9215',1)),
     ('wrong_dispatch',good.replace('fast=1','fast=0',1)),
     ('bad_extent',good.replace('kept=64','kept=65',1)),
     ('bad_min_extent',good.replace('kept=64','kept=8',1)),
     ('trailing',good.replace(' !',' ! garbage',1)),
     ('unknown_field',good.replace('kept=64','other=64',1))]
for cut in range(len(lines[0])):
    bad.append((f'truncated_{cut}',lines[0][:cut]+'\n'+'\n'.join(lines[1:])))
for name,text in bad:
    try: module.parse(text,1024,1)
    except ValueError: pass
    else: raise AssertionError('accepted '+name)
result=dict(actual_formatter_positive_cohorts=positives,invalid_rejected=len(bad),
            max_actual_format_bytes=len(p.stderr.rstrip('\n')),compile=cmd,
            scope='Actual marker text with metadata/counter stand-ins. No numerical or timing claim.')
(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('compile','actual_formatter_positive_cohorts')}))

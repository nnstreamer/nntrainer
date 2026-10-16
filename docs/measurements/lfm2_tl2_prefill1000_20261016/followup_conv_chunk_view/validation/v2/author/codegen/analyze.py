#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--before',type=Path,required=True)
p.add_argument('--v1',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
name='causallm::lfmConvPostGatePrefill(float const*, float const*, float const*, float*, unsigned int, unsigned int)::$_0::operator()(unsigned long) const'
appname='causallm::ConvBlockLayer::incremental_forwarding(nntrainer::RunLayerContext&, unsigned int, unsigned int, bool)'
def function(s,name):
 start=s.index('0000000000000000 <'+name+'>:')
 end=s.find('\nDisassembly of section ',start)
 return s[start:end if end>=0 else None]
def instructions(s):
 return [(m.group(1),m.group(2)) for m in re.finditer(r'^\s+[0-9a-f]+:\s+([0-9a-f]{8})\s+(.*)',s,re.M)]
out={};cb=[]
for variant,path in [('before',a.before),('v1',a.v1),('v2',a.candidate)]:
 s=path.read_text();app=instructions(function(s,appname))
 out[variant]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'app_instruction_count':len(app),'app_code_bytes':4*len(app),'prologue':app[:9]}
 if variant!='v1':
  f=function(s,name);seq=instructions(f);cb.append([x for x,_ in seq])
  out[variant]['callback_instruction_count']=len(seq)
  out[variant]['callback_code_bytes']=4*len(seq)
  out[variant]['callback_instructions_sha256']=hashlib.sha256(''.join(x for x,_ in seq).encode()).hexdigest()
  helper=name.split('::$_0')[0]
  out[variant]['full_helper_outlined']=bool(re.search(r'^0+ <'+re.escape(helper)+r'>:',s,re.M))
out['full_callback_raw_instruction_bytes_equal']=cb[0]==cb[1]
out['app_prologue_first9_instructions_equal']=out['before']['prologue']==out['v2']['prologue']
assert out['full_callback_raw_instruction_bytes_equal']
assert out['app_prologue_first9_instructions_equal']
assert not out['v2']['full_helper_outlined']
out['limits']='Existing full-row callback and common prologue are exact. Whole app is not byte-identical and these are not cycle/TPS measurements.'
a.output.write_text(json.dumps(out,indent=2)+'\n')
print('CODEGEN full_callback/prologue exact; app',out['before']['app_code_bytes'],'->',out['v1']['app_code_bytes'],'->',out['v2']['app_code_bytes'],'bytes PASS')

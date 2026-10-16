#!/usr/bin/env python3
"""Author mutation checks for the isolated conv chunk candidate."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--simde-include',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();D=Path(__file__).resolve().parent
rel='Applications/CausalLM/layers/conv_block_layer.cpp'
s=(D/'candidate'/rel).read_text()
mutants={
 'chunk_local_history':('const float *prev1 = r >= 1 ? cur - channels : nullptr;','const float *prev1 = r >= first + 1 ? cur - channels : nullptr;'),
 'wrong_projection_row':('projection + (r - first) * 3u * channels + channels','projection + r * 3u * channels + channels'),
 'missing_input_overlap':('!disjoint(writes[i].data, writes[i].bytes, input, bytes) ||','false ||'),
 'read_sampled_host':('const auto *view =\n      static_cast<const float *>(ops->host_view(host, chunk_bytes));','const auto *view = host;'),
 'state_before_success':('  constexpr unsigned int step = 1920;','  std::memcpy(state, gated, state_bytes);\n  constexpr unsigned int step = 1920;'),
 'missing_from_guard':('      !(htp_owns_state_ && from > 0) &&',''),
}
a.output.mkdir(parents=True,exist_ok=True);results={}
for name,(old,new) in mutants.items():
 if name in ('chunk_local_history','wrong_projection_row'):
  start=s.index('static bool lfmConvPostGatePrefillRange(')
  end=s.index('#if defined(__clang__)\n#pragma float_control(pop)',start)
  part=s[start:end];assert part.count(old)==1,(name,part.count(old))
  mutated=s[:start]+part.replace(old,new)+s[end:]
 else:
  assert s.count(old)==1,(name,s.count(old));mutated=s.replace(old,new)
 path=a.output/name/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(mutated)
 out=a.output/name/'checks';log=a.output/name/'run.log'
 cmd=['python3',str(D/'test/htp/host/run_lfm2_conv_chunk_check.py'),'--repo',str(a.repo),'--source',str(a.output/name),'--baseline',str(D/'before'),'--simde-include',str(a.simde_include),'--output',str(out),'--sanitize']
 with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 data=log.read_text()
 reason=[l for l in data.splitlines() if 'CONV_CHUNK FAIL:' in l or 'ERROR: AddressSanitizer' in l]
 killed=r.returncode!=0 and bool(reason)
 results[name]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'command':cmd,'returncode':r.returncode,'killed':killed,'reason':reason[:1]}
 print(name,results[name]['reason'],flush=True)
 if not killed:raise RuntimeError('mutant not caught or infrastructure failure: '+name)
(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')

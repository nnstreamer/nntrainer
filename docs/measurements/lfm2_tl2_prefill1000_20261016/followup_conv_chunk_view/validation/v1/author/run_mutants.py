#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
R=Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
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
results={}
for name,(old,new) in mutants.items():
 assert s.count(old)==1,(name,s.count(old))
 path=D/'mutants'/name/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s.replace(old,new))
 out=D/'mutants'/name/'checks';log=path.parents[3]/'run.log'
 cmd=['python3',str(D/'test/htp/host/run_lfm2_conv_chunk_check.py'),'--repo',str(R),'--source',str(D/'mutants'/name),'--baseline',str(D/'before'),'--simde-include',str(D/'deps/simde-0.8.2'),'--output',str(out),'--sanitize']
 with (D/'mutants'/name/'run.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 data=(D/'mutants'/name/'run.log').read_text()
 killed=r.returncode!=0 and ('CONV_CHUNK FAIL:' in data or 'ERROR: AddressSanitizer' in data)
 results[name]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'command':cmd,'returncode':r.returncode,'killed':killed,'reason':[l for l in data.splitlines() if 'CONV_CHUNK FAIL:' in l or 'ERROR: AddressSanitizer' in l][:1]}
 print(name,results[name]['reason'],flush=True)
 if not killed:raise RuntimeError('mutant not caught or infrastructure failure: '+name)
(D/'mutants/results.json').write_text(json.dumps(results,indent=2)+'\n')

import json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
base=json.loads((D/'android-command.json').read_text())
results={}
for name in ['before','candidate']:
 out=D/'codegen'/name;out.mkdir(parents=True,exist_ok=True)
 cmd=list(base['argv']);src=str(D/name/'Applications/CausalLM/layers/conv_block_layer.cpp')
 for i,x in enumerate(cmd):
  if x.endswith('/Applications/CausalLM/layers/conv_block_layer.cpp'):cmd[i]=src
 cmd[cmd.index('-o')+1]=str(out/'conv.o');cmd.insert(1,'-fstack-usage')
 with (out/'build.log').open('w')as f:r=subprocess.run(cmd,cwd=base['cwd'],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0
 su=(out/'conv.su').read_text().splitlines()
 relevant=[l for l in su if 'incremental_forwarding' in l or 'lfmConv' in l]
 results[name]={'compile':cmd,'cwd':base['cwd'],'source':src,'stack_lines':relevant}
 dis=[str(Path(cmd[0]).with_name('llvm-objdump')),'-d','--demangle',str(out/'conv.o')]
 with(out/'conv.disasm.txt').open('w')as f:subprocess.run(dis,stdout=f,check=True)
(D/'codegen/results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:v['stack_lines'] for k,v in results.items()},indent=2))

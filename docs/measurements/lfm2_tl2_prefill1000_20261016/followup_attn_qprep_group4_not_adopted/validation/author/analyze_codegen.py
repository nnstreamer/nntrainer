#!/usr/bin/env python3
"""Read-only ELF32 function bytes and v81 IO layout/stack/import comparison."""
import hashlib,json,struct,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent

def symbols(path):
 data=path.read_bytes();h=struct.unpack_from('<16sHHIIIIIHHHHHH',data)
 assert h[0][:6]==b'\x7fELF\x01\x01'
 sections=[struct.unpack_from('<IIIIIIIIII',data,h[6]+i*h[11]) for i in range(h[12])];out={}
 for table in sections:
  if table[1]!=2:continue
  string=sections[table[6]];names=data[string[4]:string[4]+string[5]]
  for i in range(table[5]//table[9]):
   name,value,size,info,other,index=struct.unpack_from('<IIIBBH',data,table[4]+i*table[9])
   if not index or index>=len(sections) or not size:continue
   section=sections[index];n=names[name:].split(b'\0',1)[0].decode()
   out[n]={'type':info&15,'bytes':data[section[4]+value:section[4]+value+size]}
 return out

plans=json.loads((D/'codegen/commands.json').read_text());binpath=Path(plans[0]['command'][0]).parent
report={}
for stem in ['hvx_quant_u8','hexkl_attn_q','hvx_attn_decode_q','nntr_hvx_attn_q']:
 a,b=[symbols(D/'codegen'/side/(stem+'.o')) for side in ['baseline','candidate']]
 common=sorted(n for n in a.keys()&b.keys() if a[n]['type']==2)
 eq=[n for n in common if a[n]['bytes']==b[n]['bytes']]
 imports={}
 for side in ['baseline','candidate']:
  p=D/'codegen'/side/(stem+'.o')
  text=subprocess.check_output([str(binpath/'llvm-nm'),'--undefined-only',str(p)],text=True)
  (p.with_suffix('.imports.txt')).write_text(text);imports[side]=text.splitlines()
 report[stem]={'same_function_bytes':eq,'changed_function_bytes':sorted(set(common)-set(eq)),
               'new_functions':sorted(n for n in b.keys()-a.keys() if b[n]['type']==2),
               'byte_exact_object':(D/'codegen/baseline'/(stem+'.o')).read_bytes()==(D/'codegen/candidate'/(stem+'.o')).read_bytes(),
               'new_import_lines':sorted(set(imports['candidate'])-set(imports['baseline']))}
assert len(report['hvx_quant_u8']['same_function_bytes'])==11
assert not report['hvx_quant_u8']['changed_function_bytes']
assert report['hvx_attn_decode_q']['byte_exact_object']
layout={}
for side in ['baseline','candidate']:
 rec=next(r for r in plans if r['side']==side and r['source'].endswith('/hvx_quant_u8.c') and r['mode']=='object')
 dst=D/'codegen'/side/'io-layout.c'
 dst.write_text('#include <stddef.h>\n#include "hexkl_attn_q.h"\nconst unsigned qprep_io_layout[]={sizeof(hexkl_attn_q_io),offsetof(hexkl_attn_q_io,q),offsetof(hexkl_attn_q_io,q_stride),offsetof(hexkl_attn_q_io,sinks),offsetof(hexkl_attn_q_io,out),offsetof(hexkl_attn_q_io,out_stride),offsetof(hexkl_attn_q_io,kv)};\n')
 cmd=rec['command'][:]
 for opt in ['-o','-dependency-file','-MT','-stack-usage-file']:
  while opt in cmd:
   at=cmd.index(opt);del cmd[at:at+2]
 cmd=[str(dst) if x==str(D/side/rec['source']) else x for x in cmd];obj=dst.with_suffix('.o');cmd+=['-o',str(obj)]
 result=subprocess.run(cmd,cwd=rec['cwd'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 dst.with_suffix('.log').write_text(result.stdout);assert result.returncode==0,result.stdout
 dst.with_suffix('.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 layout[side]=list(struct.unpack('<7I',symbols(obj)['qprep_io_layout']['bytes']))
assert layout['baseline']==[24,0,4,8,12,16,20]
assert layout['candidate']==[28,0,4,8,12,16,20]
report['v81_io_size_and_original_offsets']=layout
report['limits']='Function bytes are relocatable object bytes, not a model timing result or final linked-placement proof. Same source math and allocations are audited separately. Full root A/B uses 59 frozen common objects.'
(D/'codegen/summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

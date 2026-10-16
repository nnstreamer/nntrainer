from pathlib import Path
import hashlib,json,struct,subprocess
D=Path('/tmp/nntr-lfm2-ring-fixed-objects-20261017');O=Path('/tmp/nntr-lfm2-ring-fixed-review-20261017');v=json.loads((D/'build/validation.json').read_text());results=[]
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def sections(p):
 data=p.read_bytes();assert data[:6]==b'\x7fELF\x01\x01';e=struct.unpack_from('<16sHHIIIIIHHHHHH',data);shoff=e[6];size,count,strings=e[11:14];raw=[struct.unpack_from('<IIIIIIIIII',data,shoff+i*size) for i in range(count)];names=data[raw[strings][4]:raw[strings][4]+raw[strings][5]];out={}
 for s in raw:
  name=names[s[0]:].split(b'\0',1)[0].decode();out[name]={'type':s[1],'flags':s[2],'addr':s[3],'size':s[5],'data':data[s[4]:s[4]+s[5]] if s[1]!=8 else b''}
 return out
for label in ('control','candidate'):
 p=O/label;p.mkdir(exist_ok=True);a=v[label]['link_command'].copy();a[a.index('-o')+1]=str(p/'libnntr_hvx_skel.so');(p/'command.json').write_text(json.dumps({'argv':a,'cwd':str(D/'build/source/test/htp')},indent=2)+'\n');hashes=[]
 for i in range(2):
  with (p/f'build{i+1}.log').open('w') as f:subprocess.run(a,cwd=D/'build/source/test/htp',stdout=f,stderr=subprocess.STDOUT,check=True)
  hashes.append(h(p/'libnntr_hvx_skel.so'))
 assert hashes[0]==hashes[1]
 before=sections(D/'build'/label/'libnntr_hvx_skel.so');after=sections(p/'libnntr_hvx_skel.so');assert before.keys()==after.keys();diff=[n for n in before if before[n]!=after[n]];assert diff==['.comment'],diff
 results.append({'label':label,'whole_sha256':hashes[0],'repeated_same_path_link_exact':True,'all_alloc_sections_equal_frozen':True,'section_content_differences':['.comment'],'note':'Output path differs from frozen command and is embedded in non-allocated .comment. ELF section offsets/header therefore also differ; this is not same-path linker nondeterminism.'})
(O/'relink-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

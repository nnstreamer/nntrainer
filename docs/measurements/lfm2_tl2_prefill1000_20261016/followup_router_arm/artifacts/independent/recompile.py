from pathlib import Path
import concurrent.futures,hashlib,json,subprocess
D=Path('/tmp/nntr-lfm2-router-arm-20261017')
O=Path('/tmp/nntr-lfm2-router-arm-review-20261017/codegen')
O.mkdir(parents=True,exist_ok=True)
items=[('kernel-arm',D/'codegen/final/kernel-command.json'),('app-arm',D/'codegen/app-command.json'),('app-android-x86',D/'codegen/final/android-x86-clang/command.json'),('kernel-android-x86',D/'codegen/final/android-x86-clang/kernel-command.json'),('app-linux-x86',D/'codegen/final/linux-x86-gcc/command.json'),('kernel-linux-x86',D/'codegen/final/linux-x86-gcc/kernel-command.json')]
def run(item):
 name,p=item;data=json.loads(p.read_text());argv=data['argv'] if isinstance(data,dict) else data
 argv=argv.copy();out=O/name;out.mkdir(exist_ok=True);argv[argv.index('-o')+1]=str(out/'output.o')
 if '-fstack-usage' not in argv:argv+=['-fstack-usage']
 cwd=data.get('cwd') if isinstance(data,dict) else None
 (out/'command.json').write_text(json.dumps({'argv':argv,'cwd':cwd},indent=2)+'\n')
 with (out/'build.log').open('w') as f:r=subprocess.run(argv,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 result={'name':name,'returncode':r.returncode,'object_sha256':hashlib.sha256((out/'output.o').read_bytes()).hexdigest() if r.returncode==0 else None}
 print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,items))
(O/'results.json').write_text(json.dumps(results,indent=2)+'\n')
if any(x['returncode'] for x in results):raise SystemExit(1)
ndk=Path('/home/jungone-lee/workspace/nntrainer/android-ndk-r30/toolchains/llvm/prebuilt/linux-x86_64/bin')
for name in ('kernel-arm','app-arm'):
 with (O/name/'output.asm').open('w') as f:subprocess.run([str(ndk/'llvm-objdump'),'-dr','--demangle',str(O/name/'output.o')],stdout=f,check=True)

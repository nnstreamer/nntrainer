from pathlib import Path
import hashlib,json,shutil,subprocess,time
HERE=Path(__file__).resolve().parent
PUB=Path('/tmp/lfm2-prefill1000-20261013/publish')
BUILD=Path('/tmp/lfm2-prefill1000-20261013/publish-build')
PREV=Path('/tmp/lfm2-prefill1000-moe/eager-payload/publication-compat')
DIAG=Path('/tmp/lfm2-prefill1000-20261013/conv-norm/host-staging/publication-compat')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
commands=json.loads((PREV/'matched-install-commands.json').read_text())
(HERE/'matched-install-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
app_before={str(p):sha(p) for p in (BUILD/'app-obj').rglob('*.o')}
exports={}
for relative in ['nntrainer/utils/weight_direct_file.h','nntrainer/tensor/htp_backend/htp_expert_host_staging.h']:
 source=PUB/relative
 for prefix in ['include','include/nntrainer']:
  dest=PUB/'builddir/android_build_result'/prefix/source.name
  shutil.copy2(source,dest);assert sha(source)==sha(dest);exports[str(dest)]=sha(dest)
(HERE/'exported-headers.json').write_text(json.dumps(exports,indent=2)+'\n')
results=json.loads((HERE/'matched-install-results.json').read_text()) if (HERE/'matched-install-results.json').exists() else []
for kind in ['core','app']:
 if any(x['kind']==kind and x['returncode']==0 for x in results):
  continue
 r=commands[kind];log=HERE/(kind+'-matched-install.log');start=time.monotonic()
 with log.open('w') as f:result=subprocess.run(r['command'],cwd=r['cwd'],stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(kind=kind,returncode=result.returncode,seconds=time.monotonic()-start,log=str(log)))
 (HERE/'matched-install-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(kind,'build/install',result.returncode,flush=True);result.check_returncode()
 if kind=='core':
  installed=PUB/'builddir/android_build_result/lib/arm64-v8a'
  for name in ['libnntrainer.so','libccapi-nntrainer.so','libc++_shared.so','libsdkl.so']:
   source=BUILD/'core-libs/arm64-v8a'/name
   if not (installed/name).exists() or not source.samefile(installed/name):
    shutil.copy2(source,installed/name)
for name in ['libnntrainer.so','libccapi-nntrainer.so','libc++_shared.so']:
 assert sha(BUILD/'core-libs/arm64-v8a'/name)==sha(BUILD/'app-libs/arm64-v8a'/name)
app_after={str(p):sha(p) for p in (BUILD/'app-obj').rglob('*.o')}
(HERE/'app-object-changes.json').write_text(json.dumps({'before':app_before,'after':app_after,'changed':[p for p,s in app_after.items() if app_before.get(p)!=s]},indent=2)+'\n')
prev=json.loads((PREV/'artifacts.json').read_text())
artifacts={};dest=HERE/'artifacts';dest.mkdir(exist_ok=False)
for name in prev['artifacts']:
 if name=='libnntr_hvx_skel.so':source=PREV/'artifacts'/name
 elif name in ['libcausallm_core.so','nntrainer_causallm']:source=BUILD/'app-libs/arm64-v8a'/name
 else:source=BUILD/'core-libs/arm64-v8a'/name
 frozen=dest/name;shutil.copy2(source,frozen)
 artifacts[name]=dict(source=str(source),frozen=str(frozen),bytes=frozen.stat().st_size,sha256=sha(frozen))
source_paths=list(json.loads((HERE/'applied-source.json').read_text())['files'])
source_paths+=['Applications/CausalLM/models/causal_lm.cpp','Applications/CausalLM/models/causal_lm.h','Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp','Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.h','Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp','nntrainer/tensor/cpu_backend/compute_ops.h']
report=dict(publication_base='9b2ced98f66f65506385d9bd71759fb58a7f534e',production_only=True,host_source_sha256={p:sha(PUB/p) for p in source_paths},artifacts=artifacts,build_results=results,exported_headers=exports,app_objects_changed=[p for p,s in app_after.items() if app_before.get(p)!=s],dsp=dict(reused_unchanged=True,sha256=artifacts['libnntr_hvx_skel.so']['sha256'],source_commit='c40d68d59cca5bd0987ff33c66db2d8031572c71',registry_limit=2048),device_deployed_by_agent=False,commit_by_agent=False,root_live_changes_by_agent=False)
assert report['dsp']['sha256']=='a403c2595e86494178e03f63a1b895fdacdf55e8ca4a3076777ee44cd8232ef1'
(HERE/'artifacts.json').write_text(json.dumps(report,indent=2)+'\n')
for f in ['build_publication_diagnostic.py','pub-app-commands.json']:
 shutil.copy2(DIAG/f,HERE/f)
with (HERE/'diagnostic.log').open('w') as log:
 subprocess.run(['python3',str(HERE/'build_publication_diagnostic.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
built=json.loads((HERE/'diagnostic-build.json').read_text())
(HERE/'diagnostic-artifacts.json').write_text(json.dumps(built,indent=2)+'\n')
print('production and refreshed diagnostic frozen',flush=True)

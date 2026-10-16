#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
R=Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
A=Path('/tmp/nntr-lfm2-tl2-expand-noalias-20261016')
raw=json.loads((A/'compile-commands.json').read_text())[2]
common=raw[:raw.index('-fstack-usage')]
common=[str(D/'candidate'/Path(x).relative_to(A/'candidate')) if x.startswith(str(A/'candidate')) else x for x in common]
common += ['-I',str(R/'test/htp'),'-fstack-usage']
O=D/'v81';O.mkdir(exist_ok=True)
commands=[];results={};nm=Path(common[0]).with_name('hexagon-nm')
for mode in ['baseline','off','on']:
    root=D/('baseline' if mode=='baseline' else 'candidate')
    flags=[('-DNNTR_LFM2_TL2_EXPAND_TRACE='+('1' if mode=='on' else '0'))]
    for rel in ['nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c','test/htp/nntr_hvx_mm_u8i4.c']:
        name=Path(rel).stem+'-'+mode;out=O/(name+'.o')
        cmd=common+flags+['-c',str(root/rel),'-o',str(out)];commands.append(cmd)
        with (O/(name+'.log')).open('w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        syms=subprocess.check_output([str(nm),'-u',str(out)],text=True)
        (O/(name+'.undefined.txt')).write_text(syms)
        if '__atomic' in syms or '__sync' in syms: raise RuntimeError('unexpected atomic imports: '+syms)
        results[name]={'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'undefined_symbols':syms.splitlines()}
        cmd=common+flags+['-S',str(root/rel),'-o',str(O/(name+'.s'))];commands.append(cmd)
        with (O/(name+'-asm.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
assert results['hvx_gemv_u8tl2-baseline']['sha256']==results['hvx_gemv_u8tl2-off']['sha256']
# FARF embeds __FILE__/__LINE__ in existing error logs. Verify RPC OFF parity
# with just that diagnostic source-location metadata normalized in both TUs.
normalized=[]
for mode in ['baseline','off']:
    root=D/('baseline' if mode=='baseline' else 'candidate')
    out=O/('rpc-metadata-normalized-'+mode+'.o')
    cmd=common+['-DNNTR_LFM2_TL2_EXPAND_TRACE=0','-Wno-builtin-macro-redefined','-D__LINE__=0','-D__FILE__=\"normalized\"','-c',str(root/'test/htp/nntr_hvx_mm_u8i4.c'),'-o',str(out)]
    commands.append(cmd)
    with (O/('rpc-metadata-normalized-'+mode+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    normalized.append(hashlib.sha256(out.read_bytes()).hexdigest())
assert normalized[0]==normalized[1]
results['rpc_off_metadata_normalized_sha256']=normalized[0]
assembly=(O/'hvx_gemv_u8tl2-on.s').read_text()
assert 'memw_locked' in assembly
results['hvx_trace_off_object_identical']=True
results['rpc_trace_off_identical_after_farf_source_metadata_normalization']=True
results['inline_atomic_memw_locked']=True
results['no_atomic_runtime_imports']=True
(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
(O/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('PASS: six v81 production TU builds plus two FARF metadata-normalized builds; OFF computations identical; inline memw_locked; no atomic runtime imports')

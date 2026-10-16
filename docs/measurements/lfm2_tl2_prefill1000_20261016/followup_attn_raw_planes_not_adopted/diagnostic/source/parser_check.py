#!/usr/bin/env python3
import importlib.util, json, subprocess, tempfile
from pathlib import Path
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('parse',D/'parse.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
e=json.loads((D/'device-trace-expected.json').read_text())
def line(i,x):
 return '[LFM-RAW4] q=%u m=%u on=%u raw=%u,%u sdk=%u b=%u rc=%d END'%(i,x['m'],x['on'],x['raw_s'],x['raw_d'],x['sdk'],x['b'],x['rc'])
rows=[line(i+1,x)for i,x in enumerate(e)];bare='\n'.join(rows);prefix='10-17 12:00:00.123 12345 12346 V adsprpc : '
q=m.parse(bare,e,host=True);assert(q['calls'],q['selected'],q['raw_serial'],q['raw_deferred'],q['sdk_qk'])==(33,6,55296,86016,170472)
assert m.parse('\n'.join(prefix+x for x in rows),e,pid=12345)['calls']==33
assert m.parse(bare+'\n'+rows[0],e,host=True)['transport_duplicates']==1
bad=[('',e,None,True),(bare,e,12345,False),('\n'.join(prefix+x for x in rows),e,77,False),
 ('\n'.join(rows[1:]),e,None,True),('\n'.join(rows[:-1]),e,None,True),
 ('\n'.join([rows[1],rows[0],*rows[2:]]),e,None,True),
 (bare+'\n'+rows[0].replace('raw=10240,0','raw=10239,0'),e,None,True),
 (bare.replace('q=1 ','q=4294967295 ',1),e,None,True),
 (bare.replace(' on=1 ',' on=0 ',1),e,None,True),
 (bare.replace(' rc=0 END',' rc=-1 END',1),e,None,True),
 (bare.replace(' sdk=0 ',' sdk=4294967296 ',1),e,None,True),
 (bare.replace(' raw=10240,0 ',' raw=0,10240 ',1),e,None,True),
 (bare+' trailing',e,None,True),
 ('\n'.join(prefix+x for x in rows)+'\n'+'10-17 12:00:00.123 54321 3 V adsprpc : '+rows[0],e,12345,False)]
for cut in range(len('[LFM-RAW'),len(rows[0])):
 bad.append((rows[0][:cut]+'\n'+'\n'.join(rows[1:]),e,None,True))
for text,expected,pid,host in bad:
 try:m.parse(text,expected,pid,host)
 except ValueError:continue
 raise AssertionError('accepted invalid trace: '+text[:200])
# Use the exact production FARF format in a real C snprintf worst-value check.
s=(D/'source/nntrainer/tensor/htp_backend/hmx/hexkl_attn_q.c').read_text()
start=s.index('"[LFM-RAW4]');end=s.index('"',start+1)+1;fmt=s[start:end]
c='#include <stdio.h>\n#include <stdint.h>\n#include <limits.h>\nint main(void){char b[512];int n=snprintf(b,sizeof b,'+fmt+',UINT32_MAX,UINT32_MAX,1u,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,INT32_MIN);if(n<0||n>=224)return 1;printf("%d\\n",n);return 0;}\n'
(D/'format_check.c').write_text(c);subprocess.run(['cc','-Wall','-Wextra','-Werror',str(D/'format_check.c'),'-o',str(D/'format_check')],check=True);length=int(subprocess.check_output([str(D/'format_check')],text=True))
r={'valid':3,'negative_rejected':len(bad),'exact_c_formatter_max_payload':length,'counter_expected':q}
(D/'parser-validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

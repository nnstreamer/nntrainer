#!/usr/bin/env python3
from pathlib import Path
import json
import re
import parse_fc_regression as p

D=Path(__file__).resolve().parent
text=(D/'checks/final-topology/candidate/cohort2048.log').read_text()
lines=[s for s in text.splitlines() if 'LFM-AFC2' in s]
g=next(s for s in lines if '[LFM-AFC2G]' in s)
t=next(s for s in lines if '[LFM-AFC2T]' in s)
b=next(s for s in lines if '[LFM-AFC2B]' in s)
g1920=next(s for s in lines if '[LFM-AFC2G]' in s and 'd=1920,' in s)
base='\n'.join(lines)
assert p.parse(base,2048,1)['main']['calls']==190
assert p.parse('\n'.join(reversed(lines)),2048,1)['records']==290
replay=p.parse(base+'\n'+g+'\n'+t+'\n'+b,2048,1)
assert replay['transport_duplicate_packets']==3 and replay['records']==290
bad={
 'missing-branch':base.replace(b+'\n','',1),
 'missing-timing':base.replace(t+'\n','',1),
 'missing-geometry':base.replace(g+'\n','',1),
 'whole-call-missing':'\n'.join(lines[3:]),
 'branch-unused':base.replace(b,re.sub(r'f=1,\d+,0','f=1,0,0',b),1),
 'branch-not-selected':base.replace(b,b.replace('f=1,','f=0,'),1),
 'wrong-arena':base.replace(b,b.replace('a=8372224','a=8388608'),1),
 'wrong-bank32-on1920':base.replace(g1920,g1920.replace('x=0,32,15,1','x=0,32,32,1'),1),
 'wrong-row-chunk':base.replace(g1920,g1920.replace('d=1920,','d=1792,'),1),
 'error-code':base.replace(g,g.replace('x=0,','x=99,'),1),
 'pipeline-fallback':base.replace(g,g.replace(',32,32,1 ', ',32,32,0 '),1),
 'truncated-branch':base.replace(b,b[:-1],1),
 'branch-overflow':base.replace(b,b.replace('f=1,','f=4294967296,'),1),
 'branch-negative':base.replace(b,b.replace('f=1,','f=-1,'),1),
 'branch-duplicate-field':base.replace(b,b.replace(' !',' f=1,0,0 !'),1),
 'unknown-packet':base+'\n'+b.replace('AFC2B','AFC2X'),
 'conflicting-duplicate':base+'\n'+b.replace('f=1,','f=0,'),
 'v1-mixed':base+'\n[LFM-AFFINE-FC] old',
 'empty':''}
rejected={}
for name,badtext in bad.items():
 try:p.parse(badtext,2048,1)
 except ValueError as e:rejected[name]=str(e)
 else:raise AssertionError('accepted '+name)
for name,prompt,fast in [('wrong-prompt',1024,1),('wrong-variant',2048,0)]:
 try:p.parse(base,prompt,fast)
 except ValueError as e:rejected[name]=str(e)
 else:raise AssertionError('accepted '+name)
truncations=0
for part in (g,t,b):
 for n in range(len(part)):
  try:p.parse(base.replace(part,part[:n],1),2048,1)
  except ValueError:truncations+=1
  else:raise AssertionError('accepted truncation '+part[:n])
result={'valid_actual_formatter_cohorts':4,'valid_reverse_packet_order':True,'valid_transport_replay_reported':3,'invalid_cases_rejected':rejected,'all_prefix_truncations_rejected':truncations,'scope':'Actual C mock-clock output; not phone timing or arithmetic proof'}
(D/'checks/parser-negative.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

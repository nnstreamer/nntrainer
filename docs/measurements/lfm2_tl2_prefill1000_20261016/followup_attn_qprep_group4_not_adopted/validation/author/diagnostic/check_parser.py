#!/usr/bin/env python3
import json
from parse_trace import parse

def rec(seq,g=None,pid=42):
 if g is None:g=1152 if seq%2 else 896
 return f'10-17 02:30:00.001 {pid} 43 V adsprpc : [nntr-qprep-group4] seq={seq} groups={g} k=64 stride=2048 bytes={g*1024} !'
valid='\n'.join(rec(i) for i in range(1,161)); passes=0;negative=0
for text,model in [(valid,2048),(valid+'\n'+rec(1),2048),('',1024),('unrelated line\n',1024)]:
 parse(text,model,42);passes+=1
bad=[valid.rsplit('\n',1)[0],valid+'\n'+rec(161),valid.replace(rec(20)+'\n',''),valid+'\n'+rec(1,896),valid+'\n'+rec(1,pid=44),valid+'\n'+rec(1)[:-2],valid.replace('seq=1 ','seq=0 ',1),valid.replace('seq=1 ','seq=4294967295 ',1),valid.replace('k=64','k=32',1),valid.replace('stride=2048','stride=2049',1),valid.replace('bytes=1179648','bytes=0',1),valid.replace('groups=1152','groups=128',1),valid.replace(rec(1),rec(1,896)).replace(rec(2),rec(2,1152)),valid.replace(rec(1),rec(1).split('[nntr')[1]),valid.replace(rec(1),rec(1).replace(' !','')),valid.replace('seq=1 ','seq=-1 ',1)]
for text in bad:
 try:parse(text,2048,42)
 except ValueError:negative+=1
 else:raise AssertionError('bad trace accepted')
for text,model,pid in [(rec(1),1024,42),(valid,2048,43)]:
 try:parse(text,model,pid)
 except ValueError:negative+=1
 else:raise AssertionError('wrong cohort accepted')
print(json.dumps({'valid':passes,'negative':negative,'max_u32_seq_is_bounded':True}))

#!/usr/bin/env python3
import json
import sys
sys.dont_write_bytecode = True
from parse import parse as device_parse

def parse(text, expected_calls=None, min_hits=1):
    return device_parse(text, expected_calls, min_hits, host_unprefixed=True)

x='nntr-ring-reset q=1 used=0 !\nnntr-ring-reset q=2 used=1 !\n'
assert parse(x,2)['hits']==1
assert parse(x+x,2)['exact_transport_duplicates']==2
assert not parse(x)['full_count_independently_specified']
assert parse(x.replace('used=1','used=0'),2,0)['hits']==0
negative=[('',None), (x,1), (x,3), (x.replace('q=1','q=0'),2),
          (x.replace('q=2','q=3'),2), (x.replace('q=2','q=4294967295'),2),
          (x.replace('q=2','q=4294967296'),2), (x.replace('used=1','used=2'),2),
          (x.replace('used=1','used=-1'),2), (x.replace('used=1','used=0'),2),
          (x+'nntr-ring-reset q=2 used=0 !\n',2), (x.replace('q=1','q=-1'),2),
          ('\n'.join(x.splitlines()[::-1]),2)]
for n in range(1, len('q=2 used=1 !')):
    negative.append((x[:len(x)-n-1],2))
for text,expected in negative:
    try:parse(text,expected)
    except ValueError:pass
    else:raise AssertionError('Malformed record accepted')
prefix='10-17 01:02:03.456  123  456 V adsprpc : '
raw='\n'.join(prefix+line for line in x.splitlines())+'\n'
assert device_parse(raw,2,pid=123)['hits']==1
assert device_parse(raw+raw,2,pid=123)['exact_transport_duplicates']==2
identity_negative=[(raw.replace('  123 ', '  124 ',1),123,False), (x,123,False), (raw,None,True), (raw,None,False), (raw,123,True), (raw,0,False)]
for text,pid,bare in identity_negative:
    try:device_parse(text,2,pid=pid,host_unprefixed=bare)
    except ValueError:pass
    else:raise AssertionError('Invalid process identity accepted')
print(json.dumps({'valid':6,'negative':len(negative)+len(identity_negative),'status':'PASS'},indent=2))

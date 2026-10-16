#!/usr/bin/env python3
"""Synthetic parser tests, explicitly not DSP/device correctness evidence."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
sys.dont_write_bytecode = True
from compare_device import read_plan, parse_log, check_dumps, geometry, SCHEMA


def make_rows(plan):
    rows = []
    offset = 0
    for case in plan['cases']:
        for repeat, seed in enumerate(case['seeds']):
            row = {k: case[k] for k in ('case_id','kind','fixed','n_q','cache_from','cache_to','qh','kvh','hd','window','softcap','sinks')}
            count = case['n_q'] * case['qh'] * case['hd']
            row.update(schema=SCHEMA, record='CASE', repetition=repeat, seed=seed,
                       count=count, offset=offset, blocks=geometry(case)['blocks'],
                       mode=geometry(case)['mode'], rc=0, same_repeat=True if repeat==2 else None,
                       finite=True, input_fnv64=f'{seed:016x}')
            rows.append(row)
            offset += count * 4
    for i in range(4):
        rows.append(dict(schema=SCHEMA, record='PROBE', probe_id=i,pattern=i//2,col=i%2,
                         hd=64,rows=64,kvh=1,count=2048,layout=[1,0,32],sdk_equal=True,
                         host_dot_equal=True,planes_equal=True,rc=0,negative_count=800,
                         positive_count=800,zero_count=448,input_fnv64=f'{0x123456+i:016x}'))
    rows.append(dict(schema=SCHEMA,record='SUMMARY',cases=36,case_groups=12,probes=4,bytes=offset,rc=0))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);plan=read_plan(a.plan);rows=make_rows(plan)
    encode=lambda r:'runtime noise\n'+'\n'.join(json.dumps(x) for x in r)+'\n'
    parsed=parse_log(encode(rows),plan);negative=[]
    def bad(name,mutate):
        data=copy.deepcopy(rows);mutate(data)
        try:parse_log(encode(data),plan)
        except ValueError as e:negative.append({'name':name,'rejected':str(e)})
        else:raise AssertionError('Accepted bad case '+name)
    for key,val in [('n_q',1023),('mode',0),('blocks',159),('offset',4),('count',1),('seed',0),('rc',1),('finite',False),('same_repeat',True),('fixed',0),('input_fnv64','zzzzzzzzzzzzzzzz')]:
        bad('case_'+key,lambda data,k=key,v=val:data[0].__setitem__(k,v))
    bad('missing_case',lambda data:data.pop(0));bad('duplicate_case',lambda data:data.insert(1,copy.deepcopy(data[0])))
    bad('swapped_case',lambda data:data.__setitem__(slice(0,2),[data[1],data[0]]))
    bad('changed_aba_input',lambda data:data[2].__setitem__('input_fnv64','1111111111111111'))
    bad('unchanged_ab_input',lambda data:data[1].__setitem__('input_fnv64',data[0]['input_fnv64']))
    bad('missing_summary',lambda data:data.pop());bad('duplicate_summary',lambda data:data.append(copy.deepcopy(data[-1])))
    bad('extra_field',lambda data:data[0].__setitem__('old_toggle',1))
    bad('early_probe',lambda data:data.insert(0,data.pop(36)))
    for key,val in [('layout',[1,32,32]),('sdk_equal',False),('host_dot_equal',False),('planes_equal',False),('negative_count',2049),('rc',1),('probe_id',3)]:
        bad('probe_'+key,lambda data,k=key,v=val:data[36].__setitem__(k,v))
    for key in ('negative_count','positive_count','zero_count'):
        def omit_sign(data,k=key):
            row=data[36]
            target='positive_count' if k!='positive_count' else 'negative_count'
            row[target]+=row[k];row[k]=0
        bad('probe_missing_'+key,omit_sign)
    for key,val in [('bytes',0),('cases',35),('probes',0),('rc',1)]:
        bad('summary_'+key,lambda data,k=key,v=val:data[-1].__setitem__(k,v))
    for name,text in [('truncated',encode(rows)[:-10]),('interleaved','noise'+encode(rows).split('\n',1)[1]),('duplicate_json_key',encode(rows).replace('"case_id": 0','"case_id": 0, "case_id": 0',1)),('nonfinite_json',encode(rows).replace('"softcap": 0','"softcap": NaN',1))]:
        try:parse_log(text,plan)
        except ValueError as e:negative.append({'name':name,'rejected':str(e)})
        else:raise AssertionError('Accepted '+name)
    # One complete synthetic byte dump; use the same file for both arms so no
    # storage/copy burden is mistaken for a real control/candidate device run.
    dump=a.output/'synthetic.bin'
    with dump.open('wb') as f:
        for row in rows[:36]:
            value=float(1+row['case_id']*2+(row['repetition']==1))
            f.write(struct.pack('<f',value)*row['count'])
    checked=check_dumps(dump,dump,[parsed,parsed])
    # Exercise byte-level checks independently of the record assertions.
    with dump.open('r+b') as f:
        original=f.read(4);f.seek(0);f.write(struct.pack('<I',0x7fc00001))
    try:check_dumps(dump,dump,[parsed,parsed])
    except ValueError as e:negative.append({'name':'nonfinite_dump','rejected':str(e)})
    else:raise AssertionError('Accepted nonfinite output')
    with dump.open('r+b') as f:f.write(original)
    # A truncated alias file is enough to exercise the total-size boundary.
    small=a.output/'truncated.bin';small.write_bytes(b'\0'*4)
    try:check_dumps(dump,small,[parsed,parsed])
    except ValueError as e:negative.append({'name':'truncated_dump','rejected':str(e)})
    else:raise AssertionError('Accepted short dump')
    result={'status':'PASS','scope':'Synthetic parser-only records and bytes; no RPC/device execution.',
            'valid_full_records':len(rows),'valid_case_outputs':len(checked),'logical_dump_bytes':dump.stat().st_size,
            'negatives':negative}
    (a.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'synthetic.log').write_text(encode(rows))
    # Outputs are synthetic and redundant; only identities/counts matter to the
    # parser fixture. They are not archived as device artifacts.
    dump.unlink();small.unlink()
    print(f'PARSER SYNTHETIC PASS: {len(rows)} records, {len(checked)} outputs, {len(negative)} rejects')


if __name__=='__main__':main()

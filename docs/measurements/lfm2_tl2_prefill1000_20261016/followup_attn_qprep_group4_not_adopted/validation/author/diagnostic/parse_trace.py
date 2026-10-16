#!/usr/bin/env python3
"""Validate group4 helper dispatch within one fresh, completed child process.

Only matching tag lines are parsed; require complete PID-filtered capture and
normal generated-ID/completion checks separately. No-tag1024 proves no calls
only if the trace-enabled binary and full capture were independently verified.
Repeated byte-identical payloads are counted as transport duplicates by seq;
conflicts, gaps, truncation, mixed PID and out-of-shape counts are rejected.
"""
import argparse,json,re
from pathlib import Path
TAG='[nntr-qprep-group4]'
LINE=re.compile(r'\[nntr-qprep-group4\] seq=(\d+) groups=(\d+) k=(\d+) stride=(\d+) bytes=(\d+) !$')
PREFIX=re.compile(r'^\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+(\d+)\s+\d+\s+\w\s+')

def parse(text,model,pid=None,host=False,expected_host_calls=192):
    records={}; duplicates=0;raw=0;seen_pids=set()
    for line in text.splitlines():
        if TAG not in line: continue
        raw+=1
        if not host:
            prefix=PREFIX.match(line)
            if not prefix: raise ValueError('missing threadtime PID prefix')
            actual=int(prefix[1]);seen_pids.add(actual)
            if pid is not None and actual!=pid: raise ValueError('unexpected child PID')
            if len(seen_pids)!=1: raise ValueError('mixed child PIDs')
        tail=line[line.index(TAG):];m=LINE.fullmatch(tail)
        if not m: raise ValueError('truncated or malformed tagged record')
        seq,g,k,stride,bytes_=map(int,m.groups())
        if not 1<=seq<=0xffffffff: raise ValueError('invalid sequence')
        if g not in (1152,896) or k!=64 or stride!=2048 or bytes_!=g*4*k*4:
            raise ValueError('unsupported geometry or byte count')
        val=(g,k,stride,bytes_)
        if seq in records:
            if records[seq]!=val: raise ValueError('conflicting sequence duplicate')
            duplicates+=1
        else: records[seq]=val
    expected=expected_host_calls if host else (160 if model==2048 else 0)
    if len(records)!=expected: raise ValueError('wrong unique invocation count')
    if records and (min(records)!=1 or max(records)!=len(records)):
        raise ValueError('non-contiguous sequence')
    groups={g:sum(v[0]==g for v in records.values()) for g in (1152,896)}
    if groups[1152]!=expected//2 or groups[896]!=expected//2:
        raise ValueError('wrong 9+7 chunk counts')
    if not host:
        for seq,v in records.items():
            if v[0]!=(1152 if seq%2 else 896): raise ValueError('wrong per-head chunk order')
    return {'model':model,'pid':next(iter(seen_pids),pid),'raw_tagged_records':raw,
            'unique_calls':len(records),'transport_duplicates':duplicates,
            'groups_per_call_counts':groups,'total_groups':sum(v[0] for v in records.values()),
            'logical_gather_bytes_skipped':sum(v[3] for v in records.values()),
            'scope':'helper success after pool drain; completed single-child run and correct trace binary required separately; no time/bandwidth claim'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path)
    p.add_argument('--model',type=int,choices=[1024,2048],required=True)
    p.add_argument('--pid',type=int);p.add_argument('--host',action='store_true')
    p.add_argument('--expected-host-calls',type=int,default=192);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not a.host and a.pid is None:p.error('--pid is required for device captures')
    result=parse(a.log.read_text(),a.model,a.pid,a.host,a.expected_host_calls)
    s=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(s)
    print(s,end='')
if __name__=='__main__':main()

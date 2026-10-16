#!/usr/bin/env python3
"""Read diagnostic FARF lines; reject truncated or failed call records."""
import argparse,json,pathlib,re
FIELDS='dsp_kernel_us quant_us swiglu_worker_us dequant_wait_us acc_read_us drain_gu_us scatter_wait_us gather_us requant_wait_us blocks mm_us dma_kib dma_first_us alloc_us dma_first_kib drain_dn_us push_us staging_us acc_stride dma_desc dma_waits dma_waits_blocked dma_wait_us dma_wait_act_us dma_busy_lo_us dma_busy_hi_us dma_depth_max dma_first_ready_us dma_last_issue_us path m1_feed'.split()
assert len(FIELDS)==31
P=re.compile(r'\[LFM2-MOE-STAGES\] v=2 M=(\d+) E=(\d+) rc=(-?\d+) f=(\d+) kernel=([\d,]+)')
def parse(text):
    records=[]
    for no,line in enumerate(text.splitlines(),1):
        if '[LFM2-MOE-STAGES]' not in line: continue
        m=P.search(line)
        if not m: raise ValueError(f'line {no}: malformed stage diagnostic')
        nums=[int(v) for v in m[5].split(',')]
        if len(nums)!=len(FIELDS): raise ValueError(f'line {no}: expected 31 fields, got {len(nums)} (possibly truncated FARF)')
        rec={'line':no,'M':int(m[1]),'experts':int(m[2]),'rc':int(m[3]),'fused_flags':int(m[4]),'stages':dict(zip(FIELDS,nums))}
        if rec['rc']!=0: raise ValueError(f'line {no}: kernel rc={rec["rc"]}')
        if rec['M']<=1: raise ValueError(f'line {no}: unexpected decode record')
        rec['dma_trace_dropped_pushes']=max(0,rec['stages']['dma_desc']-512)
        records.append(rec)
    if not records: raise ValueError('no complete LFM2 fused MoE diagnostics found')
    return records
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=pathlib.Path);p.add_argument('--expected-calls',type=int);p.add_argument('--skip-initial',type=int,default=0);p.add_argument('--out',type=pathlib.Path);a=p.parse_args()
    r=parse(a.log.read_text(errors='replace'))
    if a.expected_calls is not None and len(r)!=a.expected_calls: p.error(f'expected {a.expected_calls} calls, got {len(r)}')
    if a.skip_initial < 0 or a.skip_initial >= len(r): p.error('skip-initial must leave at least one complete record')
    skipped=r[:a.skip_initial];r=r[a.skip_initial:]
    data={'source':str(a.log),'calls':len(r),'skipped_initial_records':skipped,'rows':sorted(set(q['M'] for q in r)),'fields':FIELDS,'time_sums_ms':{f:sum(q['stages'][f] for q in r)/1000 for f in FIELDS if f.endswith('_us')},'blocks':sum(q['stages']['blocks'] for q in r),'dma_trace_truncated_calls':sum(q['dma_trace_dropped_pushes']>0 for q in r),'records':r,'limitations':['Probe/FARF instrumentation adds overhead; not a normal TPS benchmark.','kernel scope excludes outer post-norm and residual epilogue, ARM staging, and RPC overhead.','swiglu_worker_us is overlapped aggregate worker time; it must not be added to kernel elapsed.','dequant_wait_us/requant_wait_us/scatter_wait_us measure exposed waits, not full worker time.','drain includes weight DMA and native expansion job waits.','DMA push ring stores only 512 descriptors; busy/first-ready/last-issue cannot characterize truncated calls.','Summary times overlap; do not sum all fields or infer DDR GB/s from truncated DMA times.']}
    s=json.dumps(data,indent=2)+'\n'; a.out.write_text(s) if a.out else print(s,end='')

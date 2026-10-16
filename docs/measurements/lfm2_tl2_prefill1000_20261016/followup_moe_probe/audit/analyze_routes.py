#!/usr/bin/env python3
"""Measure current fixed-64 HMX block padding from explicit top-k route input."""
import argparse,collections,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('routes',type=pathlib.Path);p.add_argument('--top-k',type=int,default=4);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
raw=a.routes.read_bytes(); layers=[]; tails=collections.Counter()
for no,line in enumerate(raw.decode().splitlines(),1):
    h,u,r=line.split('|'); layer,M=map(int,h.split()); union=set(map(int,u.split())); ids=list(map(int,r.split())); assert len(ids)%M==0,(no,M,len(ids)); width=len(ids)//M; assert width>=a.top_k
    cnt=collections.Counter(x for off in range(0,len(ids),width) for x in ids[off:off+a.top_k]); assert set(cnt)==union,(no,set(cnt),union)
    blocks=sum((n+63)//64 for n in cnt.values()); routed=sum(cnt.values()); padded=blocks*64
    tail=collections.Counter(n%64 for n in cnt.values() if n%64);tails.update(tail)
    layers.append({'layer':layer,'rows':M,'trace_top_n':width,'active_experts':len(cnt),'routed_rows':routed,'hmx_blocks':blocks,'hmx_row_capacity':padded,'padding_rows':padded-routed,'expert_rows':dict(sorted(cnt.items())),'tails':dict(sorted(tail.items()))})
s={k:sum(q[k] for q in layers) for k in ['active_experts','routed_rows','hmx_blocks','hmx_row_capacity','padding_rows']};s['padding_fraction']=s['padding_rows']/s['hmx_row_capacity'];s['tail_blocks']=sum(tails.values());s['tail_rows']=sum(k*v for k,v in tails.items());s['tail_blocks_le']={str(k):sum(v for n,v in tails.items() if n<=k) for k in [4,8,16,32]};s['mac_per_block']=64*(2048*3072+1536*2048);s['hmx_padded_macs']=s['mac_per_block']*s['hmx_blocks']
result={'source':str(a.routes),'sha256':hashlib.sha256(raw).hexdigest(),'top_k':a.top_k,'experts_per_layer':64,'block_rows':64,'summary':s,'layers':layers,'limitations':['Route snapshot is from an earlier normal native-TL2 run; do not silently substitute PROFILE=2 routes.','Padding MAC counts are structural arithmetic, not measured time saved.','Weight read/expand and metadata per active expert remain necessary; worker contention can negate tail-GEMV savings.']};a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(s,indent=2))

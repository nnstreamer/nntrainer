#!/usr/bin/env python3
"""Ordering test of the actual kernel's extracted out: joins and trace block.

Kernel math and DMA are not emulated here. Synthetic pending work updates the
real counter at completion, proving the extracted log reads only after all joins.
The separate actual-HVX test proves helper entries, lanes, atomic counts and bytes.
"""
from pathlib import Path
import json,re,subprocess
D=Path(__file__).resolve().parent;O=D/'host';O.mkdir(exist_ok=True)
s=(D/'candidate/nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c').read_text()
start=s.index('out:\n',s.index('int hexkl_mm_u8i4_moe_layer_run('));end=s.index('  if (hexkl_probe_on)',start)
drain=s[start+len('out:\n'):end]
start=s.index('#if defined(NNTR_LFM2_TL2_GU_TRACE)',end);end=s.index('#endif',start)+len('#endif')
trace=s[start:end]
assert s.index('  hvx_worker_pool_wait(pool);',s.index('out:\n'))<start
assert s.index('  return rc;',end)>end
slots=re.search(r'#define MOE_EXP_SLOTS (\d+)u',s)[1]
text=r'''
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
enum {AEE_SUCCESS=0};
#define NNTR_LFM2_TL2_GU_TRACE 1
#define MOE_EXP_SLOTS SLOTSu
static uint32_t counter[4], expected[4], tl2_gu_trace_seq;
static uint32_t pending,workers,logs,last_delta[4],last_active,last_native;
static void hexkl_moe_tl2_gu_trace_counts(uint32_t out[4]) {for(unsigned i=0;i<4;++i)out[i]=__atomic_load_n(&counter[i],__ATOMIC_RELAXED);}
struct job {unsigned bit;};
static void finish(unsigned bit) {assert(pending&(1u<<bit));pending&=~(1u<<bit);for(unsigned k=0;k<4;++k)__atomic_fetch_add(&counter[k],bit+k+1u,__ATOMIC_RELAXED);}
static void hvx_worker_pool_wait(void *pool) {(void)pool;finish(0);}
static void hvx_worker_pool_wait_bg(void *pool,struct job *job,uint32_t n) {(void)pool;assert(n==UINT32_MAX);finish(job->bit);}
static uint32_t hvx_worker_pool_workers(void *pool) {(void)pool;return workers;}
static void log_call(const char *fmt,...) {
 char line[128];va_list ap;va_start(ap,fmt);int n=vsnprintf(line,sizeof line,fmt,ap);va_end(ap);
 assert(pending==0&&n>0&&n<(int)sizeof line&&line[n-1]=='!');
 unsigned seq,M,K,I,N,lanes;
 assert(sscanf(line,"[nntr-tl2-gu-direct] seq=%u M=%u K=%u I=%u N=%u E=%u native=%u lanes=%u fc=%u ft=%u uc=%u ut=%u !",&seq,&M,&K,&I,&N,&last_active,&last_native,&lanes,&last_delta[0],&last_delta[1],&last_delta[2],&last_delta[3])==12);
 assert(seq==1&&M==1&&K==2048&&I==1536&&N==2048&&lanes==workers+1);
 assert(memcmp(last_delta,expected,sizeof expected)==0);++logs;
}
#define FARF(level,...) log_call(__VA_ARGS__)
static int actual_out(unsigned mask,int rc,unsigned use_m1,unsigned M,unsigned K,unsigned inter,unsigned N_out,unsigned n_active,unsigned n_tl2) {
 void *pool=0;struct job jobs[MOE_EXP_SLOTS+1],*last_job=mask&1u?&jobs[0]:NULL,exp_job[MOE_EXP_SLOTS];unsigned exp_live[MOE_EXP_SLOTS]={0};uint32_t gu_before[4];
 for(unsigned k=0;k<4;++k){counter[k]=UINT32_MAX-7u;expected[k]=k+1u;}
 pending=1u;jobs[0].bit=1u;
 if(last_job) {pending|=2u;for(unsigned k=0;k<4;++k)expected[k]+=k+2u;}
 for(unsigned i=0;i<MOE_EXP_SLOTS;++i) {exp_job[i].bit=i+2u;if(mask&(1u<<(i+1u))) {exp_live[i]=1;pending|=1u<<(i+2u);for(unsigned k=0;k<4;++k)expected[k]+=i+k+3u;}}
 tl2_gu_trace_seq=0;
 hexkl_moe_tl2_gu_trace_counts(gu_before);
'''.replace(' SLOTSu',' '+slots+'u')
text+=drain+'\n'+trace+r'''
 assert(pending==0);return rc;
}
int main(void) {
 unsigned cases=0;
 const unsigned masks[]={0,1,2,0x1ffffu,0xaaaa,0x15555};
 const unsigned pools[]={0,2,7};
 for(unsigned m=0;m<6;++m)for(unsigned w=0;w<3;++w)for(unsigned variant=0;variant<9;++variant) {
  workers=pools[w];logs=0;int rc=variant==1?79:0;
  assert(actual_out(masks[m],rc,variant==2?0:1,variant==3?1024:1,variant==4?2816:2048,variant==5?704:1536,variant==6?2816:2048,4,variant==7?0:variant==8?3:4)==rc);
  assert(logs==(variant==0||variant==8));
  if(logs)assert(last_active==4&&last_native==(variant==8?3u:4u));
  ++cases;
 }
 printf("PASS: %u actual out-join/trace combinations; all FG/last-BG/16 expansion-slot joins precede snapshot/log; success/shape/native filters and uint32 wrap exact\n",cases);
}
'''
(O/'drain-check.c').write_text(text)
cmd=['gcc','-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer','-fno-pie','-no-pie',str(O/'drain-check.c'),'-o',str(O/'drain-check')]
commands=[cmd,[str(O/'drain-check')]];(O/'drain-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for label,cmd in zip(('drain-build','drain-run'),commands):
 with (O/(label+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
print((O/'drain-run.log').read_text(),end='')

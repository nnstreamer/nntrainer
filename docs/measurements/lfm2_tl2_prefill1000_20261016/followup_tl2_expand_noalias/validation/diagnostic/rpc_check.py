#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
R=Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
O=D/'host'
old=(D/'baseline/test/htp/nntr_hvx_mm_u8i4.c').read_text()
new=(D/'candidate/test/htp/nntr_hvx_mm_u8i4.c').read_text()
def block(s,a,b):
    start=s.index(a);return s[start:s.index(b,start)]
text=r'''
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "htp_router_handoff.h"
#include "hvx_gemv_u8tl2.h"
typedef uintptr_t remote_handle64;
typedef uint32_t uint32;
enum { AEE_SUCCESS=0, AEE_EBADPARM=2, AEE_EUNSUPPORTED=3 };
#define HEXKL_MOE_FLAG_GELU_TANH 128u
typedef struct { htp_router_handoff_scratch router_handoff; unsigned char *base; const float *router_act; unsigned router_used; size_t cap; } Scratch;
typedef struct { Scratch moe_scratch; int hmx_locked; int weights_u8i4; unsigned char *vtcm_base; uint32_t vtcm_size,config_off,moe_flags; void *quant_pool; } nntr_hvx_session;
static unsigned logs, calls, drained, forced_rc, traced_count, traced_active;
static nntr_hvx_session *current;
static uint32_t last[7];
static char logline[256];
static void log_call(const char *fmt, ...) {
  va_list ap;va_start(ap,fmt);vsnprintf(logline,sizeof(logline),fmt,ap);va_end(ap);
  if (strstr(logline,"[nntr-tl2-disjoint]")) {
    assert(drained);
    unsigned M,K,I,N,E,count;
    assert(sscanf(logline,"[nntr-tl2-disjoint] M=%u K=%u I=%u N=%u E=%u count=%u",&M,&K,&I,&N,&E,&count)==6);
    assert(M==last[0] && K==last[1] && I==last[2] && N==last[3]);
    traced_count=count;traced_active=E;++logs;
  }
}
#define FARF(level,...) log_call(__VA_ARGS__)
static int hexkl_mm_u8i4_moe_layer_run(int *table,unsigned char *vtcm,uint32_t size,uint32_t config,uint32 M,uint32 K,uint32 inter,uint32 N,uint32 E,const uint32 *gu,const uint32 *dn,const uint32 *rows,const uint32 *counts,const float *rw,const float *x,float *out,void *pool,Scratch *scratch,uint32_t flags) {
  assert(table==&current->weights_u8i4 && vtcm==current->vtcm_base && size==current->vtcm_size && config==current->config_off && pool==current->quant_pool && scratch==&current->moe_scratch);
  assert(gu&&dn&&rows&&counts&&rw&&x&&out);
  ++calls;drained=0;
  last[0]=M;last[1]=K;last[2]=inter;last[3]=N;last[4]=E;last[5]=flags;last[6]=0;
  for (uint32 e=0;e<E;++e) last[6]+=counts[e]!=0;
  for (uint32 n=0;n<last[6]*784u;++n)
    hvx_tl2_expand_wh96_disjoint(NULL,0,0,49152,1,NULL);
  drained=1;
  if (!forced_rc) out[0]=17.5f;
  return (int)forced_rc;
}
'''
text+=block(old,'static int check_moe_layer_args(','int nntr_hvx_mm_u8i4_moe_layer(')
for label,src in [('original',old),('traced',new)]:
    part=block(src,'int nntr_hvx_mm_u8i4_moe_layer(','int nntr_hvx_router_logits_f32(')
    text+=part.replace('nntr_hvx_mm_u8i4_moe_layer(',label+'_moe(')
text+=r'''
int main(void) {
  const uint32 shapes[][5]={{1024,2048,1536,2048,0},{2048,2048,1536,2048,0},{1,2048,1536,2048,0},{2,2048,1536,2048,0},{1024,2816,704,2816,1},{1024,2048,1536,1024,0},{1024,2048,1536,2048,0x8000000a}};
  const uint32 gu[]={1,2,3},dn[]={4,5,6},rows[]={0,0,0};const float rw[]={1,1,1},input[]={2};
  unsigned tested=0,total_logs=0;
  for (unsigned sh=0;sh<sizeof(shapes)/sizeof(*shapes);++sh) for(unsigned active=0;active<=3;++active) for(unsigned mode=0;mode<4;++mode) {
    const uint32 *q=shapes[sh];uint32 counts[]={0,0,0};for(unsigned e=0;e<active;++e)counts[e]=1;
    nntr_hvx_session a={0},b={0};a.hmx_locked=b.hmx_locked=1;a.moe_flags=b.moe_flags=0x4000;
    int nr=(int)active, len=mode==2?nr+1:nr;float original=-3,traced=-3;int rc[2];uint32 recorded[7];unsigned call_a;
    logs=calls=0;forced_rc=mode==1?79:0;
    current=&a;
    rc[0]=original_moe(mode==3?0:(remote_handle64)&a,q[0],q[1],q[2],q[3],q[4],gu,3,dn,3,rows,nr,counts,3,rw,len,input,(int)(q[0]*q[1]),&original,(int)(q[0]*q[3]));
    assert(logs==0);call_a=calls;memcpy(recorded,last,sizeof(last));calls=0;
    current=&b;
    rc[1]=traced_moe(mode==3?0:(remote_handle64)&b,q[0],q[1],q[2],q[3],q[4],gu,3,dn,3,rows,nr,counts,3,rw,len,input,(int)(q[0]*q[1]),&traced,(int)(q[0]*q[3]));
    assert(rc[0]==rc[1]&&original==traced&&calls==call_a&&memcmp(recorded,last,sizeof(last))==0);
    const unsigned expected=mode==0&&q[0]>1&&q[1]==2048&&q[2]==1536&&q[3]==2048;
    assert(logs==expected);
    if(expected) { assert(traced_count==active*784u&&traced_active==active);++total_logs; }
    ++tested;
  }
  printf("PASS: %u actual plain RPC comparison cases; %u success-only exact delta/log cases; unchanged return/output/kernel arguments; failures and unsupported shapes excluded\n",tested,total_logs);
}
'''
(O/'rpc-check.c').write_text(text)
lib=Path('/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/tools/HEXAGON_Tools/19.0.04/Tools/libnative/lib/libnative.a')
commands=[['gcc','-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer','-fno-pie','-DNNTR_LFM2_TL2_EXPAND_TRACE=1','-I',str(D/'candidate/nntrainer/tensor/htp_backend/hvx'),'-I',str(R/'nntrainer/tensor/htp_backend'),'-c',str(O/'rpc-check.c'),'-o',str(O/'rpc-check.o')],['g++','-fsanitize=address,undefined','-no-pie','-Wl,--gc-sections',str(O/'rpc-check.o'),str(O/'actual.o'),str(lib),'-lm','-o',str(O/'rpc-check')],[str(O/'rpc-check')]]
for label,cmd in zip(['rpc-build','rpc-link','rpc-run'],commands):
    with (O/(label+'.log')).open('w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
(O/'rpc-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
print((O/'rpc-run.log').read_text(),end='')

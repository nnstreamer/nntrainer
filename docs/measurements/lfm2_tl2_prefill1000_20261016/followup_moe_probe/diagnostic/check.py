#!/usr/bin/env python3
"""Compare extracted unmodified and diagnostic fused entry points on host."""
import argparse, pathlib, re, subprocess
p=argparse.ArgumentParser(); p.add_argument('repo',type=pathlib.Path); p.add_argument('out',type=pathlib.Path); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
f='test/htp/nntr_hvx_mm_u8i4.c'; new=(a.repo/f).read_text(); old=subprocess.check_output(['git','show','HEAD:'+f],cwd=a.repo,text=True)
def block(s,start,end): return s[s.index(start):s.index(end,s.index(start))]
def enum(s):
    x=s.index('enum {'); return s[x:s.index('};',x)+2]
probe=(a.repo/'nntrainer/tensor/htp_backend/hmx/hexkl_probe.h').read_text()
pre=r'''
#define _POSIX_C_SOURCE 200112L
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uint32_t uint32;
typedef uintptr_t remote_handle64;
enum { AEE_SUCCESS=0, AEE_EBADPARM=1, AEE_ENOMEMORY=2, AEE_EINVALIDFORMAT=3 };
typedef struct { const float *pre_gamma; float pre_eps; int out_heap; } Scratch;
typedef struct { unsigned marker; } WeightTable;
typedef struct { Scratch moe_scratch; float *moe_res; uint32_t moe_res_n; void *quant_pool; WeightTable weights_u8i4; void *vtcm_base; uint32_t vtcm_size,config_off,moe_flags; int hmx_locked; } nntr_hvx_session;
static nntr_hvx_session *current_session;
#define HEXKL_MOE_FLAG_GELU_TANH 128u
#define AEE_EUNSUPPORTED 91
static void *memalign(size_t a,size_t n) { void *p=NULL; return posix_memalign(&p,a,n)?NULL:p; }
static char logline[2048];
static int logs, resets, force_rc, calls, normcalls, addcalls, want_on;
static uint64_t clock_us;
static void farf(const char *fmt,...) { if(!fmt) abort(); va_list ap; va_start(ap,fmt); vsnprintf(logline,sizeof(logline),fmt,ap); va_end(ap); if(strstr(fmt,"[LFM2-MOE-STAGES]")) ++logs; }
#define FARF(level,...) farf(__VA_ARGS__)
'''
pre += enum(probe) + '\n'
pre += r'''
static uint64_t hexkl_probe_us[HEXKL_PROBE_N];
static int hexkl_probe_on;
static void hexkl_probe_reset(int enable) { memset(hexkl_probe_us,0,sizeof(hexkl_probe_us)); hexkl_probe_on=enable; ++resets; }
static uint64_t hexkl_probe_now(void) { return clock_us; }
typedef struct { uint32 M,K,I,N,act; int prelen,heap; float eps; int lens[8]; const void *ptrs[6]; } Call;
static Call last;
static int mock_moe_layer(remote_handle64 handle,uint32 M,uint32 K,uint32 inter,uint32 N_out,uint32 act,const uint32 *gu,int gul,const uint32 *dn,int dnl,const uint32 *rows,int rl,const uint32 *counts,int cl,const float *rw,int rwl,const float *x,int xl,float *out,int ol) {
 nntr_hvx_session *s=(nntr_hvx_session *)handle; ++calls; assert(hexkl_probe_on==want_on);
 memset(&last,0,sizeof(last)); last.M=M;last.K=K;last.I=inter;last.N=N_out;last.act=act;last.heap=s->moe_scratch.out_heap;last.eps=s->moe_scratch.pre_eps;last.prelen=s->moe_scratch.pre_gamma!=NULL;
 int lens[]={gul,dnl,rl,cl,rwl,xl,ol,out==s->moe_res};memcpy(last.lens,lens,sizeof(lens)); const void *ptrs[]={gu,dn,rows,counts,rw,x}; memcpy(last.ptrs,ptrs,sizeof(ptrs));
 if(hexkl_probe_on) for(int i=0;i<HEXKL_PROBE_N;++i) hexkl_probe_us[i]=(uint64_t)i*13+7;
 clock_us+=123;
 if(force_rc) return force_rc;
 for(int i=0;i<ol;++i) out[i]=x[i%xl]*0.5f+(s->moe_scratch.pre_gamma?s->moe_scratch.pre_gamma[i%K]*0.25f:0.0f);
 return AEE_SUCCESS;
}
static int hexkl_mm_u8i4_moe_layer_run(WeightTable *weights,void *vtcm,uint32_t size,uint32_t config,uint32 M,uint32 K,uint32 inter,uint32 N,uint32 E,const uint32 *gu,const uint32 *dn,const uint32 *rows,const uint32 *counts,const float *rw,const float *x,float *out,void *pool,Scratch *scratch,uint32_t flags) {
 nntr_hvx_session *s=current_session; assert(weights==&s->weights_u8i4&&vtcm==s->vtcm_base&&size==s->vtcm_size&&config==s->config_off&&pool==s->quant_pool&&scratch==&s->moe_scratch);assert((flags&~HEXKL_MOE_FLAG_GELU_TANH)==s->moe_flags);
 uint32 act=(flags&HEXKL_MOE_FLAG_GELU_TANH)!=0; uint32 nr=0; for(uint32 i=0;i<E;++i) nr+=counts[i];
 return mock_moe_layer((remote_handle64)s,M,K,inter,N,act,gu,(int)E,dn,(int)E,rows,(int)nr,counts,(int)E,rw,(int)nr,x,(int)(M*K),out,(int)(M*N));
}
static void hvx_rmsnorm_rows_f32(const float *in,float *out,uint32 M,uint32 N,uint32 stride,const float *g,float eps,void *pool) {
 (void)stride;(void)pool; assert(hexkl_probe_on==0); ++normcalls; for(uint32 i=0;i<M*N;++i) out[i]=in[i]*g[i%N]+eps;
}
static int hvx_rmsnorm_add_res_f32(float *out,const float *x,const float *moe,const float *x2,uint32 M,uint32 N,const float *g,float eps,float scale,void *pool) {
 (void)pool; assert(hexkl_probe_on==0); ++addcalls; for(uint32 i=0;i<M*N;++i) out[i]=x[i]+moe[i]+x2[i]*scale+(g?g[i%N]:0.0f)+eps; return 0;
}
'''
pre += block(new,'enum {\n  MOE_T_DSP_TOTAL','/** @brief Shared by both entry')
pre += block(old,'static int check_moe_layer_args(','int nntr_hvx_mm_u8i4_moe_layer(')
def plain(s,prefix):
    return block(s,'int nntr_hvx_mm_u8i4_moe_layer(','int nntr_hvx_router_logits_f32(').replace('nntr_hvx_mm_u8i4_moe_layer(',prefix+'_plain(')
def entries(s,prefix):
    s=block(s,'static int moe_layer_norm_impl(','int nntr_hvx_mm_u8i4_moe_layer_timed(')
    s=s.replace('moe_layer_norm_impl',prefix+'_impl').replace('nntr_hvx_mm_u8i4_moe_layer_norm_add',prefix+'_add').replace('nntr_hvx_mm_u8i4_moe_layer_norm',prefix+'_norm')
    return s.replace('nntr_hvx_mm_u8i4_moe_layer(',prefix+'_plain(')
pre += plain(old,'base')+entries(old,'base')+'\n'+plain(new,'diag')+entries(new,'diag')
pre += r'''
static int run(int variant,int add,nntr_hvx_session *s,uint32 M,uint32 K,uint32 I,uint32 N,uint32 act,int prenorm,int postnorm,const float *g,const float *x,float *out) {
 static const uint32 gu[]={17,19},dn[]={18,20},rows[]={0,1},counts[]={1,1}; static const float rw[]={0.25f,0.75f};
 current_session=s; s->hmx_locked=1;s->moe_flags=0x4000u;
 if(add<0) {
  if(variant) return diag_plain((remote_handle64)s,M,K,I,N,act,gu,2,dn,2,rows,2,counts,2,rw,2,x,(int)(M*K),out,(int)(M*N));
  return base_plain((remote_handle64)s,M,K,I,N,act,gu,2,dn,2,rows,2,counts,2,rw,2,x,(int)(M*K),out,(int)(M*N));
 }
 if(add) {
  if(variant) return diag_add((remote_handle64)s,M,K,I,N,act,0.00001f,0.125f,g,prenorm?(int)K:0,g,postnorm?(int)N:0,g,(int)N,gu,2,dn,2,rows,2,counts,2,rw,2,x,(int)(M*K),x,(int)(M*N),out,(int)(M*N));
  return base_add((remote_handle64)s,M,K,I,N,act,0.00001f,0.125f,g,prenorm?(int)K:0,g,postnorm?(int)N:0,g,(int)N,gu,2,dn,2,rows,2,counts,2,rw,2,x,(int)(M*K),x,(int)(M*N),out,(int)(M*N));
 }
 if(variant) return diag_norm((remote_handle64)s,M,K,I,N,act,0.00001f,g,prenorm?(int)K:0,g,postnorm?(int)N:0,gu,2,dn,2,rows,2,counts,2,rw,2,x,(int)(M*K),out,(int)(M*N));
 return base_norm((remote_handle64)s,M,K,I,N,act,0.00001f,g,prenorm?(int)K:0,g,postnorm?(int)N:0,gu,2,dn,2,rows,2,counts,2,rw,2,x,(int)(M*K),out,(int)(M*N));
}
int main(void) {
 unsigned tests=0,traced=0;
 const uint32 shapes[][5]={{1,2048,1536,2048,0},{2,2048,1536,2048,0},{63,2048,1536,2048,0},{64,2048,1536,2048,0},{65,2048,1536,2048,0},{1024,2048,1536,2048,0},{2048,2048,1536,2048,0},{2,2816,704,2816,1},{2,2048,11776,2048,0},{2,2048,1536,2048,1},{2,256,1536,256,0}};
 for(size_t sh=0;sh<sizeof(shapes)/sizeof(*shapes);++sh) for(int add=-1;add<2;++add) for(int pre=0;pre<2;++pre) for(int post=0;post<2;++post) for(int fail=0;fail<2;++fail) {
  const uint32 *q=shapes[sh];uint32 M=q[0],K=q[1],I=q[2],N=q[3],act=q[4];size_t n=(size_t)M*N;
  float *g=malloc(N*sizeof(float)),*x=malloc((size_t)M*K*sizeof(float)),*a=malloc(n*sizeof(float)),*b=malloc(n*sizeof(float));assert(g&&x&&a&&b);
  for(uint32 i=0;i<N;++i) g[i]=0.5f+(float)(i%7)*0.125f;
  for(size_t i=0;i<(size_t)M*K;++i) x[i]=(float)((int)(i%257)-128)*0.03125f;
  memset(a,0x5a,n*sizeof(float));memcpy(b,a,n*sizeof(float)); nntr_hvx_session sa={0},sb={0};
  logs=resets=calls=normcalls=addcalls=hexkl_probe_on=want_on=0;clock_us=1000;force_rc=fail?73:0;
  int ra=run(0,add,&sa,M,K,I,N,act,pre,post,g,x,a);Call ca=last;int na=normcalls,aa=addcalls;assert(calls==1&&logs==0&&resets==0);
  int expected=M>1&&K==2048&&I==1536&&N==2048&&act==0;
#if !NNTR_LFM2_MOE_STAGE_TRACE
  expected=0;
#endif
  want_on=expected;calls=normcalls=addcalls=0;clock_us=1000;
  int rb=run(1,add,&sb,M,K,I,N,act,pre,post,g,x,b);
  assert(ra==rb&&calls==1&&memcmp(&ca,&last,sizeof(ca))==0&&memcmp(a,b,n*sizeof(float))==0&&na==normcalls&&aa==addcalls);
  assert(!sa.moe_scratch.pre_gamma&&!sb.moe_scratch.pre_gamma&&!sa.moe_scratch.out_heap&&!sb.moe_scratch.out_heap&&!hexkl_probe_on);
  assert(logs==expected&&resets==expected);
  if(expected) {char *csv=strstr(logline,"kernel=");assert(csv);int commas=0;for(char *z=csv;*z;++z) commas+=*z==',';assert(commas==MOE_N_STAGES-1);assert(strncmp(csv,"kernel=123,",11)==0);++traced;}
  free(sa.moe_res);free(sb.moe_res);free(g);free(x);free(a);free(b);++tests;
 }
 printf("PASS: %u actual plain/norm/add dispatch/output/scratch/failure cases; %u trace cases; stages=%d\n",tests,traced,MOE_N_STAGES);
}
'''
(a.out/'check.c').write_text(pre)
for flag in [0,1]:
    exe=a.out/f'check-{flag}'
    subprocess.run(['gcc','-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',f'-DNNTR_LFM2_MOE_STAGE_TRACE={flag}',str(a.out/'check.c'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

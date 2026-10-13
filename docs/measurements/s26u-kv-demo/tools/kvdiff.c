// kvdiff a.bin b.bin S W : per layer/band max|a-b| and rms(b) of two KV dumps (Gemma4 layout: sliding = window rows, full = S rows)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
static float hf(uint16_t h){uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023;float v;if(e==0)v=ldexpf((float)m,-24);else if(e==31)v=m?NAN:INFINITY;else v=ldexpf((float)(m|1024),(int)e-25);return s?-v:v;}
int main(int c,char**v){FILE*a=fopen(v[1],"rb"),*b=fopen(v[2],"rb");int S=atoi(v[3]),W=atoi(v[4]),W2=c>5?atoi(v[5]):W;
 printf("layer kind  K|V  band          max|d|   rms(ref)  frac>0.05*rms\n");
 for(int l=0;l<30;l++){int slid=(l%6!=5);int rows=slid?W:S,w=slid?2048:1024,base=slid?S-W:0;
  int rows2=slid?W2:S,base2=slid?S-W2:0;
  for(int kv=0;kv<2;kv++){size_t n=(size_t)rows*w,n2=(size_t)rows2*w;uint16_t*A=malloc(n*2),*B2=malloc(n2*2),*B=malloc(n*2);fread(A,2,n,a);fread(B2,2,n2,b);
   /* B2 holds absolute rows [base2, S); A holds [base, S): compare over the overlap, mapped into A's row space (rows before base2 count as mismatch-free by skipping) */
   for(int r=0;r<rows;r++){int pos=r+base; if(pos<base2){memcpy(B+(size_t)r*w,A+(size_t)r*w,w*2);} else memcpy(B+(size_t)r*w,B2+(size_t)(pos-base2)*w,w*2);} free(B2);
   int lo[3],hi[3],nb=0; for(int e=0;e<3;e++){int l0=(e==0)?0:(e==1)?2048:4096, h0=(e==0)?2048:(e==1)?4096:S; if(l0<base)l0=base; if(h0>S)h0=S; if(h0>l0){lo[nb]=l0;hi[nb]=h0;nb++;}}
   for(int t=0;t<nb;t++){double mx=0,ss=0,cnt=0,big=0;for(int r=lo[t]-base;r<hi[t]-base;r++)for(int j=0;j<w;j++){float x=hf(A[(size_t)r*w+j]),y=hf(B[(size_t)r*w+j]);double d=fabs(x-y);if(d>mx)mx=d;ss+=y*y;cnt++;}
    double rms=sqrt(ss/cnt);for(int r=lo[t]-base;r<hi[t]-base;r++)for(int j=0;j<w;j++){float x=hf(A[(size_t)r*w+j]),y=hf(B[(size_t)r*w+j]);if(fabs(x-y)>0.05*rms)big++;}
    if(l<3||l==5||l==11||l==29||big/cnt>0.05)printf("%5d %s  %s  [%4d,%4d)  %8.3f  %9.3f  %6.2f%%\n",l,slid?"slid":"full",kv?"V":"K",lo[t],hi[t],mx,rms,100*big/cnt);}
   free(A);free(B);}}return 0;}

#!/usr/bin/env python3
"""rowdiff2.py a.f32 b.f32 M N: rms(diff)/rms(ref) overall, and the rows whose relative diff exceeds 5%"""
import sys, array, math
M=int(sys.argv[3]); N=int(sys.argv[4]); a=array.array('f'); b=array.array('f')
a.fromfile(open(sys.argv[1],'rb'),M*N); b.fromfile(open(sys.argv[2],'rb'),M*N)
sd=0.0; ss=0.0; bad=[]
for r in range(M):
    rd=0.0; rs=0.0
    for c in range(N):
        d=a[r*N+c]-b[r*N+c]; rd+=d*d; rs+=b[r*N+c]**2
    sd+=rd; ss+=rs
    if rs>0 and math.sqrt(rd/rs)>0.05: bad.append(r)
print("rel rms diff %.5f  rows>5%%: %d  %s" % (math.sqrt(sd/ss) if ss else 0, len(bad), bad[:20]))

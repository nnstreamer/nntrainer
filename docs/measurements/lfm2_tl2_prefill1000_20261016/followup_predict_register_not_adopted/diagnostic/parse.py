#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict single-session LFM decode predictor branch diagnostic (not timing)."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--log',type=Path,required=True)
p.add_argument('--enabled',type=int,choices=(0,1),required=True)
p.add_argument('--tokens',type=int,default=64)
p.add_argument('--moe-layers',type=int,default=38)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
lines=[l.strip() for l in a.log.read_text().splitlines() if '[LFM-PREDICT-REG]' in l]
if len(lines)!=1:raise ValueError('expected exactly one marker from one session, got '+str(len(lines)))
m=re.fullmatch(r'\[LFM-PREDICT-REG\] callbacks=(\d+) dots=(\d+) qualified=(\d+) fast=(\d+) E',lines[0])
if not m:raise ValueError('malformed marker (use direct stderr, not duplicate logcat deliveries)')
values=dict(zip(['callbacks','dots','qualified','fast'],map(int,m.groups())))
expected=a.tokens*(a.moe_layers-1)
if a.tokens<=0 or a.moe_layers<2:raise ValueError('invalid expected cohort')
for k in ('callbacks','dots','qualified'):
 if values[k]!=expected:raise ValueError(f'{k}={values[k]} != {expected}; tracing-extra-dots/multiple sessions/partial run not accepted')
if values['fast']!=expected*a.enabled:raise ValueError('actual optimized helper call count mismatch')
a.output.write_text(json.dumps({'status':'PASS','enabled':a.enabled,**values,'scope':'Actual helper dispatch count only; no timing or numerical result proof.'},indent=2)+'\n')
print(json.dumps(values))

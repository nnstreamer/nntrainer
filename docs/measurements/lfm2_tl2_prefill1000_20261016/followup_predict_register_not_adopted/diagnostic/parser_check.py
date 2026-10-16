#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import json,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
parser=Path(__file__).with_name('parse.py')
good='[LFM-PREDICT-REG] callbacks=2368 dots=2368 qualified=2368 fast={} E\n'
cases=[('off',good.format(0),0,True),('on',good.format(2368),1,True)]
bad=[('missing',''),('duplicate',good.format(0)*2),('truncated',good.format(0)[:-3]),
 ('callbacks',good.format(0).replace('callbacks=2368','callbacks=2367')),
 ('dots',good.format(0).replace('dots=2368','dots=9472')),
 ('qualified',good.format(0).replace('qualified=2368','qualified=0')),
 ('fast',good.format(1)),('negative',good.format(-1)),
 ('extra',good.format(0).replace(' E',' other=1 E')),('garbage',good.format('nan'))]
cases += [(n,s,0,False) for n,s in bad]
results=[]
for n,s,enabled,success in cases:
 path=out/(n+'.log');path.write_text(s)
 cmd=['python3',str(parser),'--log',str(path),'--enabled',str(enabled),'--output',str(out/(n+'.json'))]
 result=subprocess.run(cmd,capture_output=True,text=True)
 assert (result.returncode==0)==success,(n,result.stdout,result.stderr)
 results.append({'name':n,'expected_success':success,'returncode':result.returncode})
(out/'validation.json').write_text(json.dumps({'valid':2,'invalid':10,'cases':results},indent=2)+'\n')
print('PREDICT TRACE PARSER 2 valid / 10 invalid PASS')

#!/usr/bin/env python3
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def functions(path, objdump):
    data=path.read_bytes(); h=struct.unpack_from('<16sHHIIIIIHHHHHH',data)
    sections=[struct.unpack_from('<IIIIIIIIII',data,h[6]+i*h[11]) for i in range(h[12])]
    limits={}
    for table in sections:
        if table[1]!=2:continue
        string=sections[table[6]];names=data[string[4]:string[4]+string[5]]
        for i in range(table[5]//table[9]):
            name,value,size,info,other,index=struct.unpack_from('<IIIBBH',data,table[4]+i*table[9])
            if info&15==2 and index and size:
                limits[names[name:].split(b'\0',1)[0].decode()]=(value,value+size)
    text=subprocess.check_output([str(objdump),'-dr',str(path)],text=True)
    records={};name=None
    for line in text.splitlines():
        match=re.match(r'^([0-9a-f]+) <([^>]+)>:',line)
        if match:
            name=match[2];records[name]=[]
        instruction=re.match(r'^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){4}[0-9a-f]{8}\s+(.*)$',line)
        if not instruction or name not in limits:continue
        address=int(instruction[1],16)
        if not limits[name][0]<=address<limits[name][1]:continue
        value=re.sub(r'0x[0-9a-f]+ <([^>]+)>',r'<\1>',instruction[2])
        value=re.sub(r'\b[0-9a-f]+:\s+(R_HEX_)',r'\1',value)
        value=re.sub(r'(loop[01]\()0x([0-9a-f]+)', lambda m: m[1]+'FUNC+'+hex(int(m[2],16)-limits[name][0]),value)
        value=re.sub(r'\s+',' ',value).strip()
        records[name].append(value)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--objdump',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads((a.build/'compilations.json').read_text());out={};a.output.mkdir(parents=True,exist_ok=True)
    for stem in ['hexkl_dma_ring','hexkl_mm_u8i4_moe']:
        f={}
        for side in ['baseline','candidate']:
            r=next(v for k,v in c.items() if k.startswith(side+'-') and v['source'].endswith(stem+'.c'))
            f[side]=functions(Path(r['object']),a.objdump)
        before,after=f['baseline'],f['candidate'];changed={}
        for name in before.keys()&after.keys():
            if before[name]!=after[name]:
                diff='\n'.join(difflib.unified_diff(before[name],after[name],fromfile='before',tofile='candidate'))+'\n'
                (a.output/(name+'.diff')).write_text(diff)
                normalize=lambda lines: [value for line in lines for value in [re.sub(r'\s+', ' ', line.replace('{', '').replace('}', '')).strip()] if value != 'nop']
                changed[name]={'before_instructions':len(before[name]),'candidate_instructions':len(after[name]),'non_nop_instruction_text_equal':normalize(before[name])==normalize(after[name]),'diff_sha256':hashlib.sha256(diff.encode()).hexdigest()}
        out[stem]={'unchanged_functions':len(before.keys()&after.keys())-len(changed),'changed':changed,'added':sorted(after.keys()-before.keys()),'removed':sorted(before.keys()-after.keys())}
    (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()

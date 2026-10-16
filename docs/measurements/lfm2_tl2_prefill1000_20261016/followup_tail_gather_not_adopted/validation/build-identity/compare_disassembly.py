#!/usr/bin/env python3
"""Screen linked instruction changes; explicitly normalize relocations/PC targets.
This is not a proof of semantic equality: packet braces, immexts, branch offsets
and nop padding are ignored, and GOT references are symbolized from ELF relocs.
Exact source/object/disassembly hashes remain the primary immutable evidence.
"""
import argparse,difflib,json,re,subprocess
from pathlib import Path

def functions(text, got=None):
    result={};name=None
    for line in text.splitlines():
        h=re.match(r'^([0-9a-f]+) <([^>]+)>:',line)
        if h:
            name=h[2];result[name]=[];continue
        m=re.match(r'^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){4}[0-9a-f]{8}\s*(.*)',line)
        if not m or name is None:continue
        s=m[2]
        if 'immext(' in s:continue
        s=re.sub(r'0x[0-9a-f]+(?= <)','TARGET',s)
        s=re.sub(r'(?<=add\(pc,##)-?0x[0-9a-f]+','PCREL',s)
        s=re.sub(r'(?<=loop0\()0x[0-9a-f]+','TARGET',s)
        s=re.sub(r'(?<=loop1\()0x[0-9a-f]+','TARGET',s)
        s=re.sub(r'<([^>]+)\+0x[0-9a-f]+>',r'<\1+OFFSET>',s)
        if got:
            def replace(m):
                value=int(m[1],16)
                return '##GOT:'+got[value] if value in got else m[0]
            # Restrict to extended offset memory accesses; this only screens
            # linked GOT changes, not general immediate arithmetic values.
            s=re.sub(r'##(-0x[0-9a-f]+)(?=\))',replace,s)
        s=s.replace('{','').replace('}','')
        s=' '.join(s.split())
        if s in ['nop','nop :endloop0','nop :endloop1']:continue
        result[name].append(s)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--tools-bin',type=Path,required=True);a=p.parse_args();D=a.directory;B=a.tools_bin
    sets={};gotmaps={}
    for kind in ['control','rebuilt','candidate']:
        elf=D/(kind+'-original.so') if kind!='candidate' else D/'candidate.so'
        syms=subprocess.check_output([str(B/'hexagon-llvm-readelf'),'-sW',str(elf)],text=True)
        rels=subprocess.check_output([str(B/'hexagon-llvm-readelf'),'-Wr',str(elf)],text=True)
        base=int(next(x.split()[1] for x in syms.splitlines() if '_GLOBAL_OFFSET_TABLE_' in x),16)
        got={}
        for line in rels.splitlines():
            f=line.split()
            if len(f)>=5 and f[2]=='R_HEX_GLOB_DAT':got[int(f[0],16)-base]=f[4]
        gotmaps[kind]={str(k):v for k,v in got.items()}
        sets[kind]=functions((D/(kind+'-original.dis')).read_text(),got)
    summary={}
    for side in ['rebuilt','candidate']:
        same=[];changes=[]
        for name in sorted(set(sets['control'])|set(sets[side])):
            x=sets['control'].get(name);y=sets[side].get(name)
            if x==y:same.append(name)
            else:
                changes.append({'symbol':name,'control_count':None if x is None else len(x),'other_count':None if y is None else len(y)})
                (D/(side+'-'+re.sub('[^a-zA-Z0-9_-]','_',name)+'-got-screen.diff')).write_text('\n'.join(difflib.unified_diff(x or [],y or [],n=2))+'\n')
        summary[side]={'same_screen_count':len(same),'changed_screen':changes}
    (D/'got-offsets.json').write_text(json.dumps(gotmaps,indent=2)+'\n')
    (D/'normalized-got-screen.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

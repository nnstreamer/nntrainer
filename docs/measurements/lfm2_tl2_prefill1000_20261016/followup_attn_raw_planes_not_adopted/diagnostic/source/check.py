#!/usr/bin/env python3
"""Enable the frozen diagnostic counters in the actual pipeline host check.
HMX/softmax/PV remain stand-ins. This tests counter placement, not DSP timing.
"""
import argparse, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--hexagon-tools',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
D=Path(__file__).resolve().parent.parent;a.output.mkdir(parents=True,exist_ok=True)
runner=(D/'candidate/tools/lfm2_attn_raw_planes_check.py').read_text()
runner=runner.replace('"-std=gnu11", "-O2"','"-DNNTR_LFM2_ATTN_RAW_PLANES_TRACE=1", "-std=gnu11", "-O2"')
runner=runner.replace('Path(__file__).resolve().parents[1] / "test/htp/host"','Path(__file__).resolve().parent / "fixtures"')
(a.output/'run.py').write_text(runner);f=a.output/'fixtures';f.mkdir(exist_ok=True)
(f/'attn_raw_planes_check.c').write_bytes((D/'candidate/test/htp/host/attn_raw_planes_check.c').read_bytes())
s=(D/'candidate/test/htp/host/attn_raw_planes_overlap_check.c').read_text()
needle='  assert(!pending && submits == joins && plane_index == 0u);'
assert needle in s
s=s.replace(needle,needle+'''
  const unsigned successful_reads = read_calls - (fail_read >= 0 && read_calls > (unsigned)fail_read);
  if (c.qk_planes) {
    assert(c.trace_sdk == 0u);
    assert(c.trace_raw_serial + c.trace_raw_deferred == successful_reads);
    assert(c.trace_raw_serial == (qx_dequant_available(&c) ? 64u : successful_reads));
  } else {
    assert(c.trace_raw_serial == 0u && c.trace_raw_deferred == 0u);
    assert(c.trace_sdk == successful_reads);
  }
''')
(f/'attn_raw_planes_overlap_check.c').write_text(s)
cmd=['python3',str(a.output/'run.py'),'--repo',str(a.repo),'--source',str(D/'diagnostic/source/nntrainer/tensor/htp_backend/hmx/hexkl_attn_q.c'),'--hexagon-tools',str(a.hexagon_tools),'--output',str(a.output/'run'),'--sanitize']
q=subprocess.run(cmd,capture_output=True,text=True,timeout=300);(a.output/'run.log').write_text(q.stdout+q.stderr)
(a.output/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');print(q.stdout,end='');print(q.stderr,end='');raise SystemExit(q.returncode)

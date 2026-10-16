#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parents[1]
c=json.loads((D/'control-device.json').read_text()); v=json.loads((D/'candidate-device.json').read_text())
phases=['body_ms','pack_ms','sums_ms','read_ms','join_ms','drain_ms','other_foreground_ms']
comparison={k:{'control':c['main'][k],'candidate':v['main'][k],'candidate_minus_control':v['main'][k]-c['main'][k]} for k in phases}
groups={k:{'calls':x['calls'],'control_body_ms':x['body_ms'],'candidate_body_ms':v['groups'][k]['body_ms'],'body_delta_ms':v['groups'][k]['body_ms']-x['body_ms'],'control_join_ms':x['join_ms'],'candidate_join_ms':v['groups'][k]['join_ms']} for k,x in c['groups'].items() if k in c['main_calls_by_shape']}
summary={
 'status':'PASS: strict source-derived 2048 cohorts and separate independent raw readback',
 'scope':'One diagnostic control/candidate pair; trace throughput excluded; parent owns device/model checks',
 'control':{k:c[k] for k in ['source_log_sha256','pids','records','unique_packets','raw_packets','transport_duplicate_packets','main_branch']},
 'candidate':{k:v[k] for k in ['source_log_sha256','pids','records','unique_packets','raw_packets','transport_duplicate_packets','main_branch']},
 'each_run':{'warmup_calls':100,'main_calls':190,'chronology':'first100 load warmups, then190 main calls','arena_bytes':8372224,'main_reads':296960,'main_joins':13390,'main_qtimer_reads_including_calibration':635200,'main_calls_by_shape':c['main_calls_by_shape']},
 'main_ms':comparison,'main_groups':groups,
 'load_warmup_body_ms':{'control':c['load_warmup']['body_ms'],'candidate':v['load_warmup']['body_ms']},
 'findings':[
  'Actual candidate fast branch processed every expected main tile; control used generic branch for every expected main tile.',
  'Instrumented main FC body decreased50.408ms; every source-derived geometry group decreased. The diagnostic does not show an FC-body slowdown.',
  'Other foreground decreased74.839ms, while read increased17.407ms and join increased6.784ms. This can reflect issue/completion overlap redistribution; it is not proof of a specific regression mechanism.',
  'Read includes completion of preceding HMX operations; other includes unbracketed issue/scalar/control work and trace-induced code changes. Neither is pure HMX or pure issue time.',
  'QKV N3072 join accounts for17.320ms control and21.996ms candidate of21.244/28.028ms total main join. These are exposed waits, not summed worker time.',
  'The uninstrumented12-run matrix reported by parent regresses1024PF840.033to827.141TPS and2048PF888.889to785.276TPS. This single heavily instrumented pair cannot attribute that end-to-end regression or establish a normal speedup.',
  'Do not subtract empty-timer estimates or sum overlapped service times. Preserve the rejected normal result separately from diagnostic evidence.'
 ],
 'performance_comparison_eligible':False,
 'no_production_edits':True,
 'artifacts':['control-device.json','candidate-device.json','root/raw-readback.json','root/phase_readback.py'],
}
(D/'root/phase-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['FC regression phase audit: diagnostic control vs candidate,2048 rows','',
 'Both strict parsers and an independent raw-field readback PASS. Each run contains870 unique G/T/B packets,290 call identities,zero transport duplicates and one PID. Chronology is100 load warmups followed by190 main kernels. Actual arena is8,372,224bytes.',
 'Main cohort: N6144 has30 each of1728/192/128 rows; N2048 has40 each of1920/128; N3072 has10 each of1728/320. The source-derived bank15 atM1920 is present.',
 'Actual main dispatch: control generic296,960/fast0; candidate generic0/fast296,960. Both have296,960 readouts and13,390 joins. Candidate is executed, not merely enabled.','',
 'Main foreground milliseconds (load excluded):','phase                 control    candidate       delta']
for k,r in comparison.items():lines.append(f'{k:23s} {r["control"]:10.3f} {r["candidate"]:12.3f} {r["candidate_minus_control"]:11.3f}')
lines+=['', 'All seven main geometry groups have a lower instrumented body time. QKV joins total17.320->21.996ms; load warmup body14.922->13.608ms is kept separate.', '',
 'Interpretation: the candidate body is50.408ms lower in this diagnostic pair, so these data do not identify an FC-body slowdown as the cause of the normal2048 regression. They show timing redistribution: read/HMX-completion and exposed join increase while unbracketed foreground declines. They do not prove return barriers, HMX serialization or a normal speedup.',
 'This single trace pair adds635,200 qtimer reads across the main cohort plus per-tile counters, altered codegen and post-body logs. Trace TPS is excluded. Read includes HMX completion; other includes issue/scalar/control work. No empty-timer correction is subtracted and no worker aggregate is treated as serial time.',
 'Parent normal12-run result remains the performance decision:1024PF840.033->827.141(-1.53%);2048PF888.889->785.276(-11.66%,0/3paired wins). IDs/settings/complete memory are root checks. This audit does not explain the end-to-end regression.','',
 'Files: phase-summary.json,raw-readback.json; ../control-device.json and ../candidate-device.json. Frozen diagnostic patches and parser are unchanged.']
(D/'root/phase-summary.txt').write_text('\n'.join(lines)+'\n')
readme=D/'README.txt'; text=readme.read_text(); index=text.index('Deferred feasibility note, NOT implemented')
text=text[:index]+'''Deferred feasibility note, NOT implemented; previous capacity statement corrected:
The nested conv split is real call/setup work, but a size-class key is NOT an
allocated capacity. fit() allocates roundup4096(requested_bytes)+4096. Existing
1920x6144 output needs45MiB+one page in the64MiB class; whole2048 needs48MiB+one
page in the SAME class, a real3MiB increase. The earlier wording that the class
already accommodated full2048 without growth was incorrect. Root explicitly
excludes the simple whole2048 retained-projection guard expansion under the
no-capacity-growth constraint. Aligning ARMstep1728 also changes the tail320
size-class demand and is not an approved capacity-preserving implementation.
No such production change is included. A separate read-only design audit may
consider consuming each existing1920+128 projection view before the next
backend call, using existing gated/output tensors and unchanged RPC chunking.
That requires explicit lifetime, alias, global-row/history and fallback proof;
no allocation or arithmetic equivalence is assumed from class names.

Device diagnostic audit (root/phase-summary.json and .txt):
Both actual2048 logs pass strict290-call/870-packet topology with100 load then
190 main calls. No duplicate packets. Main actual branches are296960 generic
control versus296960 fast candidate. Instrumented body267.168->216.760ms;
read56.320->73.727ms,join21.244->28.028ms,other124.436->49.597ms. This is phase
redistribution with a lower instrumented body, not an explanation of the normal
end-to-end regression. Trace TPS and empty-timer subtraction remain excluded.
Independent raw-field readback imports neither AFC2 parser and agrees exactly.
'''
readme.write_text(text)
validation=json.loads((D/'validation.json').read_text())
validation['status']='Frozen diagnostic: host/topology/parser/v81 and both root device phase cohorts PASS; normal regression remains unexplained'
validation['device_phase_audit']='root/phase-summary.json'
validation['independent_raw_readback']='root/raw-readback.json'
validation['deferred_conv_whole_span_prototype']='Simple whole2048 guard expansion rejected: fit actual output capacity rises45MiB+page to48MiB+page despite same64MiB class. Chunk-consumption alternative remains read-only design audit.'
(D/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
manifest=json.loads((D/'manifest.json').read_text())
for rel in ['control-device.json','candidate-device.json','root/phase-summary.json','root/phase-summary.txt','root/raw-readback.json','root/phase_readback.py','root/freeze_phase_summary.py']:
 manifest[rel]=''
for rel in manifest:manifest[rel]=hashlib.sha256((D/rel).read_bytes()).hexdigest()
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for rel in ['README.txt','validation.json','manifest.json','root/phase-summary.json','root/phase-summary.txt']:
 print(rel,hashlib.sha256((D/rel).read_bytes()).hexdigest())

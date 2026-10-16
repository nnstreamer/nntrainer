#!/usr/bin/env python3
"""Validate and archive the rejected Tile16/Tile8 experiment; no device access."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'author', 'independent', 'scratch', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    D, I, T = a.author, a.independent, a.scratch
    V, J = D / 'variant8', I / 'variant8'
    E = a.repo / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
    A = Path(__file__).resolve().parent
    pending = {}
    compress_types = {'.log', '.cpp', '.c', '.h', '.inc', '.patch', '.su', '.asm'}

    def add(source, target, compress=None):
        require(source.is_file(), 'Missing input: ' + str(source))
        if compress is None:
            compress = source.suffix in compress_types or source.name.endswith('.disasm.txt') or source.stat().st_size > 128 * 1024
        target += '.gz' if compress else ''
        require(target not in pending, 'Duplicate archive target: ' + target)
        pending[target] = (source.read_bytes(), str(source), compress)

    def generated(target, value):
        require(target not in pending, target)
        pending[target] = ((json.dumps(value, indent=2)+'\n').encode(), 'generated', False)

    def tree(root, target, extensions):
        require(root.is_dir(), str(root))
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in extensions:
                add(path, target + '/' + str(path.relative_to(root)))

    # Bind source, patches, fixtures, native binaries and root integration.
    for root in (D, V):
        frozen = read(root / 'frozen.json')
        for relative, expected in frozen['files'].items():
            require(sha(root / relative) == expected, 'Frozen author input changed: ' + str(root / relative))
    final = read(J / 'validation.json')
    require(final['independently_executed']['actual_source_asan_ubsan_on'] == 112 and
            final['independently_executed']['same_binary_off'] == 112 and
            final['independently_executed']['mutants_intended_runtime_rejected'] == 8,
            'Independent final verification incomplete')
    for relative, expected in final['frozen'].items():
        candidates = [J/'frozen'/relative, V/relative, D/relative]
        source = next((q for q in candidates if q.is_file() and sha(q) == expected), None)
        require(source is not None, 'Independent snapshot missing or changed: ' + relative)
    for relative, expected in final['artifacts'].items():
        require(sha(J/relative) == expected, 'Independent result changed: ' + relative)

    names = ['lfm-conv-chunk-isolated', 'lfm-conv-gate-tile8-off',
             'lfm-conv-gate-tile8-on', 'lfm-conv-gate-tile8-reverted']
    variants = {}
    for name in names:
        value = read(E / (name + '-artifacts.json'))
        require(len(value['binaries']) == 7, name + ': expected seven runtime hashes')
        require(all(re.fullmatch('[0-9a-f]{64}', v) for v in value['binaries'].values()), name)
        require(sha(T / (name + '.source.patch')) == value['source_patch_sha256'], name + ': patch mismatch')
        variants[name] = value
        add(E / (name + '-artifacts.json'), 'artifacts/' + name + '.json')
        add(T / (name + '.source.patch'), 'artifacts/' + name + '.source.patch')
    control, candidate = names[1:3]
    require(variants[control]['binaries'] == variants[candidate]['binaries'], 'OFF/ON runtime binaries differ')
    require(variants[control]['source_patch_sha256'] == variants[candidate]['source_patch_sha256'], 'OFF/ON source differs')
    require(variants[names[0]]['binaries'] == variants[names[3]]['binaries'], 'Restored runtime differs')
    require({k:v for k,v in variants[control]['binaries'].items() if k != 'libcausallm.so'} ==
            {k:v for k,v in variants[names[0]]['binaries'].items() if k != 'libcausallm.so'}, 'Unrelated runtime changed')
    integrated = read(V / 'root-frozen/source.json')
    require(integrated['source_sha256'] == sha(V/'candidate/Applications/CausalLM/layers/conv_block_layer.cpp'), 'Root source differs')
    require(integrated['author_patch_sha256'] == sha(V/'production.patch'), 'Root author patch differs')
    require(integrated['applied_patch_sha256'] == sha(V/'root-frozen/applied.patch') == variants[control]['source_patch_sha256'], 'Integrated patch differs')
    for name, expected in integrated['app_binaries_sha256'].items():
        require(expected == variants[control]['binaries'][name] == sha(V/'root-frozen'/name), 'Frozen app differs: ' + name)
    require(integrated['normal_dsp_sha256'] == variants[control]['binaries']['libnntr_hvx_skel.so'], 'DSP differs')
    restored = read(V / 'root-normal-restored.json')
    require(restored['valid'] and restored['decision'] == 'not_adopted' and restored['rebuilt_app_byte_identical_to_ccea_runtime'] and
            restored['all_seven_deployed_runtime_binaries_identical_to_ccea_runtime'], 'Restore incomplete')
    require(restored['restored_runtime'] == variants[names[3]], 'Restored deployment manifest differs')
    require(restored['app_rebuild_log_sha256'] == sha(V/'root-revert-build.log'), 'Rebuild log changed')
    require(restored['production_source_sha256'] == sha(D/'before/Applications/CausalLM/layers/conv_block_layer.cpp'), 'Restored source differs')
    # Apply both author and integrated patches only in a new temporary overlay.
    relative = Path('Applications/CausalLM/layers/conv_block_layer.cpp')
    base = subprocess.check_output(['git', '-C', str(a.repo), 'show', integrated['base_commit'] + ':' + str(relative)])
    require(hashlib.sha256(base).hexdigest() == restored['production_source_sha256'], 'Commit base mismatch')
    for patch in (V/'production.patch', V/'root-frozen/applied.patch'):
        with tempfile.TemporaryDirectory(prefix='patch-proof-', dir=A) as temporary:
            file = Path(temporary) / relative
            file.parent.mkdir(parents=True)
            file.write_bytes(base)
            subprocess.run(['git', 'apply', str(patch.resolve())], cwd=temporary, check=True, capture_output=True)
            require(sha(file) == integrated['source_sha256'], 'Patch does not reproduce candidate: ' + str(patch))
    for path in (V/'root-rejected-canonical').iterdir():
        source = V/'candidate'/relative if path.name == 'conv_block_layer.cpp' else V/'test/htp/host'/path.name
        require(sha(path) == sha(source), 'Removed canonical source differs')

    env = {'NNTR_NUM_THREADS':'8', 'NNTR_HTP_PAGE_CACHE':'0', 'NNTR_LOAD_PREFETCH':'0',
           'NNTR_MOE_CACHE_GROW':'0', 'NNTR_MOE_PRELOAD_EXPERTS':'128', 'NNTR_INIT_TRACE':'1',
           'NNTR_HTP_PREDICT':'2', 'NNTR_MOE_PREFETCH_RESIDUAL':'0.5', 'NNTR_HTP_TL2_STABLE_SLOTS':'1',
           'NNTR_LFM2_PREFILL_EARLY':'1', 'NNTR_LFM2_DECODE_PAIR_READ':'1',
           'NNTR_LFM2_DECODE_DEMAND_HANDOFF':'1', 'NNTR_LFM2_ROUTER_INPUT_HANDOFF':'1'}
    expected_ids = {1024:'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',
                    2048:'2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33'}
    jobs = read(V/'root-normal-jobs.json')
    require(len(jobs) == 12 and len({j['tag'] for j in jobs}) == 12, 'Expected12 unique normal runs')
    require([j['variant'] for j in jobs] == [control,candidate,candidate,control,control,candidate]*2, 'Run ordering differs')
    results, raw_rows = [], []
    for job in jobs:
        tag = job['tag']
        r, m = read(E/(tag+'.json')), read(E/(tag+'.memory.json'))
        n = r['prefill_tokens']
        require(n in expected_ids and r['tag'] == tag and r['variant'] == job['variant'] and r['model'] == job['model'], tag+': identity')
        require(r['valid'] and r['performance_comparison_eligible'] and not r['instrumented'] and not r['profile'] and
                r['returncode'] == 0 and not r['concurrent_processes'], tag+': invalid run')
        require(r['moe_cache_experts'] == 16 and r['decode_tokens'] == 64, tag+': cache/token settings')
        for key, value in env.items():
            require(r['environment'].get(key) == value, tag+': environment '+key)
        for key, value in job['env'].items():
            require(r['environment'].get(key) == str(value), tag+': job environment '+key)
        require(r['environment']['NNTR_LFM2_CONV_GATE_TILE'] == ('0' if job['variant'] == control else '1'), tag+': flag')
        require(set(r['environment']) == set(env)|{'LD_LIBRARY_PATH','ADSP_LIBRARY_PATH','NNTR_GENERATED_IDS','NNTR_LFM2_CONV_GATE_TILE'}, tag+': unexpected environment')
        require(r['ids_match_reference'] and r['ids_sha256'] == sha(E/(tag+'.ids')) == expected_ids[n] and
                len((E/(tag+'.ids')).read_text().splitlines()) == 65, tag+': IDs')
        config = E/'repro/models'/('lfm2-'+str(n))/'nntr_config.json'
        require(r['model_config_sha256'] == sha(config) and read(config)['moe_cache_experts'] == 16, tag+': config')
        peak = m['peak_dedup_sample']
        require(m['child_waited'] and m['child_exit_code'] == 0 and m['requested_period_ms'] == 20 and
                peak['dedup_complete_for_readable_size_fds'] and peak['composite_available'], tag+': monitor')
        require(not any(peak[k] for k in ('fdinfo_read_errors','unknown_identity_fds','inode_size_conflicts','invalid_fields','counter_overflow')), tag+': incomplete peak')
        require(r['full_run_memory_report_sha256'] == sha(E/(tag+'.memory.json')) and
                r['full_run_sampled_peak_kib']*1024 == m['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes'], tag+': memory hash/value')
        text = (E/(tag+'.log')).read_text()
        for label, prefix in [('prefill','prefill'),('generation','decode')]:
            values = re.findall(label+r': (\d+) tokens, ([\d.]+) ms, ([\d.]+) TPS', text)
            require(len(values) == 1, tag+': throughput occurrence')
            require(tuple(map(float,values[0])) == (r[prefix+'_tokens'],r[prefix+'_ms'],r[prefix+'_tps']), tag+': log throughput')
        require('[HTP-PREFETCH] direct slots=32 bytes=64618496 readers=2 pair_read=1' in text, tag+': direct read pool')
        results.append(r)
        row = {k:r[k] for k in ('tag','variant','prefill_tokens','prefill_tps','decode_tps','full_run_sampled_peak_kib','ids_sha256')}
        row['repetition'] = int(re.search(r'-r(\d+)$',tag)[1]); raw_rows.append(row)
        for suffix in ('.json','.ids','.log','.memory.json'):
            add(E/(tag+suffix), 'raw/'+tag+suffix, suffix in ('.log','.memory.json'))
    computed = []
    for n in (1024,2048):
        groups = {v:sorted([r for r in raw_rows if r['prefill_tokens']==n and r['variant']==v],key=lambda r:r['repetition']) for v in (control,candidate)}
        require(all([r['repetition'] for r in rows]==[1,2,3] for rows in groups.values()), 'Unbalanced group')
        row = {'tokens':n}
        for metric in ('prefill_tps','decode_tps'):
            x,y = [[r[metric] for r in groups[v]] for v in (control,candidate)]
            am,bm = statistics.median(x),statistics.median(y)
            row[metric] = {'control':x,'candidate':y,'median_control':am,'median_candidate':bm,'median_pct':100*(bm/am-1),
                           'pairs_faster':sum(b>a for a,b in zip(x,y)),'paired_deltas_pct':[100*(b/a-1) for a,b in zip(x,y)]}
        for label,v in [('control',control),('candidate',candidate)]:
            row[label+'_max_peak_kib'] = max(r['full_run_sampled_peak_kib'] for r in groups[v])
        computed.append(row)
    summary = read(V/'root-normal-summary.json')
    require(computed == summary['summary'] and raw_rows == summary['runs'] and summary['all_checks_pass'], 'Summary not reproduced')

    # Bind native correctness/micro records without copying executable bytes.
    for root, native_names in [(D,['root-native-arm','root-native-v2','root-ion-micro']), (V,['root-native-arm','root-ion-micro'])]:
        for name in native_names:
            result = read(root/(name+'.json'))
            require(result['returncode']==0 and result['log_sha256']==sha(root/(name+'.log')), 'Native log differs: '+str(root/name))
        micro = read(root/'root-ion-micro.json')
        require(micro['runtime_manifest_sha256']==sha(E/'lfm-conv-chunk-isolated-artifacts.json'), 'Micro runtime differs')
        require(sha(root/'lfm2_conv_gate_tile_ion_check')==micro['binary_sha256'], 'Micro binary differs')
    require(final['parent_executed_ion_micro']['raw_sha256']==sha(V/'root-ion-micro.log'), 'Independent micro differs')
    require(final['parent_executed_native']==read(V/'root-native-arm.json'), 'Independent native differs')

    # Canonical sources once; generated mutant sources are reproducible from the
    # mutation runner and command/result logs, avoiding repeated large payloads.
    for folder in ('baseline','before'):
        tree(D/folder,'validation/'+folder,{'.cpp'})
    for label,root in [('tile16',D),('tile8',V)]:
        prefix='validation/'+label+'/author/'
        for name in ('frozen.json','production.patch','ion-build-command.json'):
            add(root/name,prefix+name)
        for name in ('tests.patch','validation.json','README.txt'):
            if (root/name).exists():add(root/name,prefix+name)
        tree(root/'candidate',prefix+'candidate',{'.cpp','.inc'})
        for path in (root/'test/htp/host').glob('*'):
            if path.is_file() and path.suffix in {'.py','.cpp'} and (label=='tile16' or 'conv_gate_tile' in path.name):
                add(path,prefix+'test/htp/host/'+path.name)
        for name in ('host-v2-final-san.log','host-v2-final-off.log','native-v2-build.log','mutations.log',
                     'final-san.log','final-off.log','existing-chunk-san.log','existing-chunk-off.log','README.txt'):
            if (root/name).exists() and not (name=='README.txt' and prefix+name in pending):
                add(root/name,prefix+name)
        for path in root.glob('root-*'):
            if path.is_file() and path.suffix in {'.log','.json'}:add(path,prefix+path.name)
        check_folders = ['checks/mutants','checks/non-arm']
        check_folders += ['checks/v2-final-san','checks/v2-native-arm','checks/native-arm'] if label=='tile16' else ['checks/final-san','checks/native-arm','checks/existing-chunk-san','root-integrated-san','root-frozen']
        for folder in check_folders:
            if (root/folder).exists():tree(root/folder,prefix+folder,{'.json','.log','.patch'})
        # Keep small assembly slices and frame summaries, not whole disassemblies.
        for path in (root/'codegen').rglob('*'):
            if path.is_file() and path.suffix in {'.json','.log','.su','.txt'} and path.stat().st_size<128*1024:
                add(path,prefix+'codegen/'+str(path.relative_to(root/'codegen')))
        if label=='tile16':
            tree(root/'checks/native-arm',prefix+'preliminary-native-extracts',{'.cpp','.inc'})
    tree(V/'root-rejected-canonical','validation/root-removed-canonical',{'.cpp','.py'})
    for label,root in [('tile16',I),('tile8',J)]:
        prefix='validation/'+label+'/independent/'
        for path in root.iterdir():
            if path.is_file() and path.suffix in {'.json','.log','.py','.txt','.su'} and path.stat().st_size<128*1024:
                add(path,prefix+path.name)
        for folder in ('host-san','native-arm','mutants-san'):
            if (root/folder).exists():tree(root/folder,prefix+folder,{'.json','.log'})
        # Retain the independent pre-status-update snapshot that its hash names.
        if (root/'frozen/frozen.json').exists():add(root/'frozen/frozen.json',prefix+'frozen-status-at-review.json')
    for n in (1024,2048):
        tree(E/'repro/models'/('lfm2-'+str(n)), 'configuration/lfm2-'+str(n), {'.json'})
    for name in ('bench.py','device.json'):
        add(E/name,'reproduction/'+name)
    add(E/'validation/memory_monitor.cpp','reproduction/memory_monitor.cpp')
    for name in ('run_matrix.py','summarize_pair.py'):
        add(T/name,'reproduction/original-'+name)
    for name in ('curate_conv_gate_tile.py','verify_archive.py','reproduce_host.py'):
        add(A/name,'reproduction/'+name)
    add(A/'README.txt','README.txt',False)
    generated('summary.json', {'decision':'not_adopted','normal':computed,'all_65_ids_exact':True,
        'same_seven_binaries_off_on':variants[control]['binaries'], 'restored_seven_binaries':variants[names[0]]['binaries'],
        'normal_run_count':12,'cpu_only_micro':final['parent_executed_ion_micro'],
        'micro_not_model_speed_evidence':True,'reason':'Both normal prefill medians regressed; candidate removed.'})
    generated('validation/archive-checks.json', {'normal_runs_validated':12,'all_raw_memory_hashes_exact':True,
        '65_ids_each':True,'fixed_c16_page0_preload128':True,'direct_reader_marker_each':True,
        'same_seven_binaries_off_on':True,'author_and_integrated_patch_replayed':True,
        'restored_record_validated':True,'live_root_clean_not_required':True,'no_device_actions':True,
        'source':integrated['source_sha256'],'author_patch':integrated['author_patch_sha256'],
        'applied_patch':integrated['applied_patch_sha256'],'normal_summary_recomputed':True})
    total = sum(len(gzip.compress(data,mtime=0) if compress else data) for data,_,compress in pending.values())
    print('Validated12 normal runs;',len(pending),'payloads;',total,'stored bytes')
    if a.check_only:return
    require(not a.output.exists(),'Refuse to overwrite an existing archive')
    manifest={'decision':'not_adopted','source_base_commit':integrated['base_commit'],
              'reproduction':{'core_sha256':read(V/'checks/final-san/commands.json')['core_sha256']},'files':{}}
    for relative,(data,source,compress) in sorted(pending.items()):
        stored=gzip.compress(data,mtime=0) if compress else data
        dest=a.output/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(stored)
        manifest['files'][relative]={'sha256':hashlib.sha256(stored).hexdigest(),
            'uncompressed_sha256':hashlib.sha256(data).hexdigest(),'gzip':compress,
            'source':source,'stored_bytes':len(stored),'uncompressed_bytes':len(data)}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Manifest SHA256',sha(a.output/'manifest.json'))


if __name__ == '__main__':
    main()

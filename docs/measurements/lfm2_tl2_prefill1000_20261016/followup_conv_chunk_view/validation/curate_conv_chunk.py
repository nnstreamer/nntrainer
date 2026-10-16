#!/usr/bin/env python3
"""Collect frozen convolution trials and exact raw results, without binaries."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    evidence = Path(spec['evidence'])
    pending = {}
    text_suffixes = {'.txt', '.json', '.log', '.cpp', '.c', '.h', '.py',
                     '.patch', '.diff', '.s', '.su', '.dis'}
    compress_suffixes = {'.log', '.cpp', '.c', '.h', '.patch', '.diff',
                         '.s', '.su', '.dis'}

    def add(source, target, compress=None):
        source = Path(source)
        assert source.is_file(), source
        if compress is None:
            compress = (source.suffix in compress_suffixes or
                        source.stat().st_size > 128 * 1024)
        if compress:
            target += '.gz'
        assert target not in pending, target
        data = source.read_bytes()
        pending[target] = (gzip.compress(data, mtime=0) if compress else data,
                           str(source), sha(source))

    for item in spec['files']:
        add(item['source'], item['target'], item.get('compress'))
    for item in spec['trees']:
        root = Path(item['source'])
        assert root.is_dir(), root
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in text_suffixes:
                add(path, item['target'] + '/' + str(path.relative_to(root)))

    config_hashes = {}
    validated_runs = []
    for job in spec['runs']:
        tag = job['tag']
        result = read(evidence / (tag + '.json'))
        memory_path = evidence / (tag + '.memory.json')
        memory = read(memory_path)
        peak = memory['peak_dedup_sample']
        assert result['tag'] == tag and result['variant'] == job['variant'], tag
        assert result['valid'] and result['returncode'] == 0, tag
        assert not result['concurrent_processes'], tag
        instrumented = job.get('profile', False)
        assert result['profile'] == result['instrumented'] == instrumented, tag
        assert result['performance_comparison_eligible'] != instrumented, tag
        n = result['prefill_tokens']
        assert result['moe_cache_experts'] == 16 and result['decode_tokens'] == 64
        for key, value in spec['environment'].items():
            assert result['environment'][key] == str(value), (tag, key)
        assert result['ids_match_reference'], tag
        ids_path = evidence / (tag + '.ids')
        assert len(ids_path.read_text().splitlines()) == 65, tag
        assert sha(ids_path) == result['ids_sha256'] == spec['ids_sha256'][str(n)]
        config_hashes.setdefault(n, set()).add(result['model_config_sha256'])
        assert memory['child_waited'] and memory['child_exit_code'] == 0, tag
        assert memory['requested_period_ms'] == 20, tag
        assert peak['dedup_complete_for_readable_size_fds'], tag
        assert peak['composite_available'], tag
        assert not any(peak[key] for key in ('fdinfo_read_errors',
            'unknown_identity_fds', 'inode_size_conflicts', 'invalid_fields',
            'counter_overflow')), tag
        assert result['full_run_memory_report_sha256'] == sha(memory_path), tag
        assert result['full_run_sampled_peak_kib'] * 1024 == memory[
            'peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes'], tag
        for suffix in ('.json', '.ids', '.log', '.memory.json'):
            add(evidence / (tag + suffix), 'raw/' + tag + suffix,
                suffix in ('.log', '.memory.json'))
        if instrumented:
            add(evidence / (tag + '.logcat.txt'), 'raw/' + tag + '.logcat.txt', True)
        validated_runs.append(tag)
    assert all(len(values) == 1 for values in config_hashes.values())
    add(args.spec, 'validation/curation-spec.json')
    add(Path(__file__), 'validation/curate_conv_chunk.py')
    assert not args.output.exists(), 'Refuse to overwrite an existing archive'
    manifest = {'validated_runs': validated_runs, 'files': {}}
    for target, (data, source, original_sha) in sorted(pending.items()):
        dest = args.output / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        manifest['files'][target] = {'sha256': sha(dest), 'source': source,
                                     'uncompressed_sha256': original_sha}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for target, item in manifest['files'].items():
        path = args.output / target
        assert sha(path) == item['sha256']
        content = gzip.decompress(path.read_bytes()) if target.endswith('.gz') else path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == item['uncompressed_sha256']
    print(f'Validated {len(validated_runs)} runs and {len(pending)} archived files.')


if __name__ == '__main__':
    main()

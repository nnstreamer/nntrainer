#!/usr/bin/env python3
"""Additional bounded parser and actual extracted drain/sequence checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('parser', D / 'parse_trace.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)

def line(seq, e, lanes=8, pid=100, tid=101):
    counts = (e, e * 48, 0, 0) if e >= 4 else (0, 0, e * lanes, e * 48)
    return (f'10-16 22:00:00.000 {pid} {tid} V adsprpc : {parser.TAG} '
            f'seq={seq} M=1 K=2048 I=1536 N=2048 E={e} native={e} '
            f'lanes={lanes} fc={counts[0]} ft={counts[1]} '
            f'uc={counts[2]} ut={counts[3]} !')

a, b, c = line(1, 4), line(2, 1), line(3, 3)
good = '\n'.join([a, b, c])
valid = {}
for name, text in {
    'ordered': good,
    'reordered': '\n'.join([c, a, b]),
    'transport-replay': '\n'.join([a, b, a, c, b, c]),
    'different-transport-tid': '\n'.join([a, b, c, line(2, 1, tid=999)]),
}.items():
    result = parser.validate(text, 1, 2, 4)
    assert result['unique_kernel_calls'] == 3
    assert result['active_experts'] == 8 and result['tile_pairs'] == 384
    valid[name] = result
assert not valid['reordered']['first_seen_sequence_order_monotonic']
assert valid['transport-replay']['transport_duplicate_records'] == 3

bad = {
    'sequence-zero': good.replace('seq=1', 'seq=0', 1),
    'sequence-overflow': good.replace('seq=1', 'seq=4294967296', 1),
    'sparse-max-u32': good.replace('seq=3', 'seq=4294967295', 1),
    'negative-sequence': good.replace('seq=1', 'seq=-1', 1),
    'zero-lanes': good.replace('lanes=8', 'lanes=0', 1),
    'excess-lanes': good.replace('lanes=8', 'lanes=49', 1),
    'zero-active': good.replace('E=4', 'E=0', 1),
    'excess-active': good.replace('E=4', 'E=5', 1),
    'zero-native': good.replace('native=4', 'native=0', 1),
    'negative-counter': good.replace('fc=4', 'fc=-4', 1),
    'wrong-counter': good.replace('ft=192', 'ft=191', 1),
    'unknown-key': good.replace('seq=1', 'seq=1 other=2', 1),
    'duplicate-key': good.replace('seq=1', 'seq=1 seq=1', 1),
    'missing-key': good.replace('K=2048 ', '', 1),
    'missing-terminal': good[:-2],
    'trailing-payload': good + ' bad',
    'missing-prefix': parser.TAG + a.split(parser.TAG, 1)[1],
    'missing-first': '\n'.join([b, c]),
    'missing-middle': '\n'.join([a, c]),
    'missing-last': '\n'.join([a, b]),
    'conflicting-replay': good + '\n' + line(2, 1, lanes=7),
    'second-pid-replay': good + '\n' + line(2, 1, pid=200),
}
rejected = {}
for name, text in bad.items():
    try:
        parser.validate(text, 1, 2, 4)
    except ValueError as error:
        rejected[name] = str(error)
    else:
        raise AssertionError('accepted malformed trace: ' + name)

# Each truncated suffix after the complete tag must fail, even if a valid copy
# of that sequence is already present. It must not be silently deduplicated.
truncated = 0
begin = b.index(parser.TAG) + len(parser.TAG)
for end in range(begin, len(b)):
    try:
        parser.validate(good + '\n' + b[:end], 1, 2, 4)
    except ValueError:
        truncated += 1
    else:
        raise AssertionError('accepted truncated duplicate at ' + str(end))

source = (D / 'host/drain-check.c').read_text()
changes = {
    'static uint32_t counter[4], expected[4], tl2_gu_trace_seq;':
      'static uint32_t counter[4], expected[4], tl2_gu_trace_seq, expected_seq;',
    'char line[128];': 'char line[256];',
    'assert(seq==1&&': 'assert(seq==++expected_seq&&',
    ' tl2_gu_trace_seq=0;\n': '',
    '  assert(logs==(variant==0||variant==8));':
      '  assert(tl2_gu_trace_seq==expected_seq);\n'
      '  assert(logs==(variant==0||variant==8));',
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new, 1)
anchor = ' printf("PASS: %u actual out-join/trace combinations;'
assert source.count(anchor) == 1
source = source.replace(anchor, r'''
 assert(expected_seq==36u && tl2_gu_trace_seq==36u);
 tl2_gu_trace_seq=expected_seq=UINT32_MAX-1u;
 logs=0;workers=7;
 assert(actual_out(0,0,1,1,2048,1536,2048,4,4)==0);
 assert(tl2_gu_trace_seq==UINT32_MAX&&logs==1);
 assert(actual_out(0,79,1,1,2048,1536,2048,4,4)==79);
 assert(tl2_gu_trace_seq==UINT32_MAX&&logs==1);
 assert(actual_out(0,0,1,1,2048,1536,2048,4,4)==0);
 assert(tl2_gu_trace_seq==0&&expected_seq==0&&logs==2);
 char worst[256];
 int bytes=snprintf(worst,sizeof worst,
  "[nntr-tl2-gu-direct] seq=%u M=%u K=%u I=%u N=%u E=%u native=%u lanes=%u fc=%u ft=%u uc=%u ut=%u !",
  UINT32_MAX,1u,2048u,1536u,2048u,16u,16u,48u,16u,768u,144u,144u);
 assert(bytes>0&&bytes<160&&bytes+64+1<=256&&worst[bytes-1]=='!');
 printf("Independent sequence lifecycle PASS: success-only monotonic36, error-no-advance, u32wrap0, bounded-model-payload%d bytes\n",bytes);
 printf("PASS: %u actual out-join/trace combinations;''', 1)
out = D / 'sequence-lifecycle'
out.mkdir(exist_ok=True)
check = out / 'check.c'
formatter = '/tmp/lfm2-prefill1000-20261013/format/clang_format/data/bin/clang-format'
formatted = subprocess.run([formatter, '--style=file',
    '--assume-filename=/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000/test/htp/host/trace_check.c'],
    input=source, text=True, stdout=subprocess.PIPE, check=True).stdout
check.write_text(formatted)
commands = [
    ['gcc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
     '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
     '-fno-omit-frame-pointer', '-fno-pie', '-no-pie', str(check),
     '-o', str(out / 'check')],
    [str(out / 'check')],
]
for name, command in zip(('build', 'run'), commands):
    with (out / (name + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
(out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
result = {'valid_delivery_cases': valid, 'rejected_cases': rejected,
          'truncated_duplicate_rejections': truncated,
          'sequence_lifecycle': (out / 'run.log').read_text().strip(),
          'source_sha256': hashlib.sha256(check.read_bytes()).hexdigest(),
          'parser_sha256': hashlib.sha256((D / 'parse_trace.py').read_bytes()).hexdigest(),
          'scope': 'Parser/actual extracted drain+trace block; no DMA or timing proof'}
(D / 'independent-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'Independent parser: {len(valid)} valid, {len(rejected)} malformed, '
      f'{truncated} truncated duplicate cases PASS')
print(result['sequence_lifecycle'])

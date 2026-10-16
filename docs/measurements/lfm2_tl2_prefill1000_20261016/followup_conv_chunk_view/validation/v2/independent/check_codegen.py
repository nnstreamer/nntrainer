#!/usr/bin/env python3
"""Compare the actual whole-TU AArch64 callbacks; no cycle estimate."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = Path('/tmp/nntr-lfm2-conv-chunk-view-review-20261016')


def selected(text, prefix):
    parts = re.split(r'(?m)^(?=[0-9a-f]+ <)', text)[1:]
    matches = [p for p in parts if p.splitlines()[0].split(' <', 1)[1].startswith(prefix)]
    assert len(matches) == 1, [p.splitlines()[0] for p in matches]
    return matches[0]


def words(part):
    return re.findall(r'^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s', part, re.M)


report = {}
for which in ['before', 'candidate']:
    text = (HERE / (which + '-android.dis')).read_text()
    callback = selected(text, 'causallm::lfmConvPostGatePrefill(')
    app = selected(text, 'causallm::ConvBlockLayer::incremental_forwarding(')
    (HERE / (which + '-full-callback.dis')).write_text(callback)
    report[which] = {
        'callback_bytes': len(words(callback)) * 4,
        'callback_words': words(callback),
        'app_instructions': len(words(app)),
        'app_first9': words(app)[:9],
    }
    stack = (OLD / 'before-stack.su' if which == 'before'
             else HERE / 'candidate-stack.su').read_text()
    report[which]['stack_selected'] = [
        line for line in stack.splitlines()
        if ('incremental_forwardingERN9nntrainer15RunLayerContextEjjb\t' in line
            or 'lfmConvChunkedPrefill' in line
            or 'lfmConvPostGatePrefill' in line)
        and 'function.h:' not in line
    ]
assert report['before']['callback_words'] == report['candidate']['callback_words']
assert report['before']['app_first9'] == report['candidate']['app_first9']
assert report['before']['callback_bytes'] == 236
report['callbacks_and_prologue_equal'] = True
report['callback_words_sha256'] = hashlib.sha256(
    ''.join(report['before']['callback_words']).encode()).hexdigest()
report['limits'] = ('The whole incremental_forwarding method is not identical; '
                    'callback bytes and first9 instructions are exact. No time/cycle claim.')
(HERE / 'codegen.json').write_text(json.dumps(report, indent=2) + '\n')
for which in ['before', 'candidate']:
    print(which, {k: v for k, v in report[which].items() if k != 'callback_words'})
print('Callback and first9 app instruction bytes EXACT')

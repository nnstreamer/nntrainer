#!/usr/bin/env python3
"""Validate LFM router handoffs by ticket; read-only log/metadata processing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


class InvalidTrace(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidTrace(message)


TAGS = {
    'handoff': 'rows', 'handoff-producer': 'producer_rows',
    'handoff-state': 'state_rows', 'handoff-publish': 'publish_rows',
}


def parse_text(text, label='<text>'):
    groups = {key: [] for key in TAGS.values()}
    for lineno, line in enumerate(text.splitlines(), 1):
        match = re.search(r'\[nntr-router-(handoff(?:-producer|-state|-publish)?)\]', line)
        if not match:
            continue
        tag = match[1]
        fields = re.findall(r'(\w+)=([^\s]+)', line[match.end():])
        require(len(fields) == len(dict(fields)), f'line {lineno}: duplicate field')
        row = {}
        for key, value in fields:
            pointer = tag in ('handoff-state', 'handoff-publish') and key in (
                'owner', 'base', 'source', 'input')
            row[key] = value if pointer or key == 'source_scope' else int(value)
        row['_line'] = lineno
        groups[TAGS[tag]].append(row)
    rows = groups['rows']
    result = {
        'log': str(label), 'calls': len(rows), 'producers': len(groups['producer_rows']),
        'requested': sum(x['request'] for x in rows),
        'stamp_matches': sum(x['owner'] for x in rows),
        'input_copies_skipped': sum(x['used'] for x in rows),
        'bytes_skipped': sum(x['bytes'] * x['used'] for x in rows),
        'requested_fallbacks': [x for x in rows if x['fallback']],
        'unticketed_calls': [i for i, x in enumerate(rows) if not x['request']],
        'state_masks': dict(Counter(str(x['match']) for x in groups['state_rows']
                                    if x['request_ticket'])),
    }
    result.update(groups)
    return result


def address(value):
    return 0 if value in ('(nil)', 'NULL') else int(value, 16)


def unique(rows, field, category):
    result = {}
    for row in rows:
        key = row[field]
        require(0 < key <= 0x3fffffff, f'{category}: invalid ticket {key}')
        require(key not in result, f'{category}: duplicate ticket {key}')
        result[key] = row
    return result


def validate(result, capacity_table, expected_layers=38):
    rows = result['rows']
    consumers = [x for x in rows if x['request']]
    requested_states = [x for x in result['state_rows'] if x['request_ticket']]
    extra_states = [x for x in result['state_rows'] if not x['request_ticket']]
    maps = {
        'consumer': unique(consumers, 'ticket', 'consumer'),
        'producer': unique(result['producer_rows'], 'ticket', 'producer'),
        'publish': unique(result['publish_rows'], 'sent', 'publish'),
        'requested state': unique(requested_states, 'request_ticket', 'requested state'),
    }
    tickets = set(maps['consumer'])
    for name, mapping in maps.items():
        require(len(mapping) == expected_layers,
                f'{name}: expected {expected_layers} distinct tickets, got {len(mapping)}')
        require(set(mapping) == tickets, f'{name}: ticket bijection mismatch')
    warmups = [x for x in rows if not x['request']]
    require(len(warmups) == 1 and len(rows) == expected_layers + 1,
            'expected one native-MoE load warmup plus the requested layers')
    warmup = warmups[0]
    M = consumers[0]['M']
    require(str(M) in capacity_table['shape'], f'no original v81 capacity proof for M={M}')
    caps = capacity_table['shape'][str(M)]
    possible_growth = {}
    for experts, value in caps['real_by_experts'].items():
        possible_growth.setdefault(value, []).append(int(experts))
    require(len(caps['real_by_experts']) == 64, 'capacity table must cover E1..64')
    extent = M * 2048 * 4
    require(warmup['M'] == M and warmup['bytes'] == extent,
            'load warmup shape/extent mismatch')
    require(all(warmup[k] == 0 for k in ('ticket', 'request', 'owner', 'used', 'fallback', 'rc')),
            'load warmup must be an unticketed copy')
    require(warmup['cap_before'] == 0 and warmup['cap_after'] == caps['warmup_64_experts'],
            'load warmup capacity differs from original E64/4096-route sizing')
    first_publish = min(x['_line'] for x in maps['publish'].values())
    require(warmup['_line'] < first_publish, 'warmup must precede the real producers')
    warmup_states = [x for x in extra_states
                     if x['_line'] < warmup['_line'] and x['M'] == M and x['bytes'] == extent]
    require(len(warmup_states) == 1, 'missing/ambiguous load warmup state')
    growth = []
    previous_end = warmup['_line']
    previous_capacity = warmup['cap_after']
    windows = []
    for consumer in consumers:
        t = consumer['ticket']
        producer = maps['producer'][t]
        publish = maps['publish'][t]
        state = maps['requested state'][t]
        require(previous_end < publish['_line'] < producer['_line'] <
                state['_line'] < consumer['_line'], f'ticket {t}: invalid lifecycle order')
        windows.append((publish['_line'], consumer['_line']))
        previous_end = consumer['_line']
        for name, event in [('producer', producer), ('publish', publish),
                            ('state', state), ('consumer', consumer)]:
            require(event['M'] == M and event['bytes'] == extent,
                    f'ticket {t}: {name} shape/extent mismatch')
        require(producer['cached'] == 1, f'ticket {t}: producer did not retain input')
        require(publish['saved'] == t and publish['epoch'] > 0 and
                publish['epoch'] == publish['published'],
                f'ticket {t}: producer stamp mismatch')
        require(publish['source_scope'] == state['source_scope'] == 'arm',
                f'ticket {t}: not the source-separated paired-client protocol')
        owner = address(publish['owner'])
        require(owner != 0 and owner == address(publish['base']) == address(state['owner']) ==
                address(state['base']), f'ticket {t}: scratch owner changed')
        require(state['saved_ticket'] == t and state['saved_bytes'] == extent,
                f'ticket {t}: consumer stamp mismatch')
        require(state['epoch'] == state['published'] == publish['epoch'],
                f'ticket {t}: scratch epoch changed')
        require(state['match'] == 31, f'ticket {t}: incomplete DSP stamp match')
        require(consumer['request'] == consumer['owner'] == 1 and consumer['rc'] == 0,
                f'ticket {t}: request/stamp/RPC failure')
        require(consumer['used'] in (0, 1) and consumer['fallback'] == 1 - consumer['used'],
                f'ticket {t}: inconsistent copy-skip/fallback flags')
        before, after = consumer['cap_before'], consumer['cap_after']
        require(before >= previous_capacity and before % 128 == after % 128 == 0,
                f'ticket {t}: invalid/shrinking capacity sequence')
        if consumer['used']:
            require(after == before, f'ticket {t}: reuse crossed an allocation growth')
        else:
            require(after > before, f'ticket {t}: copy fallback without scratch growth')
            require(after in possible_growth,
                    f'ticket {t}: grown capacity differs from original native-LFM sizing')
            growth.append({'ticket': t, 'cap_before': before, 'cap_after': after,
                           'matching_active_experts': possible_growth[after]})
        previous_capacity = after
    # The state tag is broader than the native-LFM consumer tag (M/K only).
    # Unticketed dense/warmup calls are reported separately, never counted as
    # producers, handoff consumers or justified copy fallbacks.
    for state in extra_states:
        require(state['source_scope'] == 'arm' and state['saved_ticket'] == 0 and
                state['saved_bytes'] == 0 and address(state['owner']) == 0,
                'unticketed call unexpectedly carried an active handoff stamp')
        require(not any(start < state['_line'] < end for start, end in windows),
                'unticketed call interposed inside a live handoff')
    result.update({
        'validation': 'PASS', 'expected_layers': expected_layers,
        'ticket_bijection_count': len(tickets), 'requested_state_count': len(requested_states),
        'unticketed_state_count': len(extra_states),
        'extra_unticketed_state_count': len(extra_states) - len(warmup_states),
        'proven_capacity_fallbacks': growth,
        'capacity_formula_sha256': capacity_table['formula_sha256'],
        'capacity_proof_scope': 'Exact unchanged native-LFM sizing; top4 rows and compacted E1..64. '
                                'Matching active-expert counts are inferred from capacities, not separately logged routes.',
    })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--single-block-on', action='store_true')
    p.add_argument('--source-separated', action='store_true', help='require the source-separated protocol')
    p.add_argument('--capacity-table', type=Path,
                   default=Path(__file__).with_name('original-scratch-capacities-v81.json'))
    args = p.parse_args()
    result = parse_text(args.log.read_text(errors='replace'), args.log)
    if args.source_separated and not args.single_block_on:
        p.error('--source-separated requires --single-block-on')
    if args.single_block_on:
        try:
            validate(result, json.loads(args.capacity_table.read_text()))
        except (InvalidTrace, KeyError, ValueError) as exc:
            p.error(str(exc))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('rows')}, indent=2))


if __name__ == '__main__':
    main()

"""Verify the checked-in B/C data, Android catalog and local web assets."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify():
    source = BASE / 'data/personalization_bc_observed_v1'
    manifest = load(source / 'manifest.json')
    for name, value in manifest['files_sha256'].items():
        assert digest(source / name) == value, name
    catalog = load(BASE / 'data/demo-catalog.json')
    assert digest(source / 'manifest.json') == catalog['dataset_manifest_sha256']
    assert digest(source / 'app_catalog.json') == catalog['app_catalog_sha256']
    assert catalog['app_catalog'][9]['id'] == 'ibk_bank'
    exported = load(BASE / 'data/data-export.json')
    for name, meta in exported['files'].items():
        file = BASE / 'data' / name
        assert digest(file) == meta['sha256'], name
        rows = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
        assert len(rows) == meta['rows']
        if 'base' in meta:
            # Augmented training file: the original rows, unchanged, then generated rows.
            base = (BASE / 'data' / meta['base']).read_text(encoding='utf-8').splitlines()
            assert len(base) == meta['base_rows'] and file.read_text(encoding='utf-8').splitlines()[:len(base)] == base, name
        else:
            _, uid, split = Path(name).parts
            assert digest(source / 'by_user' / uid / split) == meta['source_sha256']
        for row in rows:
            assert 0 < len(row['input_ids']) <= 180
            assert all(type(i) is int and i >= 0 for i in row['input_ids'])
            assert len(row['target_probs']) == 14 and sum(row['target_probs']) == 1
            assert set(row['target_probs']) <= {0.0, 1.0}
            assert row['target_probs'][row['primary_label_index']] == 1
    for user in catalog['users']:
        uid = user['id']
        assert user['validation_rows'] == user['test_rows'] == 80
        train = BASE / f'data/device/{uid}' / Path(user['data_path']).name
        assert digest(train) == user['data_sha256'] and len(train.read_text(encoding='utf-8').splitlines()) == user['expected_data_rows']
        for split, key in [('validation', 'valid_sha256'), ('test', 'test_sha256')]:
            assert digest(BASE / f'data/device/{uid}/{split}.jsonl') == user[key]
        assert 1 <= len(user['scenarios']) <= 4
    kotlin = REPO / 'Applications/CausalLM/Android/SampleTestAPP/src/main/java/com/example/sampletestapp/ObservedDemoCatalog.kt'
    if kotlin.exists():
        text = kotlin.read_text(encoding='utf-8')
        assert catalog['dataset_manifest_sha256'] in text and catalog['app_catalog_sha256'] in text
        definitions = [json.loads(s) for s in re.findall(r'JSONObject\("""(.*?)"""\)', text)]
        assert definitions == catalog['users'], 'Kotlin catalog differs from packaged catalog'
    icons = load(BASE / 'assets/icons/sources.json')['icons']
    for info in icons.values():
        assert digest(BASE / info['asset']) == info['sha256']
    class References(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ('src', 'href') and value and not value.startswith(('#', 'http:', 'https:', 'data:')):
                    assert (BASE / value.split('?')[0]).is_file(), value
    References().feed((BASE / 'demo_combined.html').read_text(encoding='utf-8'))
    for name in re.findall(r'url\(["\']?([^"\')]+)', (BASE / 'demo.css').read_text(encoding='utf-8')):
        assert (BASE / name).is_file(), name
    assert (BASE / 'assets/galaxy_s26_ultra_white.png').is_file()
    return {'passed': True, 'native_files': len(exported['files']), 'users': ['B', 'C'],
            'rows_per_user': {u['label']: u['expected_data_rows'] for u in catalog['users']},
            'dataset_manifest_sha256': catalog['dataset_manifest_sha256']}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))

"""Deploy verified demo rows; optionally copy prepared native model folders."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from verify_bundle import BASE, digest, verify

REMOTE = '/sdcard/quickai_bc_observed_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--adb', default='adb')
    parser.add_argument('--cpu-model-dir', type=Path)
    parser.add_argument('--gpu-model-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', args.serial):
        parser.error('Invalid serial')
    verify()
    uploads = []
    for relative in json.loads((BASE / 'data/data-export.json').read_text(encoding='utf-8'))['files']:
        path = BASE / 'data' / relative
        uploads.append((path, REMOTE + '/' + str(Path(relative).relative_to('device')).replace('\\', '/')))
    uploads.extend([(BASE / 'data/demo-catalog.json', REMOTE + '/demo-catalog.json'),
                    (BASE / 'data/personalization_bc_observed_v1/app_catalog.json', REMOTE + '/app_catalog.json')])
    for kind, folder in [('cpu', args.cpu_model_dir), ('gpu', args.gpu_model_dir)]:
        if folder is None:
            continue
        folder = folder.expanduser().resolve()
        config = json.loads((folder / 'nntr_config.json').read_text(encoding='utf-8'))
        model = config['model_file_name']
        if Path(model).name != model or not re.fullmatch(r'[A-Za-z0-9_.-]+', model):
            parser.error('model_file_name must be a basename')
        names = ['config.json', 'decision.json', 'generation_config.json', 'nntr_config.json',
                 'tokenizer.json', 'tokenizer_config.json', model]
        for name in names:
            path = folder / name
            if not path.is_file():
                parser.error('Missing model file: ' + str(path))
            uploads.append((path, f'{REMOTE}/{kind}/{name}'))
    if args.dry_run:
        print(json.dumps({'serial': args.serial, 'remote': REMOTE,
              'uploads': [{'source': str(p), 'destination': d} for p, d in uploads]}, indent=2))
        return
    def adb(*parts):
        return subprocess.check_output([args.adb, '-s', args.serial, *parts], text=True, timeout=600)
    for path, remote in uploads:
        adb('shell', 'mkdir', '-p', remote.rsplit('/', 1)[0])
        adb('push', str(path), remote)
        actual = adb('shell', 'sha256sum', remote).split()[0]
        if actual != digest(path):
            raise RuntimeError('Device SHA256 mismatch: ' + remote)
        print('Verified:', remote, flush=True)
    print('Data ready. CPU/GPU model folders must exist under ' + REMOTE + ' before training.')


if __name__ == '__main__':
    main()

"""Pull and pin only the selected pilot images, outside measured GPU runs."""
import argparse
import json
from pathlib import Path
import subprocess

from .prepare import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--reuse-from', type=Path)
    a = p.parse_args()
    pinned = {}
    for row in json.loads((a.pilot/'instances.json').read_text()):
        iid = row['instance_id']
        path = a.pilot/'images'/f'{iid}.json'
        prior = a.reuse_from/'images'/f'{iid}.json' if a.reuse_from else None
        if not path.exists() and prior and prior.exists():
            record = json.loads(prior.read_text())
            assert record['tag'] == row['image_tag']
            subprocess.run(['docker','image','inspect',record['id']], check=True, stdout=subprocess.DEVNULL)
            save(path, record)
        if path.exists():
            record = json.loads(path.read_text())
            assert record['tag'] == row['image_tag']
            subprocess.run(['docker','image','inspect',record['id']], check=True, stdout=subprocess.DEVNULL)
        else:
            print('PULL', iid, flush=True)
            subprocess.run(['docker','pull',row['image_tag']], check=True, timeout=1800)
            info = json.loads(subprocess.check_output(['docker','image','inspect',row['image_tag']], text=True))[0]
            record = dict(tag=row['image_tag'], id=info['Id'], digests=info['RepoDigests'])
            save(path, record)
        pinned[iid] = record
        print('PINNED', iid, record['id'], flush=True)
    save(a.pilot/'images.json', pinned)
    print('IMAGES READY', len(pinned), flush=True)


if __name__ == '__main__':
    main()

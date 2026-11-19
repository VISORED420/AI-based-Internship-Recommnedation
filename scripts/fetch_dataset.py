"""Download the external source file and record its provenance.

Usage:
    python scripts/fetch_dataset.py [--force]

Writes data/external/raw/<file>.csv and data/external/manifest.json.
Refuses to overwrite an existing download unless --force is passed.
"""
import argparse
import hashlib
import json
import os
import urllib.request

DATASET = 'Uzaib52/indian-tech-career-intelligence-2026'
REVISION = 'a6182954612f555ea3732cff30f36c8bb7901e4b'
FILENAME = 'indian_tech_career_intelligence_2026.csv'
URL = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILENAME}'
OUT_DIR = 'data/external/raw'
OUT_PATH = os.path.join(OUT_DIR, FILENAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        print(f'{OUT_PATH} already exists (use --force to re-download)')
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        print('downloading', URL)
        with urllib.request.urlopen(URL) as response, open(OUT_PATH, 'wb') as handle:
            handle.write(response.read())

    with open(OUT_PATH, 'rb') as handle:
        sha256 = hashlib.sha256(handle.read()).hexdigest()
    size = os.path.getsize(OUT_PATH)
    with open(OUT_PATH, 'r', encoding='utf-8', errors='replace') as handle:
        rows = sum(1 for _ in handle) - 1

    manifest = {
        'dataset': DATASET,
        'revision': REVISION,
        'url': URL,
        'file': OUT_PATH,
        'sha256': sha256,
        'bytes': size,
        'records': rows,
        'license': 'MIT (declared in the dataset card)',
        'license_note': 'Synthetic content; real company names attached to fabricated '
                        'postings. Not used for training, never shown to end users.',
    }
    os.makedirs('data/external', exist_ok=True)
    with open('data/external/manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

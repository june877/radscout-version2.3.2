"""Download and verify the unmodified v3.3.0 inference checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path, entry):
    return path.is_file() and path.stat().st_size == entry['size'] and sha256(path) == entry['sha256']


def install(archive, root, manifest):
    if sha256(archive) != manifest['archive_sha256']:
        raise ValueError('Model archive SHA-256 does not match models-manifest.json')
    with tempfile.TemporaryDirectory(prefix='radscout-models-') as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            # Extract only explicit manifest entries after checking repository boundaries.
            for entry in manifest['files']:
                relative = Path(entry['path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid model path in manifest')
                destination = (root / relative).resolve()
                if not destination.is_relative_to(root.resolve()):
                    raise ValueError('Model path escapes repository')
                staged = stage / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry['path']) as source, staged.open('wb') as target:
                    shutil.copyfileobj(source, target)
                if not valid(staged, entry):
                    raise ValueError('Checkpoint checksum failed: ' + entry['path'])
            # All models are verified before any existing checkpoint is replaced.
            for entry in manifest['files']:
                destination = root / entry['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary_target = destination.with_suffix(destination.suffix + '.tmp')
                shutil.copyfile(stage / entry['path'], temporary_target)
                temporary_target.replace(destination)
                print('Verified:', entry['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use a previously downloaded release ZIP')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'models-manifest.json').read_text(encoding='utf-8'))
    if all(valid(root / entry['path'], entry) for entry in manifest['files']):
        print('Both v3.3.0 models are present and verified.')
        return
    if args.archive:
        install(args.archive, root, manifest)
    else:
        with tempfile.TemporaryDirectory(prefix='radscout-download-') as temporary:
            archive = Path(temporary) / 'models.zip'
            print('Downloading:', manifest['archive_url'])
            request = urllib.request.Request(manifest['archive_url'], headers={'User-Agent': 'RadScout/3.3.0'})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as target:
                shutil.copyfileobj(response, target)
            install(archive, root, manifest)


if __name__ == '__main__':
    main()

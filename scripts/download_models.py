#!/usr/bin/env python3
"""Explicit, resumable, SHA256-verified model setup using the Python stdlib."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import urllib.request


def verify(path, asset):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != asset['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == asset['sha256']


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in value:
        raise ValueError(f'Unsafe manifest path: {value}')
    return Path(*path.parts)


def asset_url(asset, endpoint):
    if 'repo' in asset:
        return f"{endpoint.rstrip('/')}/{asset['repo']}/resolve/{asset['revision']}/{asset['filename']}"
    return asset['official_url']


def _publish(part, destination, asset):
    if not verify(part, asset):
        raise ValueError(f'Integrity failure for {destination.name}; retained {part} for inspection')
    # Atomic create, not replace: another writer or a bad existing file wins
    # the race without being silently overwritten. Both paths are same folder.
    try:
        os.link(part, destination)
    except FileExistsError:
        if not verify(destination, asset):
            raise ValueError(f'Existing destination differs: {destination}')
    part.unlink()


def download(asset, destination, url, retries=3, timeout=60):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        if verify(destination, asset):
            return 'verified-existing'
        raise ValueError(f'Existing file failed size/SHA256; not overwritten: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + '.part')
    if part.is_symlink():
        raise ValueError(f'Refusing symlink partial file: {part}')
    if part.exists() and part.stat().st_size >= asset['size']:
        _publish(part, destination, asset)
        return 'verified-partial'
    for attempt in range(retries+1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'User-Agent': 'TaoMate-ComfyUI-model-setup/1', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or int(match[3]) != asset['size']:
                        raise ValueError('Server returned an invalid Content-Range')
                    mode = 'ab' if offset else 'wb'
                elif status == 200:
                    # A server ignoring Range sent the full object. Restart
                    # the partial safely rather than appending duplicate data.
                    mode = 'wb'
                    offset = 0
                else:
                    raise ValueError(f'Unexpected HTTP status: {status}')
                with part.open(mode) as stream:
                    written = offset
                    while True:
                        chunk = response.read(1024*1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > asset['size']:
                            raise ValueError('Server response exceeds pinned size')
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if part.stat().st_size < asset['size']:
                raise OSError('Incomplete response; resume on next attempt')
            _publish(part, destination, asset)
            return 'downloaded'
        except ValueError:
            # A complete bad digest or bad server metadata must not be retried
            # as though it were a transient disconnect.
            raise
        except (OSError, EOFError, http.client.IncompleteRead) as exc:
            if attempt == retries:
                raise RuntimeError(f'Download failed; resume by rerunning: {part}') from exc
            time.sleep(min(2**attempt, 8))


def reuse(asset, destination, roots):
    """Only named candidates under explicitly supplied roots; never scan."""
    candidates = [relative_path(asset['path'])]
    filename = relative_path(asset.get('filename', Path(asset['path']).name))
    if filename not in candidates:
        candidates.append(filename)
    for root in roots:
        for relative in candidates:
            source = Path(root) / relative
            if source.is_file() and verify(source, asset):
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    destination.symlink_to(source.resolve())
                except OSError as exc:
                    raise RuntimeError('Verified reuse requires symlink permission. Enable symlinks or configure ComfyUI extra_model_paths to an existing complete model directory; no large copy was made.') from exc
                return str(source)
    return None


def gpen_source(path, source, check=False):
    path = Path(path)
    if path.exists():
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        if head != source['revision'] or dirty:
            raise ValueError('Existing GPEN checkout is not the clean pinned revision; left untouched')
        return 'verified-existing'
    if check:
        raise FileNotFoundError(f'GPEN source missing: {path}')
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    subprocess.run(['git', 'clone', '--no-checkout', source['url'], str(path)], env=env, check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', source['revision']], env=env, check=True)
    return gpen_source(path, source, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'assets.json')
    parser.add_argument('--endpoint', default='https://huggingface.co', help='HF endpoint only; e.g. https://hf-mirror.com')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Print pinned assets without network or hashing')
    mode.add_argument('--check', action='store_true', help='Check size and SHA256 only; do not download')
    parser.add_argument('--reuse', type=Path, action='append', default=[], help='Explicit directory; repeatable, no scanning')
    parser.add_argument('--only', nargs='+', help='Optional asset IDs, e.g. checkpoint_config')
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--gpen-source-dir', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assets = manifest['assets']
    if args.only:
        missing = set(args.only) - {a['id'] for a in assets}
        if missing:
            parser.error(f'Unknown asset IDs: {sorted(missing)}')
        assets = [a for a in assets if a['id'] in args.only]
    if args.retries < 0:
        parser.error('--retries must be nonnegative')
    for asset in assets:
        destination = args.dest/relative_path(asset['path'])
        url = asset_url(asset, args.endpoint)
        if args.plan:
            print(json.dumps(dict(id=asset['id'], path=str(destination), size=asset['size'], sha256=asset['sha256'], url=url)))
        elif args.check:
            if not verify(destination, asset):
                raise ValueError(f'Missing or failed size/SHA256: {destination}')
            print(f"{asset['id']}: verified", flush=True)
        elif destination.exists() or destination.is_symlink():
            print(f"{asset['id']}: {download(asset, destination, url, args.retries)}", flush=True)
        else:
            reused = reuse(asset, destination, args.reuse)
            print(f"{asset['id']}: reused {reused}" if reused else f"{asset['id']}: {download(asset, destination, url, args.retries)}", flush=True)
    if args.gpen_source_dir:
        if args.plan:
            print(json.dumps(dict(gpen_source=manifest['gpen_source'], path=str(args.gpen_source_dir))))
        else:
            print(f"GPEN source: {gpen_source(args.gpen_source_dir, manifest['gpen_source'], args.check)}")


if __name__ == '__main__':
    main()

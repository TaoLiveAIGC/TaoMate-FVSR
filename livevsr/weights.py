"""Resolve local assets, Hugging Face models, and official upstream downloads."""
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen


DEFAULT_CONFIG = Path(__file__).with_name('model_sources.json')


def download_upstream(spec, *, offline=False, cache_dir=None):
    cache = Path(cache_dir).expanduser() if cache_dir else Path.home() / '.cache' / 'livevsr'
    path = cache / 'upstream' / spec['filename']
    if path.is_file():
        return path
    if offline:
        raise FileNotFoundError(f"Missing offline asset: {path}; download it first or use --model-dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as output:
        temporary = Path(output.name)
        try:
            with urlopen(spec['url'], timeout=60) as response:
                shutil.copyfileobj(response, output)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def resolve_asset(name, override, model_dir, sources, *, offline=False, cache_dir=None):
    spec = sources[name]
    if override is not None:
        path = Path(override).expanduser().resolve()
    elif model_dir is not None:
        path = Path(model_dir).expanduser().resolve() / spec['local_path']
    elif spec.get('url'):
        path = download_upstream(spec, offline=offline, cache_dir=cache_dir)
    else:
        repo = spec.get('repo_id')
        if not repo or repo.startswith('YOUR_ORG/'):
            raise ValueError(f"Configure the {name} Hugging Face repo_id, or supply --{name.replace('_', '-')} / --model-dir")
        from huggingface_hub import hf_hub_download, snapshot_download
        options = dict(repo_id=repo, revision=spec.get('revision', 'main'),
                       local_files_only=offline, cache_dir=cache_dir)
        if name == 'hubert':
            path = Path(snapshot_download(**options, allow_patterns=[
                'config.json', 'preprocessor_config.json', 'model.safetensors', 'pytorch_model.bin']))
        else:
            path = Path(hf_hub_download(**options, filename=spec['filename']))
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} asset: {path}")
    return path


def load_sources(path=DEFAULT_CONFIG, *, scale=1):
    sources = json.loads(Path(path).read_text())
    if scale == 2:
        sources['gpen_weights'] = sources['gpen_weights_1024']
    return sources

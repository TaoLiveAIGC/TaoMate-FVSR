"""Actual Comfy loader/schema check, CPU only. Pass checkout as first argument."""
import asyncio
import json
from pathlib import Path
import sys

checkout = Path(sys.argv[1]).resolve()
plugin = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(checkout))
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import nodes
import execution
import torch


async def main():
    assert await nodes.load_custom_node(str(checkout/'comfy_extras/nodes_preview_any.py'), module_parent='comfy_extras')
    assert await nodes.load_custom_node(str(plugin))
    for filename in ('cropped-face-api.json', 'yolo-gpen-api.json'):
        graph = json.loads((plugin / 'examples' / filename).read_text())
        # Avoid filesystem-specific built-in LoadImage validation for schema only.
        if graph['3']['class_type'] == 'LoadImage':
            del graph['3']
            graph['4']['inputs']['reference'] = ['2', 1]
        result = await execution.validate_prompt('taomate-cpu-schema', graph, None)
        print(filename, json.dumps(result, ensure_ascii=False, default=str))
        assert result[0] and not result[3], result
    assert not torch.cuda.is_initialized(), 'CPU schema validation initialized CUDA'
    print('ComfyUI loader + prompt schema PASS; CUDA not initialized')


asyncio.run(main())

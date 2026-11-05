"""CPU integration: real FFmpeg, disk crops, audio retention, error cleanup."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
folder = types.ModuleType('folder_paths')
folder.models_dir = '/nonexistent'
folder.get_folder_paths = lambda key: []
sys.modules['folder_paths'] = folder
spec = importlib.util.spec_from_file_location('taomate_plugin', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
from taomate_plugin.node_io import prepare_video, read_rgb, save_video, RestoredVideo, composite_frames, PreparedVideo
from taomate_plugin.nodes import reference_array, TaoMateRestore


class IOTest(unittest.TestCase):
    def test_manual_roi_roundtrip_stereo_and_frame_count(self):
        from livevsr.media import probe
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=96x64:r=25:d=0.2',
                '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-shortest', '-c:v', 'libx264',
                '-c:a', 'aac', str(source)], check=True)
            p = prepare_video(str(source), mode='manual_roi', x=17, y=1, side=61, workspace=root/'cache')
            self.assertEqual(len(p.crops), 5)
            self.assertEqual(read_rgb(p.crops[0]).shape, (512, 512, 3))
            frames = np.stack([read_rgb(path) for path in p.crops])
            out = save_video(RestoredVideo(p, frames), root/'result.mp4')
            streams = probe(out)
            video = next(s for s in streams if s['codec_type'] == 'video')
            audio = next(s for s in streams if s['codec_type'] == 'audio')
            self.assertEqual((video['width'], video['height'], int(video['nb_frames'])), (96,64,5))
            self.assertEqual((audio['channels'], audio['sample_rate']), (2, '48000'))
            with self.assertRaises(ValueError):
                prepare_video(str(source), mode='cropped_face', workspace=root/'bad')

    def test_reference_shape(self):
        with self.assertRaises(ValueError):
            reference_array(torch.zeros(2, 512, 512, 3))
        self.assertEqual(reference_array(torch.ones(1,512,512,3)).min(), 255)

    def test_yolo_missing_detection_keeps_frame_and_original_pixels(self):
        comfy = types.ModuleType('comfy')
        comfy.model_management = Mock()
        detector = types.ModuleType('taomate_plugin.crop_yolo')
        detector.prepare_yolo_crops = Mock(return_value=(
            np.array([[16,0,80,64],[0,0,0,0],[16,0,80,64],[16,0,80,64],[16,0,80,64]]),
            np.array([True,False,True,True,True]), []))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,
                {'comfy':comfy, 'taomate_plugin.crop_yolo':detector}):
            root = Path(tmp)
            source = root/'source.mp4'
            weights = root/'fake.pt'
            weights.touch()
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=96x64:r=25:d=0.2',
                            '-c:v','libx264',str(source)],check=True)
            p = prepare_video(str(source),mode='yolo_face',workspace=root/'cache',detector_path=str(weights))
            self.assertEqual(len(p.crops),5)
            result = RestoredVideo(p,np.zeros((5,512,512,3),dtype=np.uint8))
            frames = list(composite_frames(result))
            self.assertTrue(np.array_equal(frames[1],read_rgb(p.scene_paths[1])))

    def test_composite_matches_upstream_and_preserves_outside_box(self):
        import cv2
        import livevsr
        from PIL import Image
        from livevsr.composite import paste_head
        rng = np.random.default_rng(123)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = rng.integers(0,256,(80,96,3),dtype=np.uint8)
            crop = rng.integers(0,256,(512,512,3),dtype=np.uint8)
            generated = rng.integers(0,256,(512,512,3),dtype=np.uint8)
            sp, cp = root/'scene.png', root/'crop.png'
            Image.fromarray(scene).save(sp)
            Image.fromarray(crop).save(cp)
            alpha = cv2.resize(cv2.imread(str(Path(livevsr.__file__).parent/'assets/paste_face_mask.png'),
                                          cv2.IMREAD_GRAYSCALE),(512,512)).astype(np.float32)/255.
            box = np.array([16,8,80,72])
            expected = paste_head(scene,crop,generated,box,alpha)
            for roi, boxes in [((16,8,64),None),(None,box[None])]:
                p = PreparedVideo([cp],sp,sp,root,(96,80),roi,[sp],boxes,np.array([True]))
                actual = next(composite_frames(RestoredVideo(p,generated[None])))
                self.assertTrue(np.array_equal(actual,expected))
                outside = np.ones((80,96),dtype=bool)
                outside[8:72,16:80] = False
                self.assertTrue(np.array_equal(actual[outside],scene[outside]))

    def test_restore_exception_unloads_model(self):
        comfy = types.ModuleType('comfy')
        comfy.model_management = Mock()
        utils = types.ModuleType('comfy.utils')
        utils.ProgressBar = Mock()
        model = Mock()
        model.restore.side_effect = RuntimeError('simulated runtime failure')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'comfy': comfy, 'comfy.utils': utils}), \
                patch('livevsr.media.prepare_audio', return_value=np.zeros((1,640),dtype=np.int16)):
            video = types.SimpleNamespace(workspace=Path(tmp), audio='audio.wav', crops=['crop.png'])
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                TaoMateRestore().restore(model,video,torch.zeros(1,512,512,3),42,10,.8,5,4,1.7)
            model.unload.assert_called_once()
            comfy.model_management.unload_all_models.assert_called_once()

    def test_api_connections(self):
        graph = json.loads((ROOT/'examples/cropped-face-api.json').read_text())
        for item in graph.values():
            name = item['class_type']
            if name not in plugin.NODE_CLASS_MAPPINGS:
                continue
            cls = plugin.NODE_CLASS_MAPPINGS[name]
            schema = cls.INPUT_TYPES()
            required = {**schema['required'], **schema.get('optional',{})}
            self.assertEqual(set(required), set(item['inputs']))
            for key, val in item['inputs'].items():
                if isinstance(val, list):
                    src = graph[val[0]]['class_type']
                    if src in plugin.NODE_CLASS_MAPPINGS:
                        self.assertEqual(plugin.NODE_CLASS_MAPPINGS[src].RETURN_TYPES[val[1]], required[key][0])

    def test_yolo_ui_widget_order_matches_api_and_node_schema(self):
        self.check_ui_widgets('yolo-gpen')

    def test_cropped_ui_widget_order_matches_api_and_node_schema(self):
        self.check_ui_widgets('cropped-face')

    def check_ui_widgets(self, prefix):
        graph = json.loads((ROOT/'examples'/f'{prefix}-api.json').read_text())
        ui = json.loads((ROOT/'examples'/f'{prefix}-workflow.json').read_text())
        self.assertEqual(graph['8'], {'class_type':'PreviewAny','inputs':{'source':['5',0]}})
        preview = next(n for n in ui['nodes'] if n['id'] == 8)
        link = next(link for link in ui['links'] if link[0] == preview['inputs'][0]['link'])
        self.assertEqual(link[1:], [5,0,8,0,'STRING'])
        for node in ui['nodes']:
            if node['type'] not in plugin.NODE_CLASS_MAPPINGS:
                continue
            cls = plugin.NODE_CLASS_MAPPINGS[node['type']]
            schema = cls.INPUT_TYPES()
            inputs = {**schema['required'], **schema.get('optional',{})}
            api_inputs = graph[str(node['id'])]['inputs']
            self.assertEqual(set(api_inputs),set(inputs))
            expected = []
            for name, field in inputs.items():
                if isinstance(field[0],list) or field[0] in ('INT','FLOAT','STRING','BOOLEAN'):
                    expected.append(api_inputs[name])
                    if name == 'seed':
                        expected.append('fixed')
            self.assertEqual(node['widgets_values'],expected)


if __name__ == '__main__':
    unittest.main()

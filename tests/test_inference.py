import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from livevsr.cli import parser, validate
from livevsr.geometry import windows, input_indices
from livevsr.weights import load_sources, resolve_asset


class GeometryTests(unittest.TestCase):
    def test_exact_coverage_and_overlap(self):
        for count in (1, 5, 71, 72, 73, 76, 77, 128, 129, 200, 584):
            plans = windows(count)
            self.assertEqual(sum(p['emit_frames'] for p in plans), count)
            self.assertEqual(len(input_indices(count)) % 4, 1)
            for i, plan in enumerate(plans):
                self.assertEqual(len(plan['input_indices']), 77)
                self.assertTrue(all(0 <= n < count for n in plan['input_indices']))
                if i:
                    self.assertEqual(plans[i-1]['retained_stop'], plan['retained_start'])
                    self.assertEqual(plans[i-1]['latent_start'] + 16, plan['latent_start'] + 2)
                    self.assertEqual(plan['input_indices'][20], plan['retained_start']-1)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            windows(0)


class CLITests(unittest.TestCase):
    def test_input_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'input.mp4'
            video.touch()
            args = parser().parse_args(['--input', str(video), '--output', str(video), '--overwrite'])
            with self.assertRaisesRegex(ValueError, 'overwrite an input'):
                validate(args)

    def test_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, output = Path(tmp) / 'input.mp4', Path(tmp) / 'output.mp4'
            video.touch()
            output.touch()
            args = parser().parse_args(['--input', str(video), '--output', str(output)])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                validate(args)

    def test_two_x_is_explicit(self):
        args = parser().parse_args(['--input', 'input.mp4', '--output', 'out.mp4', '--scale', '2'])
        self.assertEqual(args.scale, 2)
        self.assertFalse(args.cropped)

    def test_current_defaults_and_edited_reference(self):
        args = parser().parse_args(['--input', 'input.mp4', '--output', 'out.mp4',
                                   '--edited-reference', 'edited.png'])
        self.assertEqual((args.steps, args.noise, args.audio_cfg, args.ref_cfg), (10, .8, 4., 1.7))
        self.assertEqual(args.gpen_passes, 1)
        self.assertIsNone(args.reference)

    def test_cropped_bypasses_scene_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, reference = Path(tmp) / 'face.mp4', Path(tmp) / 'ref.png'
            video.touch()
            reference.touch()
            args = parser().parse_args(['--input', str(video), '--output', str(Path(tmp) / 'out.mp4'),
                                       '--reference', str(reference), '--cropped'])
            validate(args)

    def test_full_scene_uses_bundled_mask(self):
        from PIL import Image
        from livevsr.pose import FILES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, reference = root / 'input.mp4', root / 'ref.png'
            video.touch()
            reference.touch()
            pose = root / 'models' / 'dwpose'
            pose.mkdir(parents=True)
            for name in FILES.values():
                (pose / name).touch()
            command = ['--input', str(video), '--output', str(root / 'out.mp4'),
                       '--reference', str(reference), '--model-dir', str(root / 'models')]
            args = parser().parse_args(command)
            validate(args)
            self.assertFalse((root / 'models' / 'paste_face_mask.png').exists())
            with Image.open(args.blend_mask) as mask:
                self.assertEqual((mask.mode, mask.size), ('L', (160, 208)))
                custom = root / 'custom.png'
                mask.save(custom)
            args = parser().parse_args(command + ['--blend-mask', str(custom)])
            validate(args)
            self.assertEqual(args.blend_mask, custom.resolve())


class WeightsTests(unittest.TestCase):
    def test_placeholder_never_downloads(self):
        sources = load_sources()
        sources['checkpoint']['repo_id'] = None
        with self.assertRaisesRegex(ValueError, 'Configure'):
            resolve_asset('checkpoint', None, None, sources)

    def test_local_models_do_not_require_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'TaoMate-FVSR.safetensors'
            path.touch()
            with patch('huggingface_hub.hf_hub_download') as download:
                self.assertEqual(resolve_asset('checkpoint', None, Path(tmp), load_sources()), path.resolve())
                self.assertEqual(resolve_asset('checkpoint', path, None, load_sources()), path.resolve())
                download.assert_not_called()

    def test_official_upstream_sources(self):
        sources = load_sources()
        self.assertEqual(sources['checkpoint']['repo_id'], 'TaoLiveAIGC/TaoMate-FVSR')
        self.assertEqual(sources['checkpoint']['filename'], 'TaoMate-FVSR.safetensors')
        self.assertEqual(sources['checkpoint']['config_filename'], 'config.json')
        for name in ('gpen_weights', 'gpen_detector', 'gpen_parser'):
            spec = sources[name]
            self.assertEqual(spec['url'], 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/'
                             'robin/models/' + spec['filename'])
            self.assertNotIn('repo_id', spec)

    def test_gpen_source_follows_scale(self):
        for scale, size in ((1, 512), (2, 1024)):
            sources = load_sources(scale=scale)
            spec = sources['gpen_weights']
            self.assertEqual(spec['local_path'], f'GPEN-BFR-{size}.pth')
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / spec['local_path']
                path.touch()
                self.assertEqual(resolve_asset('gpen_weights', None, Path(tmp), sources,
                                               offline=True), path.resolve())
        spec = load_sources(scale=2)['gpen_weights']
        self.assertEqual(spec['url'], 'https://www.modelscope.cn/models/damo/'
            'cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model.pt')
        with tempfile.TemporaryDirectory() as tmp:
            with patch('livevsr.weights.urlopen', return_value=io.BytesIO(b'1024 weights')) as download:
                path = resolve_asset('gpen_weights', None, None, load_sources(scale=2), cache_dir=tmp)
                self.assertEqual(path.name, 'GPEN-BFR-1024.pth')
                download.assert_called_once_with(spec['url'], timeout=60)

    def test_huggingface_checkpoint_and_offline_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'TaoMate-FVSR.safetensors'
            path.touch()
            with patch('huggingface_hub.hf_hub_download', return_value=str(path)) as download:
                for offline in (False, True):
                    download.reset_mock()
                    self.assertEqual(resolve_asset('checkpoint', None, None, load_sources(),
                                                   offline=offline, cache_dir=tmp), path)
                    options = dict(repo_id='TaoLiveAIGC/TaoMate-FVSR', revision='main',
                                   local_files_only=offline, cache_dir=tmp)
                    expected = [] if offline else [call(**options, filename='config.json')]
                    expected.append(call(**options, filename='TaoMate-FVSR.safetensors'))
                    self.assertEqual(download.call_args_list, expected)

    def test_huggingface_vae_download_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'Wan2.2_VAE.pth'
            path.touch()
            with patch('huggingface_hub.hf_hub_download', return_value=str(path)) as download:
                self.assertEqual(resolve_asset('vae', None, None, load_sources(), cache_dir=tmp), path)
                download.assert_called_once_with(repo_id='Wan-AI/Wan2.2-TI2V-5B',
                    revision='main', local_files_only=False, cache_dir=tmp, filename='Wan2.2_VAE.pth')

    def test_upstream_download_and_cached_offline_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('livevsr.weights.urlopen', return_value=io.BytesIO(b'test weights')) as download:
                path = resolve_asset('gpen_weights', None, None, load_sources(), cache_dir=tmp)
                self.assertEqual(path.read_bytes(), b'test weights')
                reused = resolve_asset('gpen_weights', None, None, load_sources(),
                                       cache_dir=tmp, offline=True)
                self.assertEqual(reused, path)
                download.assert_called_once_with(load_sources()['gpen_weights']['url'], timeout=60)

    def test_missing_upstream_offline_never_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('livevsr.weights.urlopen') as download:
                with self.assertRaisesRegex(FileNotFoundError, 'offline'):
                    resolve_asset('gpen_weights', None, None, load_sources(), cache_dir=tmp, offline=True)
                download.assert_not_called()

    def test_failed_download_does_not_leave_an_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('livevsr.weights.urlopen', side_effect=OSError('connection interrupted')):
                with self.assertRaises(OSError):
                    resolve_asset('gpen_weights', None, None, load_sources(), cache_dir=tmp)
            self.assertEqual(list((Path(tmp) / 'upstream').iterdir()), [])


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('Tensor tests require PyTorch')

    def test_audio_conditioning_projection_and_parameter_names(self):
        import torch
        from livevsr.models.audio_conditioning import AudioProjModule
        model = AudioProjModule(audio_feat_layers=(1,), audio_feat_channels=(16,),
            intermediate_dim=16, output_dim=24, num_layers=1,
            attn_head_dim=8, attn_num_heads=2).eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 20, 8, 16))
        self.assertEqual(tuple(output.shape), (1, 20, 8, 24))
        keys = model.state_dict()
        for key in ('proj_in.weight', 'layers.0.attn.to_q.weight',
                    'layers.0.ff.net.0.proj.weight', 'proj_out.weight'):
            self.assertIn(key, keys)

    def test_audio_inference_interfaces_and_zero_condition(self):
        import torch
        from livevsr.audio import HubertAudioProcessor
        from livevsr.pipeline import MODEL_CONFIG
        from livevsr.models.wan_video_dit import WanModel, inject_audio_conditioning
        from livevsr.models.audio_conditioning import prepare_audio_features
        self.assertTrue(MODEL_CONFIG['enable_audio_conditioning'])
        self.assertTrue(callable(inject_audio_conditioning))
        self.assertTrue(callable(WanModel.project_audio))
        self.assertTrue(callable(HubertAudioProcessor.build_condition))
        empty = prepare_audio_features(None, batch_size=1, device='cpu', dtype=torch.float32)
        self.assertEqual(tuple(empty.shape), (1, 20, 8, 4096))
        self.assertEqual(torch.count_nonzero(empty).item(), 0)

    def test_guidance_order_and_scale_one(self):
        import torch
        from livevsr.denoise import guided_forward
        state = torch.ones(1)
        reference, audio = torch.tensor([2.]), torch.tensor([3.])
        def fake(dit, state, ref, audio, timestep):
            return state + ref + audio
        with patch('livevsr.denoise.forward', side_effect=fake) as model:
            result = guided_forward(None, state, reference, audio, torch.ones(1), 4., 1.5)
            torch.testing.assert_close(result, torch.tensor([16.]))
            self.assertEqual(model.call_count, 4)
        with patch('livevsr.denoise.forward', side_effect=fake) as model:
            result = guided_forward(None, state, reference, audio, torch.ones(1), 1., 1.)
            torch.testing.assert_close(result, torch.tensor([6.]))
            self.assertEqual(model.call_count, 1)

    def test_schedule_endpoints(self):
        from livevsr.denoise import schedule
        ts, sigmas, following = schedule(10, 5.)
        self.assertEqual(len(ts), 10)
        self.assertEqual(float(sigmas[0]), 1.)
        self.assertEqual(float(following[-1]), 0.)
        self.assertTrue(bool((sigmas > following).all()))

    def test_first_window_lr_prefix_only(self):
        import torch
        from livevsr.pipeline import window_lr
        global_lr = torch.arange(48).view(1, 1, 48, 1, 1).float()
        snapshot = global_lr.clone()
        for start in (0, 14, 28):
            lr = window_lr(global_lr, start)
            torch.testing.assert_close(lr[:, :, :2], global_lr[:, :, :2])
            torch.testing.assert_close(lr[:, :, 2:], global_lr[:, :, start+2:start+20])
        torch.testing.assert_close(global_lr, snapshot)

    def test_k2_zeroed_in_negative_not_shared(self):
        import torch
        from livevsr.denoise import guided_forward
        reference = torch.zeros(1, 1, 20, 1, 1)
        reference[:, :, 0] = 2
        reference[:, :, 1] = 3
        reference[:, :, 5] = 7
        calls = []
        def fake(dit, state, ref, audio, timestep):
            calls.append((ref.clone(), audio.clone()))
            return state + ref * (1 + audio.item())
        with patch('livevsr.denoise.forward', side_effect=fake):
            actual = guided_forward(None, torch.zeros_like(reference), reference,
                                    torch.tensor([3.]), torch.ones(1), 4., 1.7)
        self.assertEqual(len(calls), 4)
        for ref, _ in calls[:2]:
            torch.testing.assert_close(ref, reference)
        for ref, _ in calls[2:]:
            self.assertEqual(torch.count_nonzero(ref).item(), 0)
        torch.testing.assert_close(actual, reference * 13 * 1.7)

    def test_decoder_convolution_stripes(self):
        import torch
        from livevsr.vae_memory import striped_conv
        layer = torch.nn.Conv3d(2, 3, 3, padding=0)
        inputs = torch.randn(1, 2, 4, 259, 258)
        with torch.no_grad():
            expected = layer(inputs)
            actual = striped_conv(layer, inputs, layer.weight, layer.bias)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_gpen_alignment_and_reference_follow_resolution(self):
        import sys
        from types import SimpleNamespace
        # Import before patch.dict snapshots sys.modules: OpenCV's native
        # bindings cannot be safely unloaded and re-imported in every version.
        import cv2
        import numpy as np
        from livevsr.reference import prepare_reference
        calls = []

        class Detector:
            def __init__(self, *args, **kwargs):
                pass
            def detect(self, image):
                size = image.shape[0]
                return np.array([[0, 0, size, size, .99]]), np.zeros((1, 10))

        class Parser:
            def __init__(self, *args, **kwargs):
                pass
            def process(self, image):
                return [np.full((512, 512), 255, np.uint8)]

        class Generator:
            def __init__(self, **kwargs):
                calls.append(kwargs)
            def img2tensor(self, image):
                return image
            def model(self, image):
                calls.append(image.shape)
                return image, None
            def tensor2img(self, image):
                return image

        def align(image, points, *, reference_pts, crop_size):
            self.assertEqual(crop_size, reference_pts)
            self.assertEqual(image.shape[:2], crop_size)
            return image, np.array([[1, 0, 0], [0, 1, 0]], np.float32)

        modules = {
            '__init_paths': SimpleNamespace(),
            'face_detect.retinaface_detection': SimpleNamespace(RetinaFaceDetection=Detector),
            'face_parse.face_parsing': SimpleNamespace(FaceParse=Parser),
            'face_model.face_gan': SimpleNamespace(FaceGAN=Generator),
            'align_faces': SimpleNamespace(get_reference_facial_points=lambda size, *args: size,
                                          warp_and_crop_face=align),
        }
        with patch.dict(sys.modules, modules), tempfile.TemporaryDirectory() as tmp:
            for size in (512, 1024):
                calls.clear()
                image = np.full((512, 512, 3), [30, 100, 170], np.uint8)
                result = prepare_reference(image, repo=tmp, weights=Path(tmp)/'gpen.pth',
                    detector_weights=Path(tmp)/'detector.pth', parser_weights=Path(tmp)/'parser.pth',
                    size=size)
                self.assertEqual(result.shape, (size, size, 3))
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(calls[0]['model'], f'GPEN-BFR-{size}')
                self.assertEqual((calls[0]['in_size'], calls[0]['out_size']), (size, size))
                self.assertEqual(calls[1:], [(size, size, 3)])
                np.testing.assert_array_equal(result[0,0], image[0,0])
        self.assertIs(sys.modules.get('cv2'), cv2)


if __name__ == '__main__':
    unittest.main()

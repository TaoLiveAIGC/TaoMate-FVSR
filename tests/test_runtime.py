import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from taomate_runtime import ModelHandle, _move_cache


class RuntimeContractTests(unittest.TestCase):
    def test_construction_does_not_initialize_cuda_or_load_model(self):
        with patch('torch.cuda.current_device', side_effect=AssertionError('GPU touched')):
            handle = ModelHandle('/not/downloaded/yet')
        self.assertIsNone(handle.pipeline)

    def test_input_shape_rejected_before_loading(self):
        handle = ModelHandle('/not/downloaded/yet')
        with self.assertRaisesRegex(ValueError, 'pcm'):
            handle.restore(['frame.png'], np.zeros((512,512,3), np.uint8),
                           np.zeros(640, np.int16), '/tmp/unused')
        self.assertIsNone(handle.pipeline)

    def test_concurrent_use_rejected_without_touching_cuda(self):
        handle = ModelHandle('/not/downloaded/yet')
        handle._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                handle.restore(['frame.png'], np.zeros((512,512,3), np.uint8),
                               np.zeros((1,640), np.int16), '/tmp/unused')
        finally:
            handle._lock.release()

    def test_nested_decoder_cache_preserves_sentinels_and_tensor_values(self):
        source = [None, 'Rep', (torch.arange(4), [torch.ones(2)])]
        moved = _move_cache(source, 'cpu')
        self.assertEqual(moved[:2], [None, 'Rep'])
        self.assertTrue(torch.equal(moved[2][0], source[2][0]))
        self.assertIsInstance(moved[2], tuple)

    def test_cleanup_failure_unlocks_and_preserves_original_error(self):
        handle = ModelHandle('/unused', device='cuda:0')
        with ExitStack() as stack:
            stack.enter_context(patch('torch.cuda.get_allocator_backend', return_value='native'))
            stack.enter_context(patch('torch.cuda.get_per_process_memory_fraction', return_value=1.))
            stack.enter_context(patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=24*2**30)))
            stack.enter_context(patch('torch.cuda.set_per_process_memory_fraction'))
            stack.enter_context(patch.object(handle, '_load', side_effect=InterruptedError('original cancellation')))
            stack.enter_context(patch.object(handle, 'unload', side_effect=RuntimeError('cleanup failed')))
            with self.assertRaisesRegex(InterruptedError, 'original cancellation'):
                handle.restore(['frame'], np.zeros((512,512,3), np.uint8), np.zeros((1,640), np.int16), '/unused')
        self.assertFalse(handle._lock.locked())

    def test_async_allocator_policy_is_untouched(self):
        handle = ModelHandle('/unused', device='cuda:0')
        with ExitStack() as stack:
            stack.enter_context(patch('torch.cuda.get_allocator_backend', return_value='cudaMallocAsync'))
            getter = stack.enter_context(patch('torch.cuda.get_per_process_memory_fraction'))
            setter = stack.enter_context(patch('torch.cuda.set_per_process_memory_fraction'))
            stack.enter_context(patch.object(handle, '_load', side_effect=InterruptedError('cancel')))
            stack.enter_context(patch.object(handle, 'unload'))
            with self.assertRaises(InterruptedError):
                handle.restore(['frame'], np.zeros((512,512,3), np.uint8), np.zeros((1,640), np.int16), '/unused')
        getter.assert_not_called()
        setter.assert_not_called()
        self.assertFalse(handle._lock.locked())


if __name__ == '__main__':
    unittest.main()

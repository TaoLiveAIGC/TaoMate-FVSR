import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location('download_models', Path(__file__).parents[1]/'scripts/download_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PAYLOAD = b'fixed model payload'*99


class Handler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.requests.append(self.headers.get('Range'))
        offset = int(self.headers['Range'].split('=')[1].split('-')[0]) if self.headers.get('Range') else 0
        if self.path == '/ignore-range':
            offset = 0
        self.send_response(206 if offset else 200)
        if offset:
            self.send_header('Content-Range', f'bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}')
        self.send_header('Content-Length', str(len(PAYLOAD)-offset))
        self.end_headers()
        self.wfile.write(PAYLOAD[offset:])


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/model'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'model.bin'
        self.asset = dict(path='model.bin', size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest())
        Handler.requests.clear()

    def test_download_and_skip_verified_existing(self):
        self.assertEqual(module.download(self.asset, self.path, self.url), 'downloaded')
        self.assertEqual(module.download(self.asset, self.path, self.url), 'verified-existing')
        self.assertEqual(len(Handler.requests), 1)
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertFalse(self.path.with_suffix('.bin.part').exists())

    def test_partial_range_resume(self):
        self.path.with_suffix('.bin.part').write_bytes(PAYLOAD[:17])
        module.download(self.asset, self.path, self.url)
        self.assertEqual(Handler.requests, ['bytes=17-'])
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_bad_digest_never_publishes(self):
        self.asset['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            module.download(self.asset, self.path, self.url)
        self.assertFalse(self.path.exists())
        self.assertTrue(self.path.with_suffix('.bin.part').exists())

    def test_ignored_range_restarts_instead_of_appending(self):
        self.path.with_suffix('.bin.part').write_bytes(PAYLOAD[:17])
        module.download(self.asset, self.path, self.url.replace('/model', '/ignore-range'))
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_complete_partial_is_published_without_network(self):
        self.path.with_suffix('.bin.part').write_bytes(PAYLOAD)
        self.assertEqual(module.download(self.asset, self.path, self.url), 'verified-partial')
        self.assertFalse(Handler.requests)

    def test_existing_mismatch_untouched(self):
        self.path.write_bytes(b'other')
        with self.assertRaises(ValueError):
            module.download(self.asset, self.path, self.url)
        self.assertEqual(self.path.read_bytes(), b'other')
        self.assertFalse(Handler.requests)

    def test_reuse_verified_named_file_only(self):
        source = Path(self.temp.name)/'reuse'
        source.mkdir()
        (source/'model.bin').write_bytes(PAYLOAD)
        self.assertIsNotNone(module.reuse(self.asset, self.path, [source]))
        self.assertTrue(self.path.is_symlink())
        self.assertEqual(self.path.resolve(), source/'model.bin')

    def test_manifest_path_traversal_rejected(self):
        for value in ['../secret', '/absolute', 'dir\\escape']:
            with self.assertRaises(ValueError):
                module.relative_path(value)


if __name__ == '__main__':
    unittest.main()

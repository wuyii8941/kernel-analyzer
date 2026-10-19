import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_archiver_under_test', ROOT/'tools/fetch_sources.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class SourceArchiving(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT.parent/'.cache/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = self.root/'sources'
        self.store.mkdir()
        self.index = self.root/'SOURCES.json'
        self.index.write_text('{}')
        self.todo = self.root/'todo.json'
        for name, value in (('ROOT', self.root), ('STORE', self.store), ('SOURCES', self.index)):
            patcher = patch.object(archive, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def invoke(self, entries, *options):
        self.todo.write_text(json.dumps(entries))
        with patch.object(sys, 'argv', ['fetch_sources.py', str(self.todo), *options]):
            return archive.main()

    def test_new_url_archive_uses_requested_proxy_and_no_hash(self):
        raw = b'def task(x):\n    return x\n'
        with patch.object(archive, 'fetch', return_value=raw) as fetch:
            self.assertEqual(self.invoke([dict(sid='NEW', url='https://example.org/task.py', version='v1', items={})]), 0)
        fetch.assert_called_once_with('https://example.org/task.py', 'http://127.0.0.1:27891')
        result = json.loads(self.index.read_text())['NEW']
        self.assertFalse(any('sha' in k.lower() or 'hash' in k.lower() for k in result))
        self.assertEqual(archive.gzip.decompress((self.root/result['snapshot']).read_bytes()), raw)

    def test_existing_snapshot_cannot_be_replaced_even_if_unregistered(self):
        existing = self.store/'NEW.py.gz'
        existing.write_bytes(b'original bytes')
        with patch.object(archive, 'fetch', return_value=b'new bytes'):
            result = self.invoke([dict(sid='NEW', url='https://example.org/task.py', version='v1')])
        self.assertNotEqual(result, 0)
        self.assertEqual(existing.read_bytes(), b'original bytes')
        self.assertEqual(json.loads(self.index.read_text()), {})

    def test_replace_and_resnapshot_cannot_overwrite_frozen_material(self):
        original = {'OLD': {'url': 'https://example.org/task.py', 'snapshot': 'sources/OLD.py.gz', 'version': 'v1'}}
        self.index.write_text(json.dumps(original))
        (self.store/'OLD.py.gz').write_bytes(b'original')
        for flag, entries in (('--replace', [dict(sid='OLD', url='https://example.org/task.py')]),
                              ('--resnapshot', ['OLD'])):
            with patch.object(archive, 'fetch', return_value=b'new') as fetch:
                self.assertNotEqual(self.invoke(entries, flag), 0)
                fetch.assert_not_called()
            self.assertEqual(json.loads(self.index.read_text()), original)
            self.assertEqual((self.store/'OLD.py.gz').read_bytes(), b'original')

    def test_annotation_update_cannot_relabel_source_version(self):
        original = {'OLD': {'url': 'https://example.org/v1/task.py', 'version': 'v1', 'items': {'A': 'a'}}}
        self.index.write_text(json.dumps(original))
        self.assertNotEqual(self.invoke([dict(sid='OLD', version='v2', items={'B': 'b'})], '--items-only'), 0)
        self.assertEqual(json.loads(self.index.read_text()), original)
        self.assertEqual(self.invoke([dict(sid='OLD', items={'B': 'b'})], '--items-only'), 0)
        self.assertEqual(json.loads(self.index.read_text())['OLD']['items'], {'A': 'a', 'B': 'b'})

    def test_existing_source_with_changed_identity_is_not_a_successful_skip(self):
        original = {'OLD': {'url': 'https://example.org/v1/task.py', 'version': 'v1'}}
        self.index.write_text(json.dumps(original))
        before = self.index.read_bytes()
        with patch.object(archive, 'fetch') as fetch:
            self.assertNotEqual(self.invoke([dict(sid='OLD', version='v2')]), 0)
            fetch.assert_not_called()
        self.assertEqual(self.index.read_bytes(), before)

    def test_network_failure_has_no_direct_fallback_or_false_registration(self):
        with patch.object(archive, 'subprocess') as process:
            process.run.return_value.returncode = 7
            process.run.return_value.stderr = b'proxy unavailable'
            process.run.return_value.stdout = b''
            self.assertNotEqual(self.invoke([dict(sid='NEW', url='https://example.org/task.py')]), 0)
            self.assertEqual(process.run.call_count, 1)
            command = process.run.call_args.args[0]
            self.assertEqual(command[command.index('-x')+1], 'http://127.0.0.1:27891')
            self.assertEqual(command[command.index('--noproxy')+1], '')
        self.assertEqual(list(self.store.iterdir()), [])
        self.assertEqual(json.loads(self.index.read_text()), {})

    def test_source_id_cannot_reuse_another_extension_or_escape_store(self):
        (self.store/'NEW.txt.gz').write_bytes(b'old')
        with self.assertRaises(FileExistsError):
            archive.save_snapshot('NEW', '.py', b'new')
        for sid in ('../escape', '/absolute', ''):
            with self.assertRaises(ValueError):
                archive.save_snapshot(sid, '.py', b'new')

    def test_report_distinguishes_retention_from_version_verification(self):
        spec = importlib.util.spec_from_file_location('source_status_under_test', ROOT/'tools/check_contracts.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        data = {'clauses': [{'basis': ['LOCAL.ITEM']}]}
        sources = {'LOCAL': {'snapshot': 'sources/VLLM_UA.py.gz', 'snapshot_sha256': 'legacy-display-must-not-appear',
                             'version_review': {'status': 'unverified'}}}
        self.assertEqual(checker.source_status(data, sources),
                         {'LOCAL': 'retained local source; claimed version unverified'})


if __name__ == '__main__':
    unittest.main()

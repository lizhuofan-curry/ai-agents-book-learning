"""冻结证据在Git检出后仍须精确对应原始哈希，不写回证据文件。"""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'第三章/3.7结构化索引/我的代码/06_依据原文修订与查询.py'
spec = importlib.util.spec_from_file_location('frozen_evidence_checks', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrozenEvidenceLineEndingChecks(unittest.TestCase):
    def verify(self, actual, original):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            path.write_bytes(actual)
            expected = hashlib.sha256(original).hexdigest()
            accepted = module.matches_frozen_sha(path, expected)
            self.assertEqual(path.read_bytes(), actual)
            return accepted

    def test_original_and_git_lf_copy_match_same_frozen_hash(self):
        original = '{\r\n  "事实": "原文",\r\n  "value": 7\r\n}\r\n'.encode('utf-8')
        self.assertTrue(self.verify(original, original))
        self.assertTrue(self.verify(original.replace(b'\r\n', b'\n'), original))

    def test_changed_content_or_spacing_is_rejected(self):
        original = b'{\r\n  "value": 7\r\n}\r\n'
        normalized = original.replace(b'\r\n', b'\n')
        for changed in (normalized.replace(b'7', b'8'), normalized.replace(b'  ', b' '), normalized.rstrip(b'\n')):
            with self.subTest(changed=changed):
                self.assertFalse(self.verify(changed, original))

    def test_mixed_or_lone_cr_is_not_normalized(self):
        original = b'{\r\n  "value": 7\r\n}\r\n'
        self.assertFalse(self.verify(b'{\n  "value": 7\r\n}\n', original))
        self.assertFalse(self.verify(original.replace(b'\r\n', b'\r'), original))

    def test_native_lf_hash_remains_strict(self):
        original = b'{\n  "value": 7\n}\n'
        self.assertTrue(self.verify(original, original))
        self.assertFalse(self.verify(original.replace(b'\n', b'\r\n'), original))


if __name__ == '__main__':
    unittest.main()

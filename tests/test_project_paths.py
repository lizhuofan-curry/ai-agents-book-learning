"""验证旧记录定位、歧义和越界；不改写历史文件或请求网络。"""
from pathlib import Path
import tempfile
import unittest

from project_paths import path_key_value, project_path


class ProjectPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'AI_Agents_Book'
        self.root.mkdir()
        self.root = self.root.resolve()

    def test_windows_and_posix_point_to_same_file(self):
        relative = '第三章/3.4稠密向量检索/运行结果/record.json'
        expected = self.root / relative
        for value in (relative, relative.replace('/', '\\'),
                      'E:/AI_Agents_Book/' + relative,
                      'E:\\AI_Agents_Book\\' + relative.replace('/', '\\'),
                      str(expected)):
            self.assertEqual(project_path(value, root=self.root), expected)

    def test_rejects_path_traversal(self):
        for value in ('../other', '..\\other', 'E:/AI_Agents_Book/../other',
                      'E:/other/file.json', 'E:AI_Agents_Book/file.json'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project_path(value, root=self.root)

    def test_rejects_symlink_escape(self):
        outside = Path(self.temporary.name) / 'outside'
        outside.mkdir()
        try:
            (self.root / 'link').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('OS does not permit test symlinks')
        with self.assertRaises(ValueError):
            project_path('link/file.json', root=self.root)

    def test_rejects_ambiguous_project_anchor(self):
        with self.assertRaises(ValueError):
            project_path('E:/AI_Agents_Book/AI_Agents_Book/file', root=self.root)

    def test_windows_hash_keys(self):
        self.assertEqual(path_key_value({'第三章\\source.py': 'hash'}, '第三章/source.py'), 'hash')

    def test_rejects_missing_or_ambiguous_hash_keys(self):
        for mapping in ({}, {'a/b': 'x', 'a\\b': 'x'}):
            with self.assertRaises(ValueError):
                path_key_value(mapping, 'a/b')


if __name__ == '__main__':
    unittest.main()

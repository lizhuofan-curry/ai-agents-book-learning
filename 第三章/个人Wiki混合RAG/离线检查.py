"""BM25数值、中文切词、原文跨度及文件范围检查；仅使用内存与临时示例。"""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wiki_sparse', ROOT/'我的代码/01_固定Wiki语料与关键词检索.py')
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


class SparseChecks(unittest.TestCase):
    def test_english_casefold_and_chinese_character_terms(self):
        tokens = step.tokenize('Agent 脑机接口')
        self.assertEqual(tokens, ['agent', '脑', '机', '接', '口', '脑机', '机接', '接口'])

    def test_question_boilerplate_removed_without_rewriting_topic(self):
        for original, expected in [('帮我解释什么是Agent', 'Agent'), ('transformer是什么', 'transformer'),
                                    ('什么是脑机接口', '脑机接口')]:
            self.assertEqual(step.normalize_query(original), expected)

    def test_bm25_hand_calculation_and_no_match(self):
        chunks = [{'id': f'D{i}:C1', 'document_id': f'D{i}', 'path': f'wiki/concepts/{i}.md',
                   'source_line_start': 1, 'source_line_end': 1, 'text': text}
                  for i, text in enumerate(['agent agent', 'eeg', 'agent'], 1)]
        result = step.rank_bm25(chunks, [{'id': 'Q1', 'text': 'AGENT'}, {'id': 'Q2', 'text': 'unknown'}], 1.2, .75, 5)
        self.assertEqual([h['document_id'] for h in result[0]['hits']], ['D1', 'D3'])
        expected = math.log(1+(3-2+.5)/(2+.5))*2*2.2/(2+1.2*(.25+.75*2/(4/3)))
        self.assertAlmostEqual(result[0]['hits'][0]['score'], expected)
        self.assertEqual(result[1]['hits'], [])

    def test_frontmatter_excluded_and_spans_reproduce_text(self):
        text = '---\r\ntitle: private metadata\r\n---\r\n# 标题\r\n这是实际正文abcdef123456。'
        chunks = step.split_document('D1', 'wiki/concepts/example.md', text, 10, 3)
        self.assertTrue(chunks)
        self.assertTrue(chunks[0]['text'].startswith('# 标题'))
        self.assertNotIn('private metadata', ''.join(c['text'] for c in chunks))
        for chunk in chunks:
            self.assertEqual(chunk['text'], text[chunk['source_char_start']:chunk['source_char_end']])
            self.assertLessEqual(len(chunk['text']), 10)
        self.assertEqual(chunks[0]['source_line_start'], 4)
        with self.assertRaises(ValueError):
            step.split_document('D1', 'unused', text, 10, 10)

    def test_only_knowledge_pages_within_vault_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            vault = Path(folder).resolve()
            self.assertEqual(step.source_path(vault, 'wiki/concepts/hello.md'), vault/'wiki/concepts/hello.md')
            for relative in ['../outside.md', '/tmp/outside.md', 'raw/private.md', 'wiki/outputs/report.md', 'wiki/index.md',
                             'wiki/concepts/../outputs/report.md', 'wiki/concepts/../index.md']:
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    step.source_path(vault, relative)


if __name__ == '__main__':
    unittest.main(verbosity=2)

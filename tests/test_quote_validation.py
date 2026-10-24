"""离线检查引用边界；不运行模型，不修改历史输出，不要求索引依赖。"""
import ast
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def load_pure_functions(relative, names):
    """执行源码中的纯函数，避免为引用测试导入networkx等可选运行依赖。"""
    path = ROOT / relative
    parsed = ast.parse(path.read_text(encoding='utf-8-sig'))
    definitions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in definitions} != set(names):
        raise AssertionError('引用函数缺失：' + relative)
    namespace = {'re': re}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class MemoryQuoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module('第三章/3.1用户记忆评估/我的代码/01_提取并保存长期记忆.py', 'quote_memory')

    def notes(self, quote):
        return json.dumps({'notes': [{'text': '代码示例优先用 Python', 'source_quote': quote}]}, ensure_ascii=False)

    def test_paraphrased_note_can_keep_exact_quote(self):
        result = self.module.validate_notes(self.notes('代码示例优先用Python'))
        self.assertEqual(result[0]['text'], '代码示例优先用 Python')

    def test_inserted_space_is_rejected_and_identified(self):
        with self.assertRaisesRegex(ValueError, '第1条source_quote'):
            self.module.validate_notes(self.notes('代码示例优先用 Python'))

    def test_assistant_text_is_not_user_evidence(self):
        with self.assertRaises(ValueError):
            self.module.validate_notes(self.notes('请提供快递单号'))

    def test_empty_notes_follow_the_prompt_contract(self):
        self.assertEqual(self.module.validate_notes('{"notes":[]}'), [])
        with self.assertRaises(ValueError):
            self.module.validate_notes('{"notes":null}')

    def test_case_and_punctuation_are_not_silently_fixed(self):
        for quote in ('代码示例优先用python', '代码示例优先用Python！'):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.module.validate_notes(self.notes(quote))


class MemorySourceSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module('第三章/3.1用户记忆评估/我的代码/01_提取并保存长期记忆.py', 'source_id_memory')

    def model_content(self):
        return json.dumps({'notes': [
            {'text': '用户名为林夏', 'source_id': 'u1_s1'},
            {'text': '用户偏好用中文回答', 'source_id': 'u1_s2'},
            {'text': '用户偏好代码示例优先用 Python', 'source_id': 'u1_s3'},
        ]}, ensure_ascii=False)

    def test_catalog_keeps_all_user_clauses_including_cancelled_tasks(self):
        spans = self.module.build_source_spans()
        self.assertEqual(len(spans), 6)
        self.assertIn('今天下午帮我查一下快递', [s['quote'] for s in spans])
        self.assertIn('不用查了', [s['quote'] for s in spans])
        for span in spans:
            message = self.module.CONVERSATION[span['message_index']]
            self.assertEqual(message['role'], 'user')
            self.assertEqual(message['content'][span['char_start']:span['char_end']], span['quote'])

    def test_schema_asks_for_ids_instead_of_free_form_quotes(self):
        request = self.module.build_request()
        fields = request['format']['properties']['notes']['items']['properties']
        self.assertEqual(set(fields), {'text', 'source_id'})
        self.assertIn('u1_s4', fields['source_id']['enum'])
        self.assertIn('u3_s2', fields['source_id']['enum'])

    def test_program_copies_exact_quote_while_note_may_be_paraphrased(self):
        raw = self.model_content()
        result = self.module.resolve_notes(raw, self.module.build_source_spans())
        self.assertEqual(result[2]['text'], '用户偏好代码示例优先用 Python')
        self.assertEqual(result[2]['source_quote'], '代码示例优先用Python')
        self.assertEqual(result[2]['quote_origin'], 'program_copied_from_user_message')
        self.assertNotIn('source_quote', json.loads(raw)['notes'][2])

    def test_unknown_id_and_legacy_quote_output_are_rejected(self):
        for raw in (self.model_content().replace('u1_s3', 'unknown'),
                    '{"notes":[{"text":"偏好","source_quote":"代码示例优先用 Python"}]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.module.resolve_notes(raw, self.module.build_source_spans())

    def test_tampered_catalog_is_rejected(self):
        spans = self.module.build_source_spans()
        spans[2]['quote'] = '代码示例优先用 Python'
        with self.assertRaises(ValueError):
            self.module.resolve_notes(self.model_content(), spans)

    def test_empty_selection_remains_valid(self):
        self.assertEqual(self.module.resolve_notes('{"notes":[]}', self.module.build_source_spans()), [])

    def test_save_flow_preserves_raw_response_and_records_quote_origin(self):
        raw_content = self.model_content()
        response = {'model': self.module.MODEL, 'done': True, 'done_reason': 'stop',
                    'message': {'content': raw_content}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(self.module, 'ROOT', root), patch.object(self.module.requests, 'Session') as factory, patch('sys.stdout', new_callable=io.StringIO):
                http_response = factory.return_value.__enter__.return_value.post.return_value
                http_response.status_code = 200
                http_response.text = json.dumps(response, ensure_ascii=False)
                http_response.json.return_value = response
                self.module.main()
                factory.return_value.__enter__.return_value.post.assert_called_once()
            saved = next((root / '运行结果').glob('*/record.json'))
            record = json.loads(saved.read_text(encoding='utf-8'))
            memory = json.loads((saved.parent / 'memory.json').read_text(encoding='utf-8'))
            self.assertEqual(record['raw_content'], raw_content)
            self.assertEqual(record['raw_response'], response)
            self.assertEqual(record['extraction_protocol'], 'source_span_ids_v2')
            self.assertTrue(record['memory_saved'])
            self.assertEqual(record['parsed_notes'][2]['source_quote'], '代码示例优先用Python')
            self.assertEqual(record['checks']['source_quotes_copied_by'], 'program')
            self.assertEqual(len(memory['notes']), 3)


class LogSpanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module('第三章/3.3日志脱敏/我的代码/01_单条日志规则与模型对照.py', 'quote_log')

    def test_exact_sensitive_value_and_rewritten_value(self):
        text = 'Synthetic password=abCD1234; diagnostic=OK'
        exact = json.dumps({'pii_items': [{'type': 'password', 'value': 'abCD1234'}]})
        items = self.module.parse_items(exact, text)
        spans = self.module.item_spans(items, text)
        self.assertEqual(self.module.mask_spans(text, spans), 'Synthetic password=[REDACTED]; diagnostic=OK')
        altered = exact.replace('abCD1234', 'abCD 1234')
        with self.assertRaises(ValueError):
            self.module.parse_items(altered, text)


class StructureQuoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extract = load_pure_functions('第三章/3.7结构化索引/我的代码/03_原文抽取建图与摘要.py', ['normalize', 'locate_quote'])
        cls.review = load_pure_functions('第三章/3.7结构化索引/我的代码/06_依据原文修订与查询.py', ['anchor'])

    def chunk(self):
        return {'chunk_id': 'synthetic', 'source_char_start': 100,
                'raw_text': '\n  SIMD\t\nuses XMM registers.\nSIMD uses XMM registers.  '}

    def test_normalized_quotes_retain_both_raw_offsets(self):
        chunk = self.chunk()
        quote = 'SIMD uses XMM registers.'
        extracted = self.extract.locate_quote(chunk, quote)
        reviewed = self.review.anchor({'chunk_id': 'synthetic', 'quote': quote}, {'synthetic': chunk})
        self.assertEqual(extracted, reviewed)
        self.assertEqual(len(extracted), 2)
        for citation in extracted:
            start = citation['source_char_start'] - chunk['source_char_start']
            end = citation['source_char_end'] - chunk['source_char_start']
            self.assertEqual(chunk['raw_text'][start:end], citation['raw_source_quote'])
            self.assertEqual(' '.join(citation['raw_source_quote'].split()), quote)

    def test_spacing_inside_a_word_does_not_pass(self):
        chunk = self.chunk()
        quote = 'SIMD uses X MM registers.'
        self.assertEqual(self.extract.locate_quote(chunk, quote), [])
        with self.assertRaises(ValueError):
            self.review.anchor({'chunk_id': 'synthetic', 'quote': quote}, {'synthetic': chunk})


class WebQuoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module('第二章/2.10上下文压缩策略对比/我的代码/10_DeepSeek真实网页六策略对照.py', 'quote_web')

    def example(self, quote='ExceptionGroup preserves original exception instances.'):
        text = 'ExceptionGroup\n  preserves original exception instances.'
        sources = [{'id': name, 'text': text} for name in ('S1', 'S2', 'S3')]
        rows = [{'id': key, 'value': value, 'source_id': sorted(allowed)[0], 'quote': quote}
                for key, (value, allowed) in self.module.GOLD.items()]
        return {'answers': rows}, sources

    def test_line_wrapping_is_consistently_normalized(self):
        data, sources = self.example()
        result = self.module.check_answer(json.dumps(data), sources)
        self.assertTrue(result['json_structure_valid'])
        self.assertEqual(result['original_quote_matches'], 6)

    def test_adding_or_removing_word_boundaries_is_rejected(self):
        data, sources = self.example('Exception Group preserves original exception instances.')
        result = self.module.check_answer(json.dumps(data), sources)
        self.assertEqual(result['original_quote_matches'], 0)

    def test_too_short_quote_or_wrong_source_cannot_pass(self):
        data, sources = self.example('ExceptionGroup')
        result = self.module.check_answer(json.dumps(data), sources)
        self.assertEqual(result['original_quote_matches'], 0)
        data, sources = self.example()
        altered = deepcopy(data)
        for row in altered['answers']:
            row['source_id'] = 'missing'
        self.assertEqual(self.module.check_answer(json.dumps(altered), sources)['original_quote_matches'], 0)


if __name__ == '__main__':
    unittest.main()

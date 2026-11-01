"""语义检索的向量、排序、预算和冻结输入校验；不请求模型服务或运行编码前向。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dense_step', ROOT/'我的代码/03_同一快照的中文语义检索.py')
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


class DenseChecks(unittest.TestCase):
    def test_casefold_only_changes_model_inputs_and_keeps_original_text(self):
        chunks = [{'text': 'Agent与Transformer是不同概念。'}]
        queries = [{'retrieval_query': 'Agent'}]
        documents, questions = step.model_inputs(chunks, queries, '检索：', casefold=True)
        self.assertEqual(documents, ['agent与transformer是不同概念。'])
        self.assertEqual(questions, ['检索：agent'])
        self.assertEqual(chunks[0]['text'], 'Agent与Transformer是不同概念。')
        self.assertEqual(queries[0]['retrieval_query'], 'Agent')
        raw, _ = step.model_inputs(chunks, queries, '', casefold=False)
        self.assertEqual(raw, [chunks[0]['text']])

    def test_cosine_ranking_and_stable_ties(self):
        chunks = [{'id': name, 'document_id': name, 'path': name+'.md', 'source_line_start': 1,
                   'source_line_end': 1, 'text': name} for name in ['D03', 'D01', 'D02']]
        queries = [{'query_id': 'Q1', 'query': '示例', 'retrieval_query': '示例'}]
        documents = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        vectors = np.array([[1, 0]], dtype=np.float32)
        result = step.rank_dense(chunks, queries, documents, vectors, 3)[0]
        self.assertEqual([h['chunk_id'] for h in result['hits']], ['D01', 'D03', 'D02'])
        self.assertEqual([h['score'] for h in result['hits']], [1.0, 1.0, 0.0])

    def test_wrong_dtype_shape_nan_and_zero_norm_rejected(self):
        valid = np.zeros((2, 512), dtype=np.float32)
        valid[:, 0] = 1
        step.validate_vectors(valid, 2)
        invalid = [valid.astype(np.float64), valid[:, :511], np.zeros_like(valid)]
        nan = valid.copy()
        nan[0, 0] = np.nan
        invalid.append(nan)
        for value in invalid:
            with self.assertRaises(ValueError):
                step.validate_vectors(value, 2)

    def test_budget_stops_before_model_forward(self):
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return {'input_ids': [[1, 2]]}
        def forbidden_model(*args, **kwargs):
            raise AssertionError('不应执行前向')
        record = {'embedding_forward_batches': 1, 'batches': []}
        with self.assertRaises(RuntimeError):
            step.encode(['合成'], ['S1'], 'synthetic', Tokenizer(), forbidden_model, record, lambda: None, 8, 1)
        self.assertEqual(record['embedding_forward_batches'], 1)
        self.assertEqual(record['batches'], [])

    def test_changed_snapshot_and_parent_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/'本地运行结果/wiki_sparse_fake'
            source.mkdir(parents=True)
            (source/'record.json').write_text('{}')
            (source/'corpus.json').write_text('{}')
            plan = root/'input.json'
            plan.write_text(json.dumps({'source_run': 'wiki_sparse_fake', 'source_record_sha256': 'wrong',
                                        'source_corpus_sha256': 'wrong'}))
            with patch.object(step, 'ROOT', root), patch.object(step, 'INPUT_PLAN', plan):
                with self.assertRaisesRegex(ValueError, '已变化'):
                    step.load_snapshot()
                plan.write_text(json.dumps({'source_run': '../wiki_sparse_escape'}))
                with self.assertRaisesRegex(ValueError, '越界'):
                    step.load_snapshot()


if __name__ == '__main__':
    unittest.main(verbosity=2)

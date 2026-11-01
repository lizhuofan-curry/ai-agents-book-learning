"""官方RRF手算、同分、原分数量纲和非法输入检查；不读取Wiki正文或请求模型。"""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wiki_fusion', ROOT/'我的代码/04_固定两路排名做RRF融合.py')
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def hit(name, score):
    return {'chunk_id': name, 'score': score}


class FusionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rrf, cls.reference = step.load_rrf()
        cls.rrf = staticmethod(rrf)

    def test_shared_hit_adds_two_rank_contributions(self):
        rows = step.fuse_rankings([hit('A', .9), hit('B', .8)], [hit('B', 100), hit('C', 50)], self.rrf)
        self.assertEqual([row['chunk_id'] for row in rows], ['B', 'A', 'C'])
        self.assertAlmostEqual(rows[0]['rrf_score'], 1/62+1/61)
        self.assertEqual(rows[0]['dense_rank'], 2)
        self.assertEqual(rows[0]['sparse_rank'], 1)

    def test_original_score_scale_does_not_change_fusion(self):
        first = step.fuse_rankings([hit('A', .9), hit('B', .8)], [hit('B', 20), hit('C', 10)], self.rrf)
        second = step.fuse_rankings([hit('A', 900), hit('B', -1)], [hit('B', .002), hit('C', .001)], self.rrf)
        self.assertEqual(first, second)

    def test_tie_order_and_missing_source_zero(self):
        rows = step.fuse_rankings([hit('A', 1)], [hit('B', 2)], self.rrf)
        self.assertEqual([row['chunk_id'] for row in rows], ['A', 'B'])
        self.assertEqual(rows[0]['contributions']['sparse'], 0)
        self.assertEqual(rows[1]['contributions']['dense'], 0)

    def test_empty_source_preserves_other_candidates(self):
        rows = step.fuse_rankings([], [hit('B', 20)], self.rrf)
        self.assertEqual([row['chunk_id'] for row in rows], ['B'])
        self.assertAlmostEqual(rows[0]['rrf_score'], 1/61)
        self.assertEqual(step.fuse_rankings([], [], self.rrf), [])

    def test_duplicate_unknown_and_invalid_scores_rejected(self):
        chunk = {'id': 'A', 'document_id': 'D1', 'path': 'wiki/concepts/a.md', 'text': '正文',
                 'source_line_start': 1, 'source_line_end': 1}
        full = {'chunk_id': 'A', 'score': 1, **{k: v for k, v in chunk.items() if k != 'id'}}
        step.validate_hits([full], {'A': chunk})
        for hits in [[full, full], [{**full, 'chunk_id': 'missing'}],
                     [{**full, 'score': float('nan')}], [{**full, 'score': True}],
                     [{**full, 'text': '伪造片段'}]]:
            with self.assertRaises(ValueError):
                step.validate_hits(hits, {'A': chunk})


if __name__ == '__main__':
    unittest.main(verbosity=2)

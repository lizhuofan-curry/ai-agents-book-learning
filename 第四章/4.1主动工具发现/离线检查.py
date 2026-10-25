import ast
import importlib.util
import unittest
from pathlib import Path

P = Path(__file__).resolve().parent/'我的代码/01_阅读工具目录与按需展示.py'
spec = importlib.util.spec_from_file_location('catalog_reading', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CatalogChecks(unittest.TestCase):
    def test_frozen_catalog_and_visibility(self):
        views, _ = m.prepare()
        self.assertEqual(len(views['full']), 126)
        self.assertEqual(len(views['initial']), 3)
        self.assertEqual(len(views['after_manual_demo']), 5)
        initial = {t['function']['name'] for t in views['initial']}
        self.assertTrue(initial.isdisjoint(m.MANUAL_NAMES))
        by = {t['function']['name']: t for t in views['full']}
        self.assertEqual(views['manual_candidates'], [by[n] for n in m.MANUAL_NAMES])

    def test_untrusted_call_not_evaluated(self):
        for code in ("__import__('os')", 'open("x")', '[x for x in y]'):
            with self.assertRaises(ValueError):
                m.evaluate(ast.parse(code, mode='eval').body)

    def test_definition_metrics_have_explicit_units(self):
        metrics = m.metrics([{'中文': '工具'}])
        self.assertEqual(metrics['tool_count'], 1)
        self.assertGreater(metrics['utf8_bytes'], metrics['json_characters'])
        self.assertNotIn('tokens', metrics)


if __name__ == '__main__':
    unittest.main()

"""离线检查；模拟响应不代表正式模型实验。"""
import importlib.util
import json
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/'第三章/3.7结构化索引/我的代码/09_社区发现与递归摘要.py'
spec = importlib.util.spec_from_file_location('community_step', PATH)
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


class FakeResponse:
    status_code = 200
    def __init__(self, raw):
        self.raw = raw
        self.text = json.dumps(raw)
    def raise_for_status(self):
        pass
    def json(self):
        return self.raw


class FakeSession:
    def __init__(self, raw):
        self.raw = raw
        self.count = 0
    def post(self, *args, **kwargs):
        self.count += 1
        return FakeResponse(self.raw)


class CommunityChecks(unittest.TestCase):
    def test_frozen_sources_and_edge_coverage(self):
        _, _, leaves, tasks, original, groups = step.prepare()
        assigned = [leaf['id'] for task in tasks for leaf in task['inputs']]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {n['id'] for n in leaves})
        self.assertEqual(len(original), 11)
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(tasks)+1, 5)

    def test_edgeless_and_isolated_nodes(self):
        view = {'nodes': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}], 'edges': []}
        self.assertEqual(step.communities(view), [['a'], ['b'], ['c']])
        view['edges'] = [{'source': 'a', 'target': 'b'}]
        groups = step.communities(view)
        self.assertEqual({frozenset(g) for g in groups}, {frozenset('ab'), frozenset('c')})

    def test_reference_rejection(self):
        for refs in ([], ['missing'], ['e1', 'e1']):
            with self.assertRaises(ValueError):
                step.checked_summary({'claims': [{'text': 'x', 'source_ids': refs}]}, {'e1'})

    def test_recursive_provenance_and_cycle(self):
        nodes = {'e': {'id': 'e', 'level': 0},
                 'c': {'level': 1, 'claims': [{'source_ids': ['e']}]},
                 'root': {'level': 2, 'claims': [{'source_ids': ['c']}]}}
        self.assertEqual(step.expand('root', nodes), ['e'])
        nodes['c']['claims'][0]['source_ids'] = ['root']
        with self.assertRaises(ValueError):
            step.expand('root', nodes)

    def test_model_response_and_truncation_no_retry(self):
        raw = {'model': 'test', 'done': True, 'done_reason': 'stop',
               'message': {'content': json.dumps({'claims': [{'text': 'x', 'source_ids': ['e']}]})}}
        session = FakeSession(raw)
        record = {'calls': []}
        self.assertEqual(len(step.summarize(session, 'test', {'e': 'x'}, record, lambda: None)), 1)
        raw['done_reason'] = 'length'
        session = FakeSession(raw)
        record = {'calls': []}
        with self.assertRaises(ValueError):
            step.summarize(session, 'test', {'e': 'x'}, record, lambda: None)
        self.assertEqual(session.count, 1)
        self.assertIn('raw_http_body', record['calls'][0])

    def test_malformed_and_fabricated_reference_saved(self):
        for content in ('{', '{"claims":[{"text":"x","source_ids":["invented"]}]}'):
            record = {'calls': []}
            session = FakeSession({'model': 'test', 'done': True, 'done_reason': 'stop',
                                   'message': {'content': content}})
            with self.assertRaises(ValueError):
                step.summarize(session, 'test', {'e': 'x'}, record, lambda: None)
            self.assertEqual(session.count, 1)
            self.assertIn('raw_http_body', record['calls'][0])


if __name__ == '__main__':
    unittest.main()

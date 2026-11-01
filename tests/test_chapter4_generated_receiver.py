"""独立接收方输入隔离与规则核验；假响应，不新增真实模型调用。"""
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'generated_receiver', ROOT/'第四章/4.5协作工具/我的代码/15_模型交接稿接收与核验.py')
step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(step)


def response(decision='eligible', ids=None, reason='stop'):
    answer = {'decision': decision, 'reason': '合成测试理由，不证明语义正确',
              'evidence_ids': ids if ids is not None else ['H2','H4','H5','H6'], 'missing_information': []}
    return {'model': step.transport.MODEL, 'done': True, 'done_reason': reason,
            'message': {'role': 'assistant', 'content': json.dumps(answer,ensure_ascii=False)},
            'prompt_eval_count': 100, 'eval_count': 50}


class GeneratedReceiverChecks(unittest.TestCase):
    def setUp(self):
        self.generated, self.facts = step.prepare()

    def record(self):
        return {'handoff_as_received': deepcopy(self.generated['handoff']), 'original_rule_facts': deepcopy(self.facts)}

    def test_payload_exact_handoff_without_original_sources_or_expected_answer(self):
        payload = step.make_payload(self.generated['handoff'])
        content = payload['messages'][1]['content']
        self.assertEqual([m['role'] for m in payload['messages']], ['system','user'])
        self.assertTrue(content.endswith(json.dumps(self.generated['handoff'],ensure_ascii=False)))
        self.assertNotIn('TRAIN-07', content)
        self.assertNotIn('expected_decision', content)
        self.assertNotIn('source_messages', content)
        self.assertNotIn('12 天', content)

    def test_one_call_and_immutable_handoff(self):
        record = self.record()
        before = deepcopy(record['handoff_as_received'])
        calls = []
        def fake(payload, entry, save):
            calls.append(payload)
            return response()
        with contextlib.redirect_stdout(io.StringIO()):
            step.receive(record,lambda:None,fake)
        self.assertEqual(len(calls),1)
        self.assertEqual(record['handoff_as_received'],before)
        self.assertEqual(record['decision_verification']['gate'],'decision_verified_reason_review_pending')
        self.assertFalse(record['decision_verification']['can_execute_action'])
        self.assertFalse(record['decision_verification']['can_forward_full_result'])

    def test_wrong_decision_blocked_using_original_rule(self):
        answer = {'decision':'ineligible','reason':'合成','evidence_ids':['H2','H4','H5','H6'],'missing_information':[]}
        checked = step.check_decision(answer,self.generated['handoff'],self.facts)
        self.assertEqual(checked['gate'],'blocked_decision_mismatch')
        self.assertEqual(answer['decision'],'ineligible')

    def test_missing_supporting_citations_blocked(self):
        answer = {'decision':'eligible','reason':'合成','evidence_ids':['H2'],'missing_information':[]}
        self.assertEqual(step.check_decision(answer,self.generated['handoff'],self.facts)['gate'],'blocked_missing_supporting_citations')

    def test_unseen_source_reference_rejected_with_raw_response_kept(self):
        record = self.record()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            step.receive(record,lambda:None,lambda *args:response(ids=['S2']))
        self.assertEqual(record['status'],'incomplete')
        self.assertIn('response',record['receiver_call'])

    def test_truncation_no_retry(self):
        record = self.record()
        calls = []
        def fake(*args):
            calls.append(1)
            return response(reason='length')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            step.receive(record,lambda:None,fake)
        self.assertEqual(len(calls),1)
        self.assertEqual(record['receiver_call']['response']['done_reason'],'length')

    def test_preview_no_model_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(step,'ROOT',Path(directory)), patch.object(step,'prepare',return_value=(self.generated,self.facts)):
                with patch.object(step,'receive') as receive, patch('sys.argv',['receiver','--preview']), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(step.main(),0)
                    receive.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()),[])


class GeneratedReceiverRelocationChecks(unittest.TestCase):
    """在不同目录读取冻结记录；不得回退到作者机器或跳过证据校验。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)/'另一份副本'
        self.base = self.root/'运行结果/handoff_within_window_20261101_150027_129403/record.json'
        self.generation = self.root/step.GENERATION.relative_to(step.ROOT)
        self.review = self.generation.parent/step.REVIEW.name
        sources = [step.ROOT/self.base.relative_to(self.root), step.GENERATION, step.REVIEW]
        self.original_bytes = [(source, source.read_bytes()) for source in sources]
        for source, target in zip(sources, [self.base, self.generation, self.review]):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name, value in [('ROOT', self.root), ('GENERATION', self.generation), ('REVIEW', self.review)]:
            patcher = patch.object(step, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def change_reference(self, reference):
        # 仅修改临时夹具，重新绑定核对意见；仓库冻结记录不变。
        generated = json.loads(self.generation.read_text(encoding='utf-8'))
        generated['base_record'] = reference
        self.generation.write_text(json.dumps(generated, ensure_ascii=False), encoding='utf-8')
        review = json.loads(self.review.read_text(encoding='utf-8'))
        review['source_record_sha256'] = step.sha(self.generation)
        self.review.write_text(json.dumps(review, ensure_ascii=False), encoding='utf-8')

    def test_relocated_checkout_reads_local_evidence_without_rewriting_sources(self):
        before = {path: path.read_bytes() for path in [self.base, self.generation, self.review]}
        actual_sha = step.sha
        with patch.object(step, 'sha', wraps=actual_sha) as checked:
            generated, facts = step.prepare()
        self.assertEqual(facts['expected_decision'], 'eligible')
        self.assertEqual(generated['base_sha256'], actual_sha(self.base))
        self.assertEqual([call.args[0] for call in checked.call_args_list],
                         [self.generation, self.base.resolve()])
        for path, content in [*before.items(), *self.original_bytes]:
            self.assertEqual(path.read_bytes(), content)

    def test_windows_and_project_relative_source_references(self):
        tail = '第四章/4.5协作工具/运行结果/handoff_within_window_20261101_150027_129403/record.json'
        for reference in ['D:\\study\\another-checkout\\'+tail.replace('/', '\\'), tail]:
            with self.subTest(reference=reference):
                self.change_reference(reference)
                self.assertEqual(step.prepare()[1]['expected_decision'], 'eligible')

    def test_other_case_and_parent_traversal_rejected(self):
        tail = '第四章/4.5协作工具/运行结果/handoff_within_window_20261101_150027_129403/record.json'
        for reference in [tail.replace('150027_129403', '150028_129403'), '../'+tail, 'record.json']:
            with self.subTest(reference=reference):
                self.change_reference(reference)
                with self.assertRaisesRegex(ValueError, '默认案例'):
                    step.prepare()

    def test_modified_local_evidence_rejected(self):
        self.base.write_bytes(self.base.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, '原始任务记录已变化'):
            step.prepare()

    def test_missing_local_evidence_does_not_fall_back_to_historical_path(self):
        self.base.unlink()
        with self.assertRaises(FileNotFoundError):
            step.prepare()


if __name__ == '__main__':
    unittest.main()

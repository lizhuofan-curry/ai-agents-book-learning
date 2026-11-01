"""假模型响应＋真实教材主循环与文件/执行工具；不调用Ollama。"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
ENV=ROOT.parents[1]/'.venv-ch5'
if Path(sys.prefix).resolve()!=ENV.resolve():
    python=ENV/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    os.execv(str(python),[str(python),str(Path(__file__)),*sys.argv[1:]])
spec=importlib.util.spec_from_file_location('local_coding_task',ROOT/'我的代码/02_官方主循环本机编码任务.py')
step=importlib.util.module_from_spec(spec);spec.loader.exec_module(step)

CODE='''def greet(name):
    return f"Hello, {name}!"

def main():
    print(greet("World"))
    print(greet("Alice"))

if __name__ == "__main__":
    main()
'''


def response(name=None,arguments=None,reason='stop'):
    message={'role':'assistant','content':'' if name else '合成测试：已完成。'}
    if name:message['tool_calls']=[{'function':{'name':name,'arguments':arguments}}]
    return {'model':step.MODEL,'done':True,'done_reason':reason,'message':message,
            'prompt_eval_count':10,'eval_count':20}


class CodingTaskChecks(unittest.TestCase):
    def test_real_official_loop_write_execute_and_tool_feedback(self):
        record={'task':'合成测试：写并执行问候脚本','model_calls':[],'events':[],'messages':[],'test_only':True}
        requests=[]
        with tempfile.TemporaryDirectory(prefix='ch5-offline-') as folder:
            workspace=Path(folder)
            def fake(payload,entry,save):
                requests.append(payload)
                if len(requests)==1:
                    self.assertEqual({x['function']['name'] for x in payload['tools']},step.TOOLS)
                    raw=response('Write',{'file_path':str(workspace/'hello_world.py'),'content':CODE})
                elif len(requests)==2:
                    result=next(x for x in payload['messages'] if x['role']=='tool')
                    self.assertEqual(result['tool_name'],'Write')
                    self.assertIn('bytes_written',json.loads(result['content']))
                    raw=response('Bash',{'command':'python3 hello_world.py'})
                else:
                    result=[x for x in payload['messages'] if x['role']=='tool'][-1]
                    self.assertEqual(result['tool_name'],'Bash')
                    self.assertIn('Hello, Alice!',result['content'])
                    raw=response()
                entry.update(http_status=200,raw_http_body=json.dumps(raw,ensure_ascii=False))
                return raw
            with contextlib.redirect_stdout(io.StringIO()):step.run_task(record,lambda:None,workspace,fake)
            self.assertEqual((workspace/'hello_world.py').read_text(),CODE)
            self.assertEqual(len(requests),3)
            self.assertTrue(all(record['completion_checks'].values()))

    def test_done_without_file_or_execution_not_success(self):
        record={'task':'合成','model_calls':[],'events':[],'messages':[]}
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            step.run_task(record,lambda:None,Path(folder),lambda *args:response())
        self.assertEqual(record['status'],'finished_with_failed_checks')
        self.assertTrue(record['completion_checks']['agent_reported_done'])
        self.assertFalse(record['completion_checks']['independent_artifact_checks'])

    def test_truncated_response_preserved_no_tools_or_retry(self):
        record={'task':'合成','model_calls':[],'events':[],'messages':[]}
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            step.run_task(record,lambda:None,Path(folder),lambda *args:response(reason='length'))
        self.assertEqual(len(record['model_calls']),1)
        self.assertEqual(record['model_calls'][0]['response']['done_reason'],'length')
        self.assertFalse(any(x['type']=='tool_execution_start' for x in record['events']))

    def test_path_escape_and_command_chaining_rejected(self):
        calls=[]
        class Original:
            def get_tool(self,*args):
                class Tool:
                    def execute(self,params):calls.append(params)
                return Tool()
        with tempfile.TemporaryDirectory() as folder:
            record={};workspace=Path(folder)
            registry=step.TaskRegistry(Original(),workspace,record,lambda:None)
            with self.assertRaises(ValueError):registry.get_tool('Write',None).execute({'file_path':'../escape.py','content':CODE})
            (workspace/'hello_world.py').write_text(CODE)
            with self.assertRaises(ValueError):registry.get_tool('Bash',None).execute({'command':'python3 hello_world.py; pwd'})
            with self.assertRaises(ValueError):registry.get_tool('Bash',None).execute({'command':'python3 hello_world.py','run_in_background':True})
        self.assertEqual(calls,[])

    def test_program_with_import_rejected_before_execution(self):
        with self.assertRaises(ValueError):step.validate_code('import os\nprint(os.getcwd())\n')
        with self.assertRaises(ValueError):step.validate_code('greet = eval\ngreet("1+1")\n')
        with self.assertRaises(ValueError):step.validate_code('def main(print=eval):\n    print("1+1")\n')

    def test_prints_correct_but_greet_wrong_is_caught(self):
        code=CODE.replace('return f"Hello, {name}!"','return "Wrong"').replace('print(greet("World"))','print("Hello, World!")').replace('print(greet("Alice"))','print("Hello, Alice!")')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'hello_world.py').write_text(code)
            check=step.check_artifact(path)
            self.assertTrue(check['checks']['script_output_matches'])
            self.assertFalse(check['checks']['greet_return_values'])
            self.assertFalse(check['passed'])

    def test_message_conversion_preserves_arguments_and_tool_correlation(self):
        messages=[{'role':'assistant','content':'','tool_calls':[{'id':'local-1-0','type':'function','function':{'name':'Write','arguments':json.dumps({'content':'a"b','file_path':'hello_world.py'})}}]},
                  {'role':'tool','tool_call_id':'local-1-0','content':'原始工具反馈'}]
        result=step.native_messages(messages)
        self.assertEqual(result[0]['tool_calls'][0]['function']['arguments']['content'],'a"b')
        self.assertEqual(result[1],{'role':'tool','content':'原始工具反馈','tool_name':'Write'})

    def test_model_budget_blocks_extra_request(self):
        called=[]
        adapter=step.NativeCompletions({'model_calls':[{}]*step.MAX_REQUESTS},lambda:None,lambda *args:called.append(1))
        with self.assertRaises(RuntimeError):adapter.create()
        self.assertEqual(called,[])


if __name__=='__main__':unittest.main(verbosity=2)

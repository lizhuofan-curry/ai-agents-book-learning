"""教材CodingAgent主循环＋本机Ollama适配：模型创建并运行hello_world.py。"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
SDK_ENV = PROJECT/'.venv-ch5'
MODEL = 'qwen3.5:9b'
URL = 'http://127.0.0.1:11434/api/chat'
CHINA = timezone(timedelta(hours=8))
MAX_REQUESTS = 6
TOOLS = {'Read','Write','Edit','Bash'}


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_code(text):
    """固定问候任务的可执行语法子集；拒绝导入、属性访问及其他函数调用。"""
    tree=ast.parse(text)
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.JoinedStr,
             ast.FormattedValue,ast.Name,ast.Load,ast.Store,ast.Constant,ast.Expr,ast.Call,
             ast.If,ast.Compare,ast.Eq,ast.Assign,ast.BinOp,ast.Add,ast.keyword,ast.Pass)
    for node in ast.walk(tree):
        if not isinstance(node,allowed):
            raise ValueError('本次固定问候任务不支持语法：'+type(node).__name__)
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in {'print','greet','main'}):
            raise ValueError('本次仅允许调用print、greet和main')
        if isinstance(node,ast.FunctionDef) and node.name not in {'greet','main'}:
            raise ValueError('本次仅需greet和main函数')
        if isinstance(node,ast.FunctionDef):
            args=node.args
            expected=['name'] if node.name=='greet' else []
            if node.decorator_list or [arg.arg for arg in args.args]!=expected or args.defaults or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg:
                raise ValueError('本次函数签名固定为greet(name)和main()，不使用默认参数或装饰器')
        if isinstance(node,ast.Assign) and any(not isinstance(target,ast.Name) or target.id in {'print','greet','main','__builtins__','__name__'} for target in node.targets):
            raise ValueError('本次不改写内置输出、函数或模块标识绑定')
    return tree


def allowed_command(command, workspace):
    parts=shlex.split(command)
    if len(parts)!=2 or parts[0] not in {'python','python3',sys.executable}:
        raise ValueError('Bash本次仅接受 python3 hello_world.py（或python/当前解释器），无命令串联')
    candidate=(workspace/parts[1]).resolve()
    if candidate!=workspace/'hello_world.py':
        raise ValueError('本次只能运行hello_world.py')
    validate_code(candidate.read_text(encoding='utf-8'))


class TaskRegistry:
    """约束范围后调用官方工具；错误原样回传教材循环，不代填模型代码。"""
    def __init__(self, original, workspace, record, save):
        self.original,self.workspace,self.record,self.save=original,Path(workspace).resolve(),record,save

    def get_tool(self,name,state):
        self.record['tool_dispatch_attempts']=self.record.get('tool_dispatch_attempts',0)+1
        self.save()
        if name not in TOOLS or self.record['tool_dispatch_attempts']>12:
            raise ValueError('未知工具或已达12次工具分发预算')
        tool=self.original.get_tool(name,state)
        def execute(params):
            actual=deepcopy(params)
            if name in {'Read','Write','Edit'}:
                path=(self.workspace/params['file_path']).resolve()
                if path!=self.workspace/'hello_world.py':
                    raise ValueError('本次文件工具仅能访问练习目录中的hello_world.py')
                if name=='Write':
                    validate_code(params['content'])
            else:
                if params.get('run_in_background',False) is not False:
                    raise ValueError('本次不启用后台命令')
                timeout=params.get('timeout',10000)
                if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not 0<timeout<=10000:
                    raise ValueError('本次命令超时须在1至10000毫秒之间')
                allowed_command(params['command'],self.workspace)
                # 此默认值在模型可见说明中明确；不替换模型选择的命令或代码。
                actual.setdefault('timeout',10000)
            return tool.execute(actual)
        return SimpleNamespace(execute=execute)


def native_messages(messages):
    """教材兼容接口消息转成本机原生消息；工具编号由客户端维护，不伪称服务端ID。"""
    result=[];names={}
    for message in messages:
        item={'role':message['role'],'content':message.get('content') or ''}
        if message.get('tool_calls'):
            item['tool_calls']=[]
            for call in message['tool_calls']:
                function=call['function'];arguments=function['arguments']
                if isinstance(arguments,str):arguments=json.loads(arguments)
                names[call['id']]=function['name']
                item['tool_calls'].append({'function':{'name':function['name'],'arguments':arguments}})
        if message['role']=='tool':
            item['tool_name']=names[message['tool_call_id']]
        result.append(item)
    return result


def local_post(payload, entry, save):
    import httpx2
    with httpx2.Client(trust_env=False,follow_redirects=False,timeout=600) as client:
        response=client.post(URL,json=payload)
        entry.update(http_status=response.status_code,raw_http_body=response.text)
        save()
        response.raise_for_status()
        return response.json()


class NativeCompletions:
    def __init__(self,record,save,post=local_post):
        self.record,self.save,self.post=record,save,post

    def create(self,**kwargs):
        calls=self.record['model_calls']
        if len(calls)>=MAX_REQUESTS:raise RuntimeError('达到6次本机请求预算')
        request={'model':MODEL,'stream':False,'think':False,
                 'messages':native_messages(kwargs['messages']),'tools':deepcopy(kwargs['tools']),
                 'options':{'temperature':0,'seed':42,'num_predict':2048,'num_ctx':16384}}
        entry={'status':'started','official_call_parameters':deepcopy(kwargs),'native_request':request}
        calls.append(entry);self.save()
        print(f'\n第{len(calls)}次本机模型请求。',flush=True)
        try:
            raw=self.post(deepcopy(request),entry,self.save)
            entry['response']=raw;self.save()
            if raw.get('model')!=MODEL or raw.get('done') is not True or raw.get('done_reason')!='stop':
                raise ValueError('模型名/结束状态异常，保留结果，不追加重试')
            message=raw['message']
            if message.get('role')!='assistant':raise ValueError('模型角色异常')
            mapped=[]
            for i,call in enumerate(message.get('tool_calls') or []):
                function=call['function']
                if not isinstance(function['arguments'],dict):raise ValueError('原生工具参数不是对象')
                mapped.append(SimpleNamespace(index=i,id=f'local-{len(calls)}-{i}',
                    function=SimpleNamespace(name=function['name'],arguments=json.dumps(function['arguments'],ensure_ascii=False))))
            entry.update(status='completed',prompt_tokens_reported=raw.get('prompt_eval_count'),output_tokens_reported=raw.get('eval_count'))
            self.save()
            # 原始响应完整保存；这里仅把一次非流式结果转为教材解析器需要的delta形状。
            return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                content=message.get('content') or '',tool_calls=mapped))])])
        except BaseException as exc:
            entry.update(status='incomplete',error={'type':type(exc).__name__,'message':str(exc)})
            self.save();raise


def check_artifact(workspace):
    target=workspace/'hello_world.py'
    if not target.is_file():return {'exists':False,'passed':False}
    text=target.read_text(encoding='utf-8');tree=validate_code(text)
    process=subprocess.run([sys.executable,'-I',str(target)],cwd=workspace,
                           capture_output=True,text=True,timeout=10,stdin=subprocess.DEVNULL)
    # 用另一个进程验证greet返回值，不以打印输出或语法通过代替函数行为。
    verify=('import runpy,sys,json,contextlib,io\n'
            'n=runpy.run_path(sys.argv[1],run_name="verified_module"); f=n.get("greet")\n'
            'values={x:bool(callable(f) and f(x)=="Hello, "+x+"!") for x in ["Bob","", "小明"]}\n'
            'buffer=io.StringIO()\n'
            'with contextlib.redirect_stdout(buffer): n["main"]()\n'
            'print(json.dumps({"returns":values,"main_stdout":buffer.getvalue()},ensure_ascii=False))')
    function=subprocess.run([sys.executable,'-I','-c',verify,str(target)],cwd=workspace,
                            capture_output=True,text=True,timeout=10,stdin=subprocess.DEVNULL)
    checks={'script_exit_zero':process.returncode==0,
            'script_output_matches':process.stdout.splitlines()==['Hello, World!','Hello, Alice!'],
            'has_main_guard':any(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                                 and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__'
                                 and any(isinstance(x,ast.Constant) and x.value=='__main__' for x in n.test.comparators) for n in tree.body)}
    try:
        values=json.loads(function.stdout.splitlines()[-1]);returns=values['returns']
        checks['greet_return_values']=function.returncode==0 and set(returns)=={'Bob','','小明'} and all(returns.values())
        checks['main_function_behavior']=values['main_stdout'].splitlines()==['Hello, World!','Hello, Alice!']
    except (ValueError,IndexError,KeyError,TypeError):
        checks.update(greet_return_values=False,main_function_behavior=False)
    return {'exists':True,'source':text,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'script_run':{'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr},
            'function_run':{'returncode':function.returncode,'stdout':function.stdout,'stderr':function.stderr},
            'checks':checks,'passed':all(checks.values())}


def run_task(record,save,workspace,post=local_post):
    reference=ROOT/'官方资料/coding-agent'
    os.environ['PYTHON_DOTENV_DISABLED']='1'
    sys.path.insert(0,str(reference))
    from agent import CodingAgent
    old_cwd=Path.cwd();old_ceiling=os.environ.get('GIT_CEILING_DIRECTORIES')
    os.chdir(workspace);os.environ['GIT_CEILING_DIRECTORIES']=str(workspace.parent)
    agent=None
    try:
        agent=CodingAgent(api_key='ollama',model=MODEL,base_url='http://127.0.0.1:11434/v1',provider='openai')
        agent.client.close()  # 不通过SDK发请求；本机适配器显式控制原生上下文长度和记录。
        agent.client=SimpleNamespace(chat=SimpleNamespace(completions=NativeCompletions(record,save,post)))
        agent.system_prompt+='\n\n本次是全新目录内的最小教学任务，仅提供Read、Write、Edit、Bash。TodoWrite和其他工具未提供，本次无需额外任务规划，直接依用户要求创建并运行hello_world.py。只有真实写入和运行后才能报告完成；遵守各工具本次的范围说明。'
        agent.tools=[deepcopy(x) for x in agent.tools if x['name'] in TOOLS]
        for tool in agent.tools:
            tool['description']+=' 本次教学仅访问练习目录hello_world.py；Bash只接受python3 hello_world.py（也可python或当前解释器），默认10000毫秒且不超过10000，不支持后台。代码仅用greet/main/print、字符串及main条件块，无需import。'
            if tool['name']=='Bash':
                properties=tool['input_schema']['properties']
                properties['timeout'].update(minimum=1,maximum=10000,default=10000,
                    description='本次教学默认10000毫秒，填写时须为1至10000毫秒。')
                properties['run_in_background'].update(enum=[False],default=False,
                    description='本次教学仅前台运行，只能为false。')
                properties['command']['description']='本次只运行python3 hello_world.py，可用python或当前解释器替代python3；不拼接其他命令。'
        agent.tool_registry=TaskRegistry(agent.tool_registry,workspace,record,save)
        record['effective_system_prompt']=agent.system_prompt
        record['model_visible_tools']=deepcopy(agent.tools);save()
        for event in agent.run(record['task'],max_iterations=MAX_REQUESTS):
            record['events'].append(deepcopy(event));record['messages']=deepcopy(agent.messages);save()
            if event['type']=='tool_call':print('模型选择：'+event['tool']+' '+json.dumps(event['input'],ensure_ascii=False),flush=True)
            elif event['type']=='tool_execution_complete':print('工具返回：'+json.dumps(event['result'],ensure_ascii=False),flush=True)
            elif event['type']=='text_delta':print(event['delta'],end='',flush=True)
            elif event['type'] in {'error','max_iterations_reached'}:print('\n停止：'+json.dumps(event,ensure_ascii=False),flush=True)
        record['artifact_verification']=check_artifact(workspace)
        events=record['events']
        executions=[e for e in events if e['type']=='tool_execution_complete']
        record['completion_checks']={
            'agent_reported_done':any(e['type']=='done' for e in events),
            'no_engine_error_or_budget_stop':not any(e['type'] in {'error','max_iterations_reached'} for e in events),
            'within_tool_dispatch_budget':record.get('tool_dispatch_attempts',0)<=12,
            'model_used_write':any(e['tool']=='Write' and 'bytes_written' in e['result'] for e in executions),
            'model_ran_script_successfully':any(e['tool']=='Bash' and e['result'].get('exit_code')==0
                and e['result'].get('output','').splitlines()==['Hello, World!','Hello, Alice!'] for e in executions),
            'independent_artifact_checks':record['artifact_verification']['passed']}
        record['status']='coding_task_checked_review_pending' if all(record['completion_checks'].values()) else 'finished_with_failed_checks'
        save()
    finally:
        if agent:
            record['messages']=deepcopy(agent.messages)
            for session in agent.system_state.shell_sessions.values():
                process=session.process
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
                if process is not None:
                    for pipe in (process.stdin,process.stdout,process.stderr):
                        if pipe is not None and not pipe.closed:pipe.close()
                    session.process=None
        os.chdir(old_cwd)
        if old_ceiling is None:os.environ.pop('GIT_CEILING_DIRECTORIES',None)
        else:os.environ['GIT_CEILING_DIRECTORIES']=old_ceiling
        save()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    os.environ['PYTHON_DOTENV_DISABLED']='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if Path(sys.prefix).resolve()!=SDK_ENV.resolve():
        python=SDK_ENV/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        os.execv(str(python),[str(python),str(Path(__file__)),*sys.argv[1:]])
    offline=load(Path(__file__).with_name('01_官方Starter离线检查.py'),'coding_entry_check')
    _,_,manifest_sha,count=offline.prepare()
    print('第五章第二步：教材主循环＋本机Ollama，创建并运行问候脚本。')
    print('最多6次qwen3.5:9b原生请求、12次工具分发；每次输出2048、上下文16384；无重试、云端或模型下载。')
    if args.preview:
        print(f'CODING_TASK_PREVIEW_ONLY：{count}份参考通过，无模型/工具请求、无结果目录。');return 0
    directory=ROOT/'运行结果'/('local_coding_task_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    workspace=directory/'练习项目';workspace.mkdir(parents=True,exist_ok=False);path=directory/'record.json'
    task=(f'请只在当前练习目录{workspace}创建hello_world.py。定义greet(name)，返回字符串Hello, <name>!；'
          '定义main()，依次print(greet("World"))与print(greet("Alice"))，用if __name__ == "__main__"调用main。'
          '请通过Write工具真实写入，再通过Bash执行python3 hello_world.py，根据真实输出判断是否完成。'
          '本次无需import，只用函数、字符串、print和main条件块；不要创建其他文件，不运行其他命令。最后简洁中文说明结果。')
    record={'status':'running','created_at':datetime.now(CHINA).isoformat(),'task':task,
            'manifest_sha256':manifest_sha,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model':MODEL,'endpoint':URL,'model_calls':[],'events':[],'messages':[],
            'maximum_model_requests':MAX_REQUESTS,'maximum_tool_dispatches':12,'automatic_retries':0,'cloud_requests':0,
            'scope':'official CodingAgent control flow and four tool implementations with local native non-streaming adapter and fixed greeting-task guards; not full CLI/provider benchmark or OS sandbox'}
    def save():
        tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
    save();print('记录路径：'+str(path),flush=True)
    try:
        run_task(record,save,workspace)
        print('\n验收：'+json.dumps(record['completion_checks'],ensure_ascii=False))
        print('CODING_TASK_SAVED：实际请求、工具事件及产物已保存，仍需核对解释。')
        return 0 if all(record['completion_checks'].values()) else 1
    except (Exception,KeyboardInterrupt) as exc:
        record.update(status='incomplete',error={'type':type(exc).__name__,'message':str(exc)});print('本步未完成，不重试：'+str(exc));return 1
    finally:
        save();print('记录路径：'+str(path),flush=True)


if __name__=='__main__':raise SystemExit(main())

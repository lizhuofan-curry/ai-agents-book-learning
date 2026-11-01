"""校验固定源码并运行官方main.py --list-tools；不初始化Agent或调用模型。"""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
REFERENCE = ROOT/'官方资料'
CHINA = timezone(timedelta(hours=8))


def prepare():
    manifest_bytes = (REFERENCE/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    for item in manifest['files']:
        path = (REFERENCE/item['path']).resolve()
        if not path.is_relative_to(REFERENCE.resolve()):
            raise ValueError('参考路径越界')
        blob = path.read_bytes()
        sha = hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        if sha != item['git_blob_sha']:
            raise ValueError('固定参考已变化：'+item['path'])
    official = REFERENCE/'coding-agent'
    definitions = json.loads((official/'tools.json').read_text())['tools']
    names = {item['name'] for item in definitions}
    module = ast.parse((official/'tool_registry.py').read_text())
    registry = next(node.value for node in ast.walk(module)
                    if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Attribute)
                    and node.target.attr=='_tools' and isinstance(node.value,ast.Dict))
    registered = {key.value for key in registry.keys}
    if names != registered:
        raise ValueError('工具定义和注册表名称不一致，需先核对')
    return official, sorted(names), hashlib.sha256(manifest_bytes).hexdigest(), len(manifest['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    args = parser.parse_args()
    official,names,manifest_sha,source_count = prepare()
    python = PROJECT/'.venv-ch5'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():
        raise RuntimeError('第五章独立环境尚未准备好，请反馈；不要自行填写或更换API Key')
    print('第五章第一步：官方Starter离线入口。')
    print(f'固定参考{source_count}份文件哈希一致；工具定义与注册表名称一致，共{len(names)}个。')
    print('只运行官方main.py --list-tools；0次模型请求，不执行工具，不初始化CodingAgent。')
    if args.preview:
        print('CODING_STARTER_PREVIEW_ONLY：未启动官方程序、无结果目录。')
        return 0
    directory = ROOT/'运行结果'/('official_cli_offline_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=False)
    path = directory/'record.json'
    command = [str(python),str(official/'main.py'),'--list-tools']
    record = {'status':'running','created_at':datetime.now(CHINA).isoformat(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'upstream_commit':'dbc046eb896ac4e39aa19c7774c8bf49583b89a6',
              'manifest_sha256':manifest_sha,'reference_files_verified':source_count,
              'defined_and_registered_tools':names,'command':command,'working_directory':str(official),
              'model_requests':0,'tools_executed':0,'dotenv_loading_disabled':True,
              'scope':'official CLI list-tools and source integrity checks only; not real model coding task or full official test suite'}
    def save():
        tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        tmp.replace(path)
    save()
    try:
        env=dict(os.environ,PYTHON_DOTENV_DISABLED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
        for key in ('ANTHROPIC_API_KEY','OPENAI_API_KEY','OPENROUTER_API_KEY'):
            env.pop(key,None)
        completed=subprocess.run(command,cwd=official,env=env,stdin=subprocess.DEVNULL,
                                 capture_output=True,text=True,encoding='utf-8',timeout=30)
        record.update(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
        print(completed.stdout,end='',flush=True)
        if completed.stderr:
            print(completed.stderr,file=sys.stderr,end='',flush=True)
        listed={match.group(1) for match in re.finditer(r'^  ([A-Za-z]+)\s+',completed.stdout,re.MULTILINE)}
        record['checks']={'official_cli_exit_zero':completed.returncode==0,
                          'listed_tools_match_definitions':listed==set(names)}
        passed=all(record['checks'].values())
        record['status']='official_cli_offline_checked' if passed else 'failed_offline_check'
        print('核对：'+json.dumps(record['checks'],ensure_ascii=False))
        print('CODING_STARTER_OFFLINE_SAVED：仅离线入口，不代表已完成真实编码任务。')
        return 0 if passed else 1
    except (Exception,KeyboardInterrupt) as exc:
        record.update(status='incomplete',error={'type':type(exc).__name__,'message':str(exc)})
        print('本步未完成：'+str(exc))
        return 1
    finally:
        save()
        print('记录路径：'+str(path),flush=True)


if __name__=='__main__':
    raise SystemExit(main())

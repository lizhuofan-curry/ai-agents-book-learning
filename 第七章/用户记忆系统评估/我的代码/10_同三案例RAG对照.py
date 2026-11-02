"""同三案例纯RAG对照；最多6次本机生成、3次检索、12批CPU前向，0次裁判。"""
import argparse
from datetime import datetime
import fcntl
import importlib.metadata
import os
from pathlib import Path
import sys

import 本机记忆适配 as base
import RAG本机适配 as rag
import 批量JSONCards适配 as journal
import 跨层RAG适配 as batch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--preview',action='store_true')
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--resume',type=Path,help='只继续原目录未开始题；不补跑任何已开始题')
    args=parser.parse_args()
    os.environ.update(PYTHON_DOTENV_DISABLED='1',PYTHONDONTWRITEBYTECODE='1',MAX_RETRIES='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
    for key in list(os.environ):
        if key.endswith('_API_KEY'): os.environ.pop(key)
    network={'allowed_events':[],'blocked_events':[]}
    def audit(event, values):
        if event in {'socket.connect','socket.getaddrinfo','socket.sendto'}:
            allow=not(args.preview or args.self_test) and (
                (event=='socket.connect' and values[1]==('127.0.0.1',11434)) or
                (event=='socket.getaddrinfo' and values[0]=='127.0.0.1' and values[1]==11434))
            network['allowed_events' if allow else 'blocked_events'].append(event)
            if not allow: raise base.StopPipeline('仅允许固定本机Ollama；离线模式禁止全部网络')
    sys.addaudithook(audit)
    plan,cache,provenance,embedding,_=batch.verify_inputs()
    print('第七章同三案例纯RAG：只检索原对话，不把旧卡片或答案交给模型。',flush=True)
    print('最多6次本机qwen生成、3次真实search_memory、12批CPU BGE前向；0次裁判/外部网络。',flush=True)
    print('固定6轮/重叠2轮、casefold、Top5、无重排；不重试、不截断、不扩跑60题。',flush=True)
    if args.preview:
        print('CROSS_RAG_PREVIEW：固定三题/原结果/模型资产已核对，0次模型或前向。')
        return 0
    rag.augment_dependencies()
    official=base.load_official()
    framework=official.UserMemoryEvaluationFramework()
    selected=[framework.get_test_case(x['case_id']) for x in plan['cases']]
    if any(c is None or c.category!=s['layer'] for c,s in zip(selected,plan['cases'])):
        raise ValueError('固定三案例缺失或层次变化')
    if args.self_test:
        tests=rag.load(Path(__file__).with_name('跨层RAG离线检查.py'),'cross_rag_tests')
        return tests.run(official,selected,plan,provenance,embedding,cache,network)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(cache),local_files_only=True,trust_remote_code=False)
    parent=(base.ROOT/'本地运行结果').resolve()
    with (parent/'.rag_cross_layer.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('已有同协议RAG进程运行，不并发重发')
        if args.resume:
            directory=args.resume.resolve()
            if directory.parent!=parent or not directory.name.startswith('rag_cross_layer_'):
                raise ValueError('恢复目录越界或不属于本协议')
            record=journal.load_checkpoint(directory/'record.json')
            batch.check_resume(record,plan,provenance,directory)
            record.setdefault('resume_events',[]).append(datetime.now(journal.CHINA).isoformat())
        else:
            directory=parent/('rag_cross_layer_'+datetime.now(journal.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
            directory.mkdir(exist_ok=False)
            record=batch.new_record(plan,provenance)
            record['created_at']=datetime.now(journal.CHINA).isoformat()
        previous=record['network_audit']
        record['runtime_versions']={n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','numpy','safetensors','rich','tenacity']}
        def save():
            record['network_audit']={k:previous[k]+network[k] for k in network}
            journal.persist(directory/'record.json',record)
        def guard():
            if batch.verify_inputs()[2]!=provenance:
                raise ValueError('发送前来源/实现/分词/嵌入资产变化')
        save()
        print('记录路径：'+str(directory/'record.json'),flush=True)
        try:
            batch.run_cases(official,selected,record,plan,tokenizer,directory,save,guard,embedding)
            guard()
            if network['blocked_events']: raise ValueError('有被阻断的外部连接尝试')
            print('小批摘要：'+str(record['summary']),flush=True)
            print('CROSS_RAG_SAVED：原查询/向量/Top5/回答与失败已保存；相关性及来源忠实性待核对，尚无评分。',flush=True)
            return 0 if record['summary']['answers_saved']==3 else 1
        except (Exception,base.StopPipeline,KeyboardInterrupt) as exc:
            record.update(status='batch_interrupted_or_blocked',error={'type':type(exc).__name__,'message':str(exc)})
            print('CROSS_RAG_INCOMPLETE：失败及已占预算保留，不重试。'+str(exc),flush=True)
            return 1
        finally:
            save()
            print('记录路径：'+str(directory/'record.json'),flush=True)


if __name__=='__main__':
    raise SystemExit(main())

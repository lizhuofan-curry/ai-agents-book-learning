"""按官方元数据核对公开BGE模型文件；只下载，不执行前向。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MANIFEST = ROOT/'官方资料/BGE模型清单.json'
MODEL = 'BAAI/bge-small-zh-v1.5'
REVISION = '7999e1d3359715c523056ef9478215996d62a620'
FILES = {'README.md', 'config.json', 'model.safetensors', 'special_tokens_map.json',
         'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'}


def manifest():
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if data['model_id'] != MODEL or data['revision'] != REVISION or {f['name'] for f in data['files']} != FILES:
        raise ValueError('模型版本或文件集合不符')
    directory = (PROJECT/data['cache_path']).resolve()
    if not directory.is_relative_to((PROJECT/'.uv-cache/huggingface/local_models').resolve()):
        raise ValueError('模型缓存目录越界')
    return data, directory


def verify(raw, item):
    if len(raw) != item['bytes']:
        raise ValueError('文件大小不符：' + item['name'])
    sha256 = hashlib.sha256(raw).hexdigest()
    if item['lfs_sha256']:
        if sha256 != item['lfs_sha256']:
            raise ValueError('权重LFS SHA256不符')
    else:
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if blob != item['git_blob_sha1']:
            raise ValueError('Git blob SHA不符：' + item['name'])
    return sha256


def verify_local_files():
    data, directory = manifest()
    checked = []
    for item in data['files']:
        path = directory/item['name']
        checked.append({'name': item['name'], 'sha256': verify(path.read_bytes(), item)})
    return data, directory, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    data, directory = manifest()
    print(f'固定中文嵌入模型：{MODEL}@{REVISION}，7份文件，共{sum(f["bytes"] for f in data["files"])}字节。')
    if args.preview:
        print('MODEL_DOWNLOAD_PREVIEW_ONLY：不下载、不执行模型。')
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    path = ROOT/'官方资料/BGE下载核对.json'
    record = {'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'status': 'running', 'model_id': MODEL, 'revision': REVISION,
              'files': [], 'download_requests': 0, 'embedding_forward_batches': 0,
              'generation_model_requests': 0, 'automatic_retries': 0}

    def save():
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temp.replace(path)
    save()
    try:
        for item in data['files']:
            target = directory/item['name']
            entry = {'name': item['name'], 'url': f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{item["name"]}'}
            record['files'].append(entry)
            if target.exists():
                entry.update(status='reused_verified', sha256=verify(target.read_bytes(), item), bytes=target.stat().st_size)
                save()
                print('本地核对：' + item['name'], flush=True)
                continue
            record['download_requests'] += 1
            entry['status'] = 'downloading'
            save()
            print('下载：' + item['name'], flush=True)
            with urllib.request.urlopen(entry['url'], timeout=60) as response:
                raw = response.read(item['bytes']+1)
            actual_sha = verify(raw, item)
            temporary = target.with_suffix(target.suffix+'.part')
            temporary.write_bytes(raw)
            temporary.replace(target)
            entry.update(status='downloaded_verified', sha256=actual_sha, bytes=len(raw))
            save()
        record['status'] = 'all_files_downloaded_verified_no_inference'
        save()
        print('BGE_MODEL_READY：7份文件完整性核对通过，尚未执行嵌入前向。')
        return 0
    except Exception as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('下载未完成，不重试：' + type(exc).__name__)
        return 1
    finally:
        save()


if __name__ == '__main__':
    raise SystemExit(main())

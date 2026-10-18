"""下载固定版本的公开重排权重至E盘，核验每份文件；不执行模型前向。"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
DIRECTORY = PROJECT / ('.uv-cache/huggingface/local_models/ms-marco-MiniLM-L6-v2_' + REVISION[:12])
MANIFEST = ROOT / '官方资料/轻量重排模型清单.json'
FILES = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.txt', 'README.md']
CHINA = timezone(timedelta(hours=8))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(path, meta):
    size = path.stat().st_size
    if size != meta['size']:
        raise ValueError(f"{meta['rfilename']}: 文件长度不符")
    if meta.get('lfs'):
        if sha(path) != meta['lfs']['sha256']:
            raise ValueError('权重LFS SHA256不符')
    elif hashlib.sha1(f'blob {size}\0'.encode() + path.read_bytes()).hexdigest() != meta['blobId']:
        raise ValueError('Git blob SHA1不符')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if MANIFEST.exists():
        raise ValueError('已有下载清单，先核对既有文件，不覆盖记录')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    now = datetime.now(CHINA)
    log = ROOT / '安装记录' / ('reranker_download_' + now.strftime('%Y%m%d_%H%M%S_%f') + '.json')
    record = {'model_id': MODEL, 'revision': REVISION, 'local_path': str(DIRECTORY),
              'status': 'downloading', 'created_at': now.isoformat(), 'files': [],
              'model_forward_batches': 0, 'scope': 'Public model file preparation only; no experiment inference.'}

    def save():
        temp = log.with_suffix('.json.tmp')
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.replace(log)

    save()
    try:
        session = requests.Session()
        response = session.get(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}',
                               params={'blobs': 'true'}, timeout=(20, 60))
        response.raise_for_status()
        info = response.json()
        if info['sha'] != REVISION:
            raise ValueError('模型固定版本不一致')
        metadata = {x['rfilename']: x for x in info['siblings']}
        record['total_planned_bytes'] = sum(metadata[name]['size'] for name in FILES)
        save()
        print('固定重排模型：', MODEL, REVISION, flush=True)
        print('下载总字节：', record['total_planned_bytes'], flush=True)
        for name in FILES:
            meta = metadata[name]
            path = DIRECTORY / name
            url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
            if not path.exists():
                part = path.with_suffix(path.suffix + '.part')
                with session.get(url, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    with part.open('wb') as stream:
                        received, milestone = 0, 20 * 1024 * 1024
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                                received += len(chunk)
                                if received >= milestone:
                                    print(f'  {name}: {received}/{meta["size"]}', flush=True)
                                    milestone += 20 * 1024 * 1024
                validate(part, meta)
                part.replace(path)
            else:
                validate(path, meta)
            record['files'].append({'name': name, 'bytes': path.stat().st_size, 'sha256': sha(path),
                                    'git_blob_sha1': meta['blobId'],
                                    'upstream_lfs_sha256': meta.get('lfs', {}).get('sha256'), 'url': url})
            save()
            print('已核验：', name, flush=True)
        record.update(status='downloaded_verified', completed_at=datetime.now(CHINA).isoformat())
        save()
        MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('RERANKER_FILES_VERIFIED：仅文件准备完成，模型前向0批。', flush=True)
        print('模型路径：', DIRECTORY, flush=True)
        return 0
    except Exception as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__,
                      'message': re.sub(r'https?://\S+', '[URL omitted]', str(exc))})
        save()
        print('下载未完成：', record['error'], flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

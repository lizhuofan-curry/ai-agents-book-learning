"""生成官方固定柱状图并阅读；只准备图片，尚不比较模型路线。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT/'官方资料'
CHINA = timezone(timedelta(hours=8))
QUESTION = '哪个季度营收最高，具体数值是多少？请说明单位。'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    manifest = json.loads((SOURCES/'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = (SOURCES/item['path']).resolve()
        if not path.is_relative_to(SOURCES.resolve()) or sha(path) != item['sha256']:
            raise ValueError('冻结官方来源变化')
    # 将绘图库缓存限制在临时目录；只调用已检查的图片函数，不生成PDF。
    os.environ['MPLCONFIGDIR'] = str(Path(tempfile.gettempdir())/'ai_agents_book_ch4_chart_mpl')
    spec = importlib.util.spec_from_file_location('official_multimodal_generator', SOURCES/'create_sample.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.QUARTERS != ['Q1', 'Q2', 'Q3', 'Q4'] or module.REVENUE != [120, 150, 95, 180]:
        raise ValueError('固定合成图表数据变化')
    return module, manifest


def generate(directory):
    from PIL import Image
    directory = Path(directory)
    official, manifest = prepare()
    chart = directory/'sample_chart.png'
    if chart.exists():
        raise FileExistsError('不覆盖已有图片')
    official.create_chart(chart)
    with Image.open(chart) as image:
        image.load()
        details = {'format': image.format, 'width': image.width, 'height': image.height, 'mode': image.mode}
        if image.format != 'PNG' or image.size != (900, 600):
            raise ValueError('测试图格式或尺寸异常')
    reference = {'data_kind': 'synthetic teaching chart',
                 'quarters': official.QUARTERS, 'revenue': official.REVENUE,
                 'unit': 'million USD', 'expected_quarter': official.QUARTERS[official.REVENUE.index(max(official.REVENUE))],
                 'expected_value': max(official.REVENUE),
                 'scope': 'human reference only; future model input must exclude this file and generator arrays'}
    return chart, details, reference, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print('4-3第一步：生成并人工阅读一张固定柱状图；无模型、网络或MCP请求。')
    print('后续问题：', QUESTION)
    print('合成教学图；数字、季度和单位在图中，标准参考单独保存。')
    if args.preview:
        prepare()
        print('MULTIMODAL_CHART_PREVIEW_ONLY：来源及固定数据检查通过，未生成正式图片。')
        return 0
    if args.check:
        with tempfile.TemporaryDirectory(prefix='multimodal_chart_check_') as temp:
            _, details, _, _ = generate(temp)
            print('MULTIMODAL_CHART_OFFLINE_CHECK_PASSED：', details)
        return 0
    directory = ROOT/'运行结果'/('chart_reading_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
              'script_sha256': sha(Path(__file__)), 'question': QUESTION,
              'model_requests': 0, 'network_requests': 0, 'mcp_requests': 0,
              'scope': 'single synthetic chart generation and human reading; no OCR, extraction or model comparison yet'}

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    save()
    try:
        chart, details, reference, manifest = generate(directory)
        (directory/'标准参考_仅供核对.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        reading = '\n'.join(['# 先看图再看标准参考\n', '![固定柱状图](sample_chart.png)\n',
            '这是合成教学图，不是真实公司经营记录。\n',
            '先看：横轴是什么、纵轴单位是什么、每根柱顶写了什么。\n',
            '问题：'+QUESTION+'\n',
            '人工确认后再打开独立参考文件；后续模型不能接收参考答案、生成数据或其他组输出。\n',
            '本次没有OCR、没有生成图像描述，也没有模型回答，不能把参考字段当成提取成果。\n'])
        (directory/'图片阅读说明.md').write_text(reading, encoding='utf-8')
        record.update(status='chart_saved_human_review_pending', manifest=manifest,
                      manifest_sha256=sha(SOURCES/'manifest.json'), chart_filename=chart.name,
                      chart_sha256=sha(chart), image=details,
                      matplotlib_version=importlib.metadata.version('matplotlib'),
                      pillow_version=importlib.metadata.version('pillow'))
        print('图片路径：', chart)
        print('图片信息：', details)
        print('先打开图片，辨认季度、数值和单位；再看独立标准参考。')
        print('MULTIMODAL_CHART_SAVED：图片和阅读材料已保存，待人工核对；尚未调用模型。')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print('本步未完成：', record['error'])
        return 1
    finally:
        save()
        print('记录路径：', path)


if __name__ == '__main__':
    raise SystemExit(main())

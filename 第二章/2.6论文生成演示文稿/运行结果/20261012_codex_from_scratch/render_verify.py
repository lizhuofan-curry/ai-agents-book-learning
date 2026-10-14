"""执行官方缩略图脚本、LibreOffice与Poppler，并保存执行和包检查证据。"""
from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import subprocess
import hashlib
import json
import zipfile
from xml.etree import ElementTree as ET
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent.parent
config=json.loads((LAB/'工具链配置.json').read_text(encoding='utf-8-sig'))
env=os.environ.copy()
env.update(config['environment'])
poppler=Path(config['programs']['pdftoppm']).parent
office=Path(config['programs']['soffice']).parent
env['PATH']=str(office)+os.pathsep+str(poppler)+os.pathsep+env.get('PATH','')
receipts=[]
def run(label,command):
    print('RUN:',label,flush=True)
    begin=datetime.now(timezone.utc).isoformat()
    result=subprocess.run([str(x) for x in command],cwd=ROOT,env=env,capture_output=True,
                          text=True,encoding='utf-8',errors='replace',timeout=240)
    (ROOT/f'{label}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    receipts.append(dict(stage=label,command=[str(x) for x in command],start=begin,
                         end=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode))
    (ROOT/'execution_receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
    if result.returncode:
        raise RuntimeError(label+': '+result.stdout+'\n'+result.stderr)
    print(result.stdout.strip(),flush=True)

run('official_thumbnail',[sys.executable,LAB/'skills/pptx/scripts/thumbnail.py',ROOT/'presentation.pptx',ROOT/'preview/thumbnails','--cols','4'])
run('libreoffice_pdf',[office/'soffice.exe','-env:UserInstallation='+(ROOT/'lo-profile').as_uri(),'--headless','--convert-to','pdf','--outdir',ROOT/'preview',ROOT/'presentation.pptx'])
run('poppler_render',[poppler/'pdftoppm.exe','-png','-r','150',ROOT/'preview/presentation.pdf',ROOT/'preview/slide'])

ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/package/2006/relationships'}
report={'pptx_sha256':hashlib.sha256((ROOT/'presentation.pptx').read_bytes()).hexdigest()}
manifest=json.loads((ROOT/'source_visuals/manifest.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(ROOT/'presentation.pptx') as archive:
    names=archive.namelist()
    slides=sorted([x for x in names if x.startswith('ppt/slides/slide') and x.endswith('.xml')],key=lambda x:int(Path(x).stem[5:]))
    report['slide_count']=len(slides)
    media={n:hashlib.sha256(archive.read(n)).hexdigest() for n in names if n.startswith('ppt/media/')}
    report['original_image_checks']=[]
    for image in manifest['visuals']:
        matches=[n for n,h in media.items() if h==image['sha256']]
        owners=[]
        for number in image['slide_numbers']:
            relations=ET.fromstring(archive.read(f'ppt/slides/_rels/slide{number}.xml.rels'))
            linked=[x.attrib['Target'] for x in relations if '/image' in x.attrib.get('Type','')]
            owners.append({'slide':number,'matching_image_linked':any(Path(m).name==Path(t).name for m in matches for t in linked)})
        report['original_image_checks'].append({'label':image['label'],'byte_identical_media':matches,'owners':owners})
    report['slides']=[]
    for number,file in enumerate(slides,1):
        element=ET.fromstring(archive.read(file))
        text='\n'.join(x.text or '' for x in element.findall('.//a:t',ns))
        tables=len(element.findall('.//a:tbl',ns))
        out=[]
        for transform in element.findall('.//a:xfrm',ns):
            off=transform.find('a:off',ns); extent=transform.find('a:ext',ns)
            if off is not None and extent is not None:
                x,y,cx,cy=[int(z) for z in (off.get('x'),off.get('y'),extent.get('cx'),extent.get('cy'))]
                if x < -1000 or y < -1000 or x+cx > 9144000+1000 or y+cy > 5143500+1000:
                    out.append([x,y,cx,cy])
        report['slides'].append({'number':number,'text':text,'native_tables':tables,'outside_slide_shapes':out})
    report['notes_count']=sum(n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml') for n in names)
    report['zip_crc_ok']=archive.testzip() is None

pdf=PdfReader(ROOT/'preview/presentation.pdf')
report['pdf_page_count']=len(pdf.pages)
report['pdf_extractable_characters']=[len(p.extract_text() or '') for p in pdf.pages]
report['slide_images']=len(list((ROOT/'preview').glob('slide-*.png')))
before=json.loads((ROOT/'skill_hashes_before.json').read_text(encoding='utf-8-sig'))
report['official_skill_files_unchanged']=all(hashlib.sha256((LAB/'skills/pptx'/item['path']).read_bytes()).hexdigest().upper()==item['sha256'] for item in before)
report['official_file_count']=len(before)
report['html_checks']=json.loads((ROOT/'html_checks.json').read_text())
(ROOT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with (ROOT/'disclosure.jsonl').open('a',encoding='utf-8') as stream:
    stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),stage='06_validation_scripts',
        action='实际执行官方thumbnail.py，LibreOffice打开并导出PDF，Poppler渲染所有幻灯片',
        evidence='execution_receipts.json、validation.json、preview/',
        boundary='包检查和渲染完成，不代替逐页视觉检查；未在Microsoft PowerPoint中验收'),ensure_ascii=False)+'\n')
assert report['slide_count']==report['pdf_page_count']==report['slide_images']==14
assert report['notes_count']==14 and report['zip_crc_ok'] and report['official_skill_files_unchanged']
assert all(x['byte_identical_media'] and all(y['matching_image_linked'] for y in x['owners']) for x in report['original_image_checks'])
assert all(not x['imageFailures'] and not x['overflow'] for x in report['html_checks'])
assert all(not x['outside_slide_shapes'] for x in report['slides'])
print('RENDER_AND_PACKAGE_CHECKS_PASSED',flush=True)

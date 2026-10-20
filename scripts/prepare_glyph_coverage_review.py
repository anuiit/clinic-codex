"""Prepare a portable human coverage review; never mutate training labels."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,shutil
from collections import Counter
from pathlib import Path
from PIL import Image
from backend.services.approved_corrections import image_pixel_sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/glyph-coverage-review-20261004-v2'
INPUTS={
    'dataset':ROOT/'backend/training_runs/20261004-yolo-synthetic-v2/dataset.json',
    'sam':ROOT/'reports/nested-branch-rejection-results-20261004.json',
    'yolo':ROOT/'backend/training_runs/20261004-yolo-partial-label-v2/results.json',
    'audit':ROOT/'reports/yolo-annotation-coverage-audit-20261004.json',
    'confirmation':ROOT/'reports/yolo-annotation-coverage-confirmation-20261004.json',
    'taxonomy':ROOT/'backend/codex_model/config.json',
}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def iou(a,b):
    x,y,w,h=a;u,v,s,t=b
    intersection=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
    return intersection/(w*h+s*t-intersection) if intersection else 0.


def validate_review(packet,review):
    """Validate an exported human draft, without importing or granting eligibility."""
    if not isinstance(review,dict) or review.get('schema_version')!='glyph-coverage-decisions.v1':raise ValueError('Invalid decision schema')
    if review.get('packet_sha256')!=packet['packet_sha256']:raise ValueError('Different review packet')
    if not isinstance(review.get('reviewer'),str) or len(review['reviewer'])>160:raise ValueError('Invalid reviewer')
    pages=review.get('pages')
    if not isinstance(pages,dict) or set(pages)!=set(p['group'] for p in packet['pages']):raise ValueError('Page identity mismatch')
    total=Counter()
    for page in packet['pages']:
        record=pages[page['group']]
        if not isinstance(record,dict) or record.get('source_image_pixel_sha256')!=page['source_image_pixel_sha256']:raise ValueError('Pixel identity mismatch')
        decisions=record.get('decisions')
        manual=record.get('manual_items',[])
        if not isinstance(manual,list) or len(manual)>10000:raise ValueError('Invalid manual items')
        for z in manual:
            if not isinstance(z,dict) or not isinstance(z.get('id'),str) or not z['id'].startswith('manual-') or z.get('kind')!='candidate':raise ValueError('Invalid manual item')
            box=z.get('bbox')
            if not isinstance(box,list) or len(box)!=4 or any(type(v) is not int for v in box):raise ValueError('Invalid manual bbox')
            x,y,w,h=box
            if x<0 or y<0 or w<=0 or h<=0 or x+w>page['width'] or y+h>page['height']:raise ValueError('Manual bbox outside page')
        all_items=page['known']+page['candidates']+manual
        items={z['id']:z for z in all_items}
        if len(items)!=len(all_items):raise ValueError('Repeated item identity')
        if not isinstance(decisions,dict):raise ValueError('Invalid decision map')
        for key,decision in decisions.items():
            if key in items:item=items[key]
            else:raise ValueError('Unknown item')
            if not isinstance(decision,dict):raise ValueError('Invalid decision')
            status=decision.get('status')
            if status not in {'accepted','rejected','deleted'}:raise ValueError('Invalid status')
            if status=='deleted' and item['kind']!='known':raise ValueError('Only known annotations may be deleted')
            if status=='rejected' and item['kind']=='known':raise ValueError('Delete known annotation explicitly')
            if not review['reviewer'].strip():raise ValueError('Reviewer required')
            if status=='accepted':
                name=decision.get('class_name');box=decision.get('bbox')
                if name not in packet['class_names']:raise ValueError('Unknown class')
                if not isinstance(box,list) or len(box)!=4 or any(type(v) is not int for v in box):raise ValueError('Integer bbox required')
                x,y,w,h=box
                if x<0 or y<0 or w<=0 or h<=0 or x+w>page['width'] or y+h>page['height']:raise ValueError('Bbox outside page')
            total[status]+=1
        status=record.get('coverage_status')
        if status not in {'unreviewed','incomplete','complete'}:raise ValueError('Invalid coverage status')
        if status!='unreviewed' and not review['reviewer'].strip():raise ValueError('Reviewer required for coverage')
        if status=='complete':
            if record.get('full_page_inspected') is not True:raise ValueError('Full page inspection required')
            if any(z['id'] not in decisions for z in page['candidates']+manual):raise ValueError('Unresolved candidates')
            if not isinstance(record.get('reviewed_at'),str) or not record['reviewed_at'].strip() or not isinstance(record.get('reviewed_by'),str) or not record['reviewed_by'].strip():raise ValueError('Review date and reviewer required')
            final=[dict(z,**decisions.get(z['id'],{})) for z in page['known'] if decisions.get(z['id'],{}).get('status')!='deleted']+[dict(d,id=key,kind='candidate') for key,d in decisions.items() if items[key]['kind']=='candidate' and d['status']=='accepted']
            for z in page['candidates']:
                if z.get('confirmed_target') and not any(d.get('class_name')==z['suggested_class'] and iou(d['bbox'],z['bbox'])>=.5 for d in final):raise ValueError('Confirmed missing target still uncovered')
            if not isinstance(record.get('duplicate_resolutions'),dict):raise ValueError('Duplicate decisions required')
            for pair in page['duplicate_pairs']:
                choice=record['duplicate_resolutions'].get(pair['id'])
                if choice not in {'keep_a','keep_b','distinct'}:raise ValueError('Unresolved duplicate')
                if choice in {'keep_a','keep_b'}:
                    deleted=pair['b' if choice=='keep_a' else 'a']
                    if decisions.get(deleted,{}).get('status')!='deleted':raise ValueError('Duplicate deletion missing')
                    kept=pair['a' if choice=='keep_a' else 'b']
                    if decisions.get(kept,{}).get('status')=='deleted':raise ValueError('Both duplicate boxes removed')
                elif any(decisions.get(pair[k],{}).get('status')=='deleted' for k in ['a','b']):raise ValueError('Distinct objects cannot be deleted by duplicate resolution')
            static={tuple(sorted([p['a'],p['b']])) for p in page['duplicate_pairs']}
            for j,a in enumerate(final):
                for b in final[j+1:]:
                    identity=tuple(sorted([a['id'],b['id']]))
                    if identity not in static and a['class_name']==b['class_name'] and iou(a['bbox'],b['bbox'])>=.8:
                        if record['duplicate_resolutions'].get('dynamic-'+'::'.join(identity))!='distinct':raise ValueError('New overlapping same-class boxes need explicit resolution')
        total['pages_'+status]+=1
    return dict(valid=True,counts=dict(total),applied=False,training_eligibility_granted=False)


def empty_review(packet):
    return dict(schema_version='glyph-coverage-decisions.v1',packet_sha256=packet['packet_sha256'],reviewer='',pages={p['group']:dict(source_image_pixel_sha256=p['source_image_pixel_sha256'],decisions={},manual_items=[],coverage_status='unreviewed',full_page_inspected=False,reviewed_at=None,reviewed_by=None,duplicate_resolutions={}) for p in packet['pages']})


def build():
    assert not OUT.exists(),'Preserve previous packet; choose a new output version'
    sources={k:read(path) for k,path in INPUTS.items()};data=sources['dataset'];names=sources['taxonomy']['class_names']
    assert sources['confirmation']['export_is_exhaustive'] is False
    assert sources['confirmation']['original_audit_sha256']==sha(INPUTS['audit'])
    sam={p['group']:p for p in sources['sam']['outer_pages'] if p['seed']==42}
    yolo={p['group']:p for p in sources['yolo']['pages']}
    assert len(data['pages'])==17 and sum(len(p['gt']) for p in data['pages'])==343
    assert all(names[g['label']]==g['class_name'] for p in data['pages'] for g in p['gt'])
    pages=[];OUT.mkdir();(OUT/'images').mkdir()
    for index,original in enumerate(data['pages'],1):
        group=original['group'];path=Path(original['image']);image=Image.open(path).convert('RGB')
        known=[dict(id=f'known-{j}',kind='known',bbox=g['bbox'],class_name=g['class_name'],annotation_keys=g['annotation_keys']) for j,g in enumerate(original['gt'])]
        assert len({z['id'] for z in known})==len(known)
        candidate=[]
        if group==sources['audit']['example']['group']:
            candidate.append(dict(id='confirmed-s14-head',kind='candidate',bbox=sources['audit']['example']['visible_head_marker'],suggested_class='tlacatl',sources=['Confirmation métier'],similarity=None,confirmed_target=True,note='Tête confirmée comme annotatable. Le rectangle orange est un repère à ajuster et valider.'))
        proposals=[]
        for source,rows in [('SAM',sam[group]['methods']['gated']['predictions']),('YOLO',yolo[group]['views']['raw']['predictions'])]:
            for row in rows:
                if max((iou(row['bbox'],g['bbox']) for g in original['gt']+original['ignore']),default=0)>=.5:continue
                proposals.append(dict(bbox=row['bbox'],suggested_class=names[row['prediction']],similarity=row['similarity'],source=source))
        for proposal in sorted(proposals,key=lambda p:-p['similarity']):
            existing=next((c for c in candidate if iou(c['bbox'],proposal['bbox'])>=.8),None)
            evidence=dict(source=proposal['source'],bbox=proposal['bbox'],class_name=proposal['suggested_class'],similarity=proposal['similarity'])
            if existing:
                existing.setdefault('evidence',[]).append(evidence)
                if proposal['source'] not in existing['sources']:existing['sources'].append(proposal['source'])
            else:candidate.append(dict(id=f'candidate-{len(candidate)}',kind='candidate',bbox=proposal['bbox'],suggested_class=proposal['suggested_class'],similarity=proposal['similarity'],sources=[proposal['source']],evidence=[evidence],confirmed_target=False,note='Suggestion de modèle, pas une annotation validée.'))
        candidate.sort(key=lambda c:(not c['confirmed_target'],-len(c['sources']),-(c['similarity'] or 0)))
        duplicates=[]
        for a in range(len(known)):
            for b in range(a+1,len(known)):
                if known[a]['class_name']==known[b]['class_name'] and iou(known[a]['bbox'],known[b]['bbox'])>=.8:
                    duplicates.append(dict(id=f'duplicate-{a}-{b}',a=known[a]['id'],b=known[b]['id'],iou=iou(known[a]['bbox'],known[b]['bbox'])))
        filename=f'images/page-{index:02}.png';shutil.copyfile(path,OUT/filename);assert sha(path)==sha(OUT/filename)
        pages.append(dict(group=group,title='S14' if group==sources['audit']['example']['group'] else f'Planche {index:02}',analysis_ids=sorted({k.split(':')[0] for z in known for k in z['annotation_keys']}),image=filename,image_sha256=sha(path),source_image_pixel_sha256=image_pixel_sha256(image),width=image.width,height=image.height,known=known,ignored=original['ignore'],candidates=candidate,duplicate_pairs=duplicates))
    pages.sort(key=lambda p:(p['title']!='S14',p['title']))
    files=list(INPUTS.values())+[Path(__file__),ROOT/'scripts/glyph_coverage_review.html',ROOT/'backend/services/approved_corrections.py']
    packet=dict(schema_version='glyph-coverage-packet.v1',hash_algorithm='sha256(width+"x"+height+":RGB:"+RGB_pixel_bytes), matching approved_corrections.image_pixel_sha256',class_names=names,pages=pages,input_sha256={str(p):sha(p) for p in files},policy='Human review only. Candidates merge at IoU>=.8, unknown regions have maxIoU<.5 against any known or ignored box. All decisions initially unreviewed. Complete requires full-page inspection plus resolved suggestions and duplicates; it does not grant training eligibility. No new inference, scoring, fitting or corpus modification.',scope='17 previously examined training pages,343 known records,7 ignores. No dev/test pages or new independent lot.')
    packet['packet_sha256']=hashlib.sha256(json.dumps(packet,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    write(OUT/'packet.json',packet);(OUT/'data.js').write_text('const DATA = '+json.dumps(packet,ensure_ascii=False).replace('<','\\u003c')+';\n',encoding='utf-8')
    shutil.copyfile(ROOT/'scripts/glyph_coverage_review.html',OUT/'index.html')
    review=empty_review(packet);write(OUT/'decisions-initial.json',review);assert validate_review(packet,review)['valid']
    validate_assets(packet,OUT)
    summary=dict(pages=len(pages),known=sum(len(p['known']) for p in pages),ignored=sum(len(p['ignored']) for p in pages),candidates=sum(len(p['candidates']) for p in pages),duplicate_pairs=sum(len(p['duplicate_pairs']) for p in pages),confirmed_missing_targets=sum(c['confirmed_target'] for p in pages for c in p['candidates']),by_page=[dict(title=p['title'],group=p['group'],known=len(p['known']),candidates=len(p['candidates']),duplicate_pairs=len(p['duplicate_pairs'])) for p in pages],packet_sha256=packet['packet_sha256'],source_files_unchanged=all(sha(p)==h for p,h in packet['input_sha256'].items()),labels_modified=False,optimizer_updates=0,activated=False)
    assert summary['source_files_unchanged'];write(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def validate_assets(packet,folder):
    base=folder.resolve()
    unsigned={k:v for k,v in packet.items() if k!='packet_sha256'}
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=packet['packet_sha256']:raise ValueError('Packet digest mismatch')
    for page in packet['pages']:
        relative=Path(page['image'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='images':raise ValueError('Unsafe image path')
        path=(base/relative).resolve()
        if not path.is_relative_to(base):raise ValueError('Image outside packet')
        if sha(path)!=page['image_sha256']:raise ValueError('Image bytes changed')
        with Image.open(path) as image:
            if image.size!=(page['width'],page['height']) or image_pixel_sha256(image)!=page['source_image_pixel_sha256']:raise ValueError('Image pixels changed')
    return True


def check():
    assert iou([0,0,10,10],[0,0,10,10])==1 and iou([0,0,10,10],[20,0,10,10])==0
    packet=dict(packet_sha256='fixture',class_names=['tlacatl'],pages=[dict(group='p',source_image_pixel_sha256='pixel',width=100,height=100,known=[dict(id='known-0',kind='known',bbox=[30,30,10,10],class_name='tlacatl')],candidates=[dict(id='candidate-0',kind='candidate')],duplicate_pairs=[])])
    review=empty_review(packet);assert validate_review(packet,review)['valid']
    bad=copy.deepcopy(review);bad['pages']['p'].update(coverage_status='complete',full_page_inspected=True,reviewed_at='2026-10-04',reviewed_by='test');bad['reviewer']='test'
    try:validate_review(packet,bad)
    except ValueError:pass
    else:raise AssertionError('Unresolved candidate must block completeness')
    bad['pages']['p']['decisions']['candidate-0']=dict(status='accepted',class_name='tlacatl',bbox=[0,0,10,10]);assert validate_review(packet,bad)['valid']
    bad['pages']['p']['decisions']['candidate-0']['bbox']=[95,0,10,10]
    try:validate_review(packet,bad)
    except ValueError:pass
    else:raise AssertionError('Out-of-page box must fail')
    def fails(value):
        try:validate_review(packet,value)
        except ValueError:return
        raise AssertionError('Invalid review accepted')
    orphan=empty_review(packet);orphan['reviewer']='test';orphan['pages']['p']['decisions']['manual-orphan']=dict(status='rejected');fails(orphan)
    bad['pages']['p']['decisions']['candidate-0']['bbox']=[0,0,10,10];bad['pages']['p']['reviewed_at']='';fails(bad)
    bad['pages']['p']['reviewed_at']='2026-10-04';bad['pages']['p']['decisions']['candidate-0']['bbox']=[30,30,10,10];fails(bad)
    bad['pages']['p']['duplicate_resolutions']['dynamic-candidate-0::known-0']='distinct';assert validate_review(packet,bad)['valid']
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp);(folder/'images').mkdir();asset=folder/'images/a.png';Image.new('RGB',(2,2),'white').save(asset)
        fixture=dict(pages=[dict(image='images/a.png',image_sha256=sha(asset),width=2,height=2,source_image_pixel_sha256=image_pixel_sha256(Image.open(asset)))])
        fixture['packet_sha256']=hashlib.sha256(json.dumps(fixture,sort_keys=True,ensure_ascii=False).encode()).hexdigest();assert validate_assets(fixture,folder)
        Image.new('RGB',(2,2),'black').save(asset)
        try:validate_assets(fixture,folder)
        except ValueError:pass
        else:raise AssertionError('Modified image accepted')
    print('CHECK_PASSED: partial coverage, pending candidate and invalid bbox rejected')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['build','check','validate']);parser.add_argument('--review',type=Path);args=parser.parse_args()
    if args.phase=='build':build()
    elif args.phase=='check':check()
    else:
        if args.review is None:parser.error('--review is required')
        packet=read(OUT/'packet.json');validate_assets(packet,OUT)
        print(json.dumps(validate_review(packet,read(args.review)),indent=2))



"""CEN text ablations on frozen glyph crops; no active-model writes."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, re, sys, time, unicodedata
from html.parser import HTMLParser
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'backend/training_runs/cen-semantic-deps'),str(ROOT),str(ROOT/'backend')]
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scripts import research_recognition_mechanisms as mechanism
from scripts import research_recognition_stability as core
from scripts import research_base_corpus_long as base
from scripts.train_segmentation import read_json,write_json,sha256
from codex_pipeline.data.tlachia.catalog import CachedTransport
SOURCE=ROOT/'reports/cen-semantic-20261004-v2'
HTTP_STORE=ROOT/'reports/cen-semantic-20261004/http'
RUN=ROOT/'backend/training_runs/20261004-cen-semantics-v2'
CATALOG=ROOT/'reports/tlachia-p3-20260813/catalog/api-global-elements-v1.json'
KINDS=['name','meaning','gdn','notice','visual']
ALPHAS=[0.,.05,.1,.2,.4,.6,.8,1.]
HF=Path.home()/'.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots'
CLIP_CONFIG=HF/'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CLIP_WEIGHTS=HF/'c237dc49a33fc61debc9276459120b7eac67e7ef/model.safetensors'
SIGLIP=ROOT/'backend/training_runs/cen-semantic-models/models--google--siglip2-base-patch16-224/snapshots/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'

class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.parts=[];self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in {'script','style'}:self.hidden+=1
        if tag in {'br','p','div','li'}:self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in {'script','style'}:self.hidden=max(0,self.hidden-1)
        if tag in {'p','div','li'}:self.parts.append('\n')
    def handle_data(self,value):
        if not self.hidden:self.parts.append(value)

def plain(value):
    p=PlainText();p.feed(str(value or ''))
    return '\n'.join(re.sub(r'\s+',' ',s).strip() for s in ''.join(p.parts).splitlines() if s.strip())

def prose(value):
    text=plain(value)
    text=re.split(r'(?im)^\s*(?:GLIFOS|GLYPHES|GLYPHS|VALORES F[ÓO]NICOS|VALEURS PHON[IÉ]QUES|CITATIONS)\s*:?\s*$',text,maxsplit=1)[0]
    text=re.sub(r'\b(?:\d{2,4}|[A-Z][A-Za-z0-9-]*)(?:[_.-][A-Za-z0-9()]+){2,}\b','',text)
    text=re.sub(r'https?://\S+','',text)
    return ' '.join(s for s in text.splitlines() if len(s)>30 and not re.match(r'^[\d. :]+',s))[:2400].strip()

def visual_sentences(text):
    # ponytail: lexical extraction can retain interpretation; expert visual attributes require review.
    keywords=r'glif|glyph|dessin|trac[eé]|pint|color|couleur|profil|cabeza|t[eê]te|rectang|triang|circul|contour|repr[eé]sent|configur|dibuj|forma|forme|muros|pluma|plume|ojo|yeux|face'
    return ' '.join(s for s in re.split(r'(?<=[.!?])\s+',text) if re.search(keywords,s,re.I))[:1600]

def norm(value):return unicodedata.normalize('NFC',str(value)).strip().casefold()

def sources(offline=False):
    SOURCE.mkdir(parents=True,exist_ok=True)
    names=read_json(core.RUNTIME/'config.json')['class_names']
    by=collections.defaultdict(list)
    for n in read_json(CATALOG)['payload']['nodes']['element']:by[norm(n['element'])].append(n)
    assert len(names)==286 and all(norm(n) in by for n in names)
    transport=CachedTransport(HTTP_STORE,offline=offline,max_requests=900,min_interval_seconds=.15)
    dictionaries,dict_hash=transport.post_json({'mode':'dict'})
    entries=[]
    for i,name in enumerate(names):
        raw=[r for n in by[norm(name)] for r in n.get('raw_variants',[n.get('raw',{})])]
        meaning_counts={lang:collections.Counter(plain(r.get(field)) for r in raw if r.get(field)) for lang,field in [('es','meaning'),('fr','meaning2')]}
        meanings={lang:[s for s,n in sorted(counts.items(),key=lambda v:(-v[1],v[0]))] for lang,counts in meaning_counts.items()}
        choices=collections.Counter((str(r['codexId']),str(r['theme'])) for r in raw if r.get('txt') and r.get('codexId') and r.get('theme'))
        notice='';info=None;attempts=[]
        # Fixed catalogue choice, independent of held pages and image scores.
        for (codex_id,theme),count in sorted(choices.items(),key=lambda x:(-x[1],x[0]))[:2]:
            query=dict(mode='pres',pres='tlachia',id=codex_id,type='element',text=theme,lang='fr')
            response,response_hash=transport.post_json(query)
            cleaned=prose(response.get('text'))
            attempts.append(dict(query=query,response_sha256=response_hash,file=response.get('file'),prose_chars=len(cleaned)))
            if cleaned:
                notice=cleaned;info=dict(attempts[-1],url=f'https://cen.sup-infor.com/tlachia/{codex_id}/element',theme=theme);break
        lookup=re.sub(r'_\d+$','',name).replace('_',' ')
        query=dict(mode='trad',nah=True,norm=True,cond=dict(mode='all',field='norma'),dic='',sort=0,word=lookup)
        response,response_hash=transport.post_json(query)
        lexical=[]
        for block in response.get('gdn',[]):
            if norm(block.get('norma',''))!=norm(lookup):continue
            for item in block.get('gdn',[]):
                text=re.sub(r'\[[^\]]*\]','',plain(item.get('trad'))).strip()
                if item.get('dic')!='5' and re.search(r'[A-Za-zÀ-ÿ]{3}',text):
                    lexical.append(dict(text=text[:1200],entry_id=item.get('id'),dictionary_id=str(item.get('dic')),paleo=item.get('paleo')))
        priority=['24','21','22','1']
        lexical.sort(key=lambda x:(priority.index(x['dictionary_id']) if x['dictionary_id'] in priority else 10,x['dictionary_id'],str(x['entry_id'])))
        entries.append(dict(class_label=i,name=name,match='exact_tlachia_label',meaning_counts=meaning_counts,meanings=meanings,notice=notice,visual=visual_sentences(notice),notice_source=info,notice_attempts=attempts,
            lexical=lexical[:4],gdn_query=query,gdn_response_sha256=response_hash,gdn_url='https://cen.sup-infor.com/gdn/'+lookup.replace(' ','%20'),gdn_scope='shared_lemma_for_variant' if lookup!=name else 'exact_lemma'))
        if (i+1)%10==0:print('CEN',i+1,'/286','notices',sum(bool(e['notice']) for e in entries),'gdn',sum(bool(e['lexical']) for e in entries),flush=True)
    result=dict(schema_version='cen-class-text.v1',source='https://cen.sup-infor.com/home/hellow',api='https://cen2.sup-infor.com/Systeme/getCEN.json',captured_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        catalog_path=str(CATALOG),catalog_sha256=sha256(CATALOG),taxonomy_sha256=hashlib.sha256(json.dumps(names,ensure_ascii=False).encode()).hexdigest(),dictionaries=dictionaries,dictionaries_response_sha256=dict_hash,entries=entries,
        coverage=dict(classes=len(entries),meaning=sum(any(e['meanings'].values()) for e in entries),notice=sum(bool(e['notice']) for e in entries),visual=sum(bool(e['visual']) for e in entries),gdn=sum(bool(e['lexical']) for e in entries)),
        policy='Exact class-label mapping, no merging. Variant lexical definitions can be shared. External class knowledge only; no images or per-object labels used.')
    for entry in entries:
        for field in ['notice','visual']:
            assert not re.search(r'\b(?:\d{2,4}|[A-Z][A-Za-z0-9-]*)(?:[_.-][A-Za-z0-9()]+){2,}\b',entry[field]),(entry['name'],field)
    write_json(SOURCE/'class-text.json',result);print('COVERAGE',result['coverage'],flush=True)

def prompts(e,kind):
    name=e['name'];named=[f'An Aztec pictographic glyph representing {name}.']
    meaning=([f'Un glifo nahua {name}: {v}.' for v in e['meanings']['es'][:1]]+[f'Un glyphe nahua {name} : {v}.' for v in e['meanings']['fr'][:1]]) or named
    if kind=='name':return named
    if kind=='meaning':return meaning
    if kind=='gdn':return [f"Un glyphe nahua {name}. {v['text']}" for v in e['lexical'][:2]] or meaning
    text=e['visual'] if kind=='visual' else e['notice']
    return [f'Un glyphe nahua {name}. {text}'] if text else meaning

def setup():
    torch.set_num_threads(4);torch.manual_seed(42)
    d=mechanism.setup();fixture=mechanism.load_fixture(d)
    d['components']=read_json(core.SNAPSHOT/'snapshot_manifest.json')['components']
    rows={r['row_id']:r for r in csv.DictReader((core.SNAPSHOT/'metadata.csv').open())}
    queries=[]
    for i in range(d['n']):
        if d['splits'][i]=='train' and d['groups'][i] in d['business']:
            r=rows[d['objects'][i]];queries.append(dict(index=i,row_id=r['row_id'],view='raw',group=d['groups'][i],label=int(d['labels'][i]),path=str(core.SNAPSHOT/r['image_path']),output_sha256=r['output_sha256']))
    for j,r in enumerate(fixture['rows']):
        if r['split']=='train':queries.append(dict(index=d['n']+r['runtime_index'],row_id=r['row_id'],view='mask',group=r['source_group'],label=int(d['labels'][d['n']+r['runtime_index']]),fixture_index=j))
    assert collections.Counter(q['view'] for q in queries)=={'raw':343,'mask':254}
    d['queries']=queries;d['fixture']=fixture;return d

def register(d):
    RUN.mkdir(parents=True,exist_ok=True)
    paths=d['paths']+[Path(__file__),SOURCE/'class-text.json',CATALOG,mechanism.RUN/'crop-fixture.pt',mechanism.RUN/'crop-manifest.json',core.RUN/'cv-results.json',Path(base.__file__),Path(mechanism.__file__),Path(core.__file__),CLIP_WEIGHTS]
    paths += [p for p in CLIP_CONFIG.iterdir() if p.is_file() and p.suffix=='.json']+[CLIP_CONFIG/'merges.txt']
    paths += [p for p in SIGLIP.iterdir() if p.is_file()]
    import importlib.metadata
    versions={k:importlib.metadata.version(k) for k in ['transformers','tokenizers','safetensors','numpy','sentencepiece']}
    versions['torch']=torch.__version__
    protocol=dict(schema_version='cen-semantics.v1',versions=versions,input_sha256={str(p):sha256(p) for p in paths},models=['openai/clip-vit-base-patch32','google/siglip2-base-patch16-224'],frozen=True,dtype='float32',prompts=KINDS,
        text_policy='CEN only, no generated visual facts. Meanings: most frequent per language; all variants retained in source audit. Notices stripped of occurrence lists. Visual sentences are lexical extraction, not expert validation.',
        image_policy='Same 343 train annotation crops and 254 verified SAM pixels. White square padding before official preprocessing.',
        fusion='Row-standardized cosines, convex fusion. Alpha chosen per view/kind on remaining training pages, excluding outer components and pixels.',alpha_grid=ALPHAS,
        calibration_objective='mean per-page top1; tie chooses smallest alpha',shuffled_control='derangement seed123, same calibration',folds=mechanism.folds(),
        baselines=['historical DINO projection/stored prototypes','fold train-only recomputed DINO prototypes'],rare_bins='natural unique training pixels 1..4 after exclusions',
        locked_test_scored=False,dev_scored=False,limitations=['Pages repeatedly explored; no new blind lot','Historical DINO projection provenance unknown','CLIP primarily English, CEN French/Spanish','Conditional naming only; incomplete bbox coverage prevents full detector evaluation','Variant lemmas and some catalogue meanings are ambiguous'],activated=False)
    path=RUN/'preregistration.json'
    if path.exists():assert read_json(path)==protocol,'protocol drift'
    else:write_json(path,protocol)

def load_model(which):
    if which=='clip':
        from transformers import CLIPConfig,CLIPModel,CLIPProcessor
        from safetensors.torch import load_file
        m=CLIPModel(CLIPConfig.from_pretrained(CLIP_CONFIG,local_files_only=True));state=load_file(str(CLIP_WEIGHTS))
        for key in ['text_model.embeddings.position_ids','vision_model.embeddings.position_ids']:state.pop(key,None)
        m.load_state_dict(state,strict=True);p=CLIPProcessor.from_pretrained(CLIP_CONFIG,local_files_only=True);limit=77
    else:
        from transformers import SiglipModel,SiglipImageProcessor,AutoTokenizer
        from types import SimpleNamespace
        m=SiglipModel.from_pretrained(SIGLIP,local_files_only=True,use_safetensors=True)
        p=SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(SIGLIP,local_files_only=True,use_fast=True),image_processor=SiglipImageProcessor.from_pretrained(SIGLIP,local_files_only=True))
        limit=m.config.text_config.max_position_embeddings
    return m.cuda().eval(),p,limit

@torch.no_grad()
def encode(which):
    d=setup();register(d);model,p,limit=load_model(which)
    text=read_json(SOURCE/'class-text.json')
    assert [e['name'] for e in text['entries']]==[d['cache']['class_names'][i] for i in range(286)]
    vectors={};audit=[]
    for kind in KINDS:
        per_class=[]
        for e in text['entries']:
            ps=prompts(e,kind);chunks=[]
            for prompt in ps:
                tokens=p.tokenizer.encode(prompt,add_special_tokens=False,verbose=False)
                for start in range(0,len(tokens),limit-12):chunks.append(p.tokenizer.decode(tokens[start:start+limit-12]))
            assert all(len(p.tokenizer.encode(chunk,add_special_tokens=True,verbose=False))<=limit for chunk in chunks),'text window overflow'
            batch=p.tokenizer(chunks,padding='max_length' if which=='siglip2' else True,truncation=True,max_length=limit,return_tensors='pt').to('cuda')
            z=F.normalize(model.get_text_features(**batch).float(),dim=1)
            per_class.append(F.normalize(z.mean(0),dim=0).cpu());audit.append(dict(kind=kind,name=e['name'],prompts=ps,chunks=chunks))
        vectors[kind]=torch.stack(per_class);print('TEXT_ENCODED',which,kind,vectors[kind].shape,flush=True)
    images=[];hashes=[]
    for q in d['queries']:
        if q['view']=='raw':
            assert sha256(Path(q['path']))==q['output_sha256'],'snapshot crop bytes drift'
            image=Image.open(q['path']).convert('RGB')
        else:
            j=q['fixture_index'];arr=mechanism.render_views(d['fixture']['rgb'][j].numpy(),d['fixture']['masks'][j].numpy())['mask'];image=Image.fromarray(arr)
        hashes.append(mechanism.digest(np.array(image)));side=max(image.size);square=Image.new('RGB',(side,side),'white')
        square.paste(image,((side-image.width)//2,(side-image.height)//2));images.append(square)
    features=[]
    for start in range(0,len(images),16):
        batch=p.image_processor(images=images[start:start+16],return_tensors='pt').to('cuda');features.append(F.normalize(model.get_image_features(**batch).float(),dim=1).cpu())
        if start%160==0:print('IMAGE_ENCODED',which,start,'/',len(images),flush=True)
    torch.save(dict(text=vectors,images=torch.cat(features),queries=d['queries'],pixel_sha256=hashes),RUN/f'{which}-features.pt')
    write_json(RUN/f'{which}-prompt-audit.json',dict(prompts=audit));verify()

def standardize(scores):return (scores-scores.mean(1,keepdim=True))/scores.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)

def stats(pred,labels,groups,rare=None):
    correct=pred==labels
    per_class=[float(correct[labels==c].float().mean()) for c in labels.unique()]
    rare_classes=[c for c in labels.unique().tolist() if rare is not None and 1<=rare[c]<=4]
    return dict(n=len(labels),correct=int(correct.sum()),accuracy=float(correct.float().mean()),class_macro=float(np.mean(per_class)),classes=len(per_class),
        page_macro=float(np.mean([float(correct[torch.tensor([x==g for x in groups],device=labels.device)].float().mean()) for g in sorted(set(groups))])),
        rare_class_macro=float(np.mean([float(correct[labels==c].float().mean()) for c in rare_classes])) if rare_classes else None,rare_classes=len(rare_classes))

def choose_alpha(a,b,labels,groups):
    values=[stats(((1-alpha)*a+alpha*b).argmax(1),labels,groups)['page_macro'] for alpha in ALPHAS];best=max(values)
    return next(alpha for alpha,value in zip(ALPHAS,values) if abs(value-best)<1e-12),values

@torch.no_grad()
def evaluate(which):
    d=setup();register(d);cache=torch.load(RUN/f'{which}-features.pt',weights_only=True,map_location='cpu');assert cache['queries']==d['queries']
    z=d['teacher'][[q['index'] for q in d['queries']]].cpu();labels=torch.tensor([q['label'] for q in d['queries']]);groups=[q['group'] for q in d['queries']]
    historical=z@d['old'].cpu().T;text_scores={k:cache['images']@v.T for k,v in cache['text'].items()}
    generator=torch.Generator().manual_seed(123);perm=torch.randperm(286,generator=generator)
    while (perm==torch.arange(286)).any():perm=torch.randperm(286,generator=generator)
    folds=[];predictions={v:collections.defaultdict(list) for v in ['raw','mask']}
    for fold,held in enumerate(mechanism.folds()):
        ids,excluded=base.train_indices(d,'combined',held);held_pixels={d['pixels'][i] for i in range(d['n']) if d['groups'][i] in held}
        unique={(int(d['labels'][i]),d['pixels'][i]):i for i in ids};ids=sorted(unique.values());counts=collections.Counter(int(d['labels'][i]) for i in ids)
        center,_=core.centroid(d['teacher'],d['labels'],ids);recomputed=z@center.cpu().T
        result=dict(fold=fold,held=held,excluded_groups=excluded,reference_count=len(ids),views={})
        for view in ['raw','mask']:
            outer=[j for j,q in enumerate(d['queries']) if q['view']==view and q['group'] in held]
            calibration=[j for j,q in enumerate(d['queries']) if q['view']==view and q['group'] not in excluded and d['pixels'][q['index']] not in held_pixels]
            assert outer and calibration and not set(outer)&set(calibration)
            assert not {groups[j] for j in outer}&{groups[j] for j in calibration}
            a=standardize(historical);methods={'dino_historical':historical.argmax(1),'dino_recomputed':recomputed.argmax(1)};selections={}
            for kind in KINDS:
                b=standardize(text_scores[kind])
                for control,bb in [('',b),('_shuffled',b[:,perm])]:
                    alpha,curve=choose_alpha(a[calibration],bb[calibration],labels[calibration],[groups[j] for j in calibration])
                    methods[f'fusion_{kind}{control}']=((1-alpha)*a+alpha*bb).argmax(1);selections[kind+control]=dict(alpha=alpha,calibration_page_macro=curve)
                methods['text_'+kind]=text_scores[kind].argmax(1)
            result['views'][view]=dict(n=len(outer),calibration_n=len(calibration),selections=selections,methods={m:stats(p[outer],labels[outer],[groups[j] for j in outer],counts) for m,p in methods.items()})
            for method,pred in methods.items():predictions[view][method].extend(dict(row_id=d['queries'][j]['row_id'],group=groups[j],label=int(labels[j]),prediction=int(pred[j]),fold=fold,natural_support=counts[int(labels[j])]) for j in outer)
        folds.append(result);print('FOLD',which,fold,{v:{m:x['correct'] for m,x in result['views'][v]['methods'].items() if m in ['dino_historical','text_meaning','fusion_meaning','fusion_notice']} for v in ['raw','mask']},flush=True)
    report=dict(schema_version='cen-semantics-results.v1',model=which,folds=folds,aggregate={},predictions=predictions,activated=False,dev_scored=False,locked_test_scored=False)
    for view,methods in predictions.items():
        aggregate={}
        for method,rows in methods.items():
            assert len(rows)==(343 if view=='raw' else 254) and len({r['row_id'] for r in rows})==len(rows)
            p=torch.tensor([r['prediction'] for r in rows]);y=torch.tensor([r['label'] for r in rows]);metrics=stats(p,y,[r['group'] for r in rows])
            rare_classes=sorted({r['label'] for r in rows if 1<=r['natural_support']<=4})
            metrics['rare_class_macro']=float(np.mean([np.mean([r['prediction']==r['label'] for r in rows if r['label']==c and 1<=r['natural_support']<=4]) for c in rare_classes])) if rare_classes else None;metrics['rare_classes']=len(rare_classes)
            metrics['per_class']=[dict(name=d['cache']['class_names'][c],n=int((y==c).sum()),correct=int(((y==c)&(p==c)).sum())) for c in y.unique().tolist()]
            ref={r['row_id']:r for r in methods['dino_historical']};metrics['fixed']=sum(r['prediction']==r['label'] and ref[r['row_id']]['prediction']!=r['label'] for r in rows);metrics['broken']=sum(r['prediction']!=r['label'] and ref[r['row_id']]['prediction']==r['label'] for r in rows)
            aggregate[method]=metrics
        report['aggregate'][view]=aggregate
    write_json(RUN/f'{which}-results.json',report);verify()
    print('AGGREGATE',which,{v:{m:{k:x[k] for k in ['correct','n','accuracy','class_macro','fixed','broken']} for m,x in methods.items()} for v,methods in report['aggregate'].items()},flush=True)

def verify():
    p=read_json(RUN/'preregistration.json');assert p['input_sha256']=={path:sha256(Path(path)) for path in p['input_sha256']},'input drift'
    artifacts={str(x):sha256(x) for x in RUN.iterdir() if x.is_file() and x.name!='verification.json'}
    write_json(RUN/'verification.json',dict(inputs_unchanged=True,active_model_unchanged=True,dev_scored=False,locked_test_scored=False,artifact_sha256=artifacts))

def check():
    assert prose('<p>A glyph drawn in red with a circular contour.</p><p>GLIFOS</p>073_01_A_01: foo')=='A glyph drawn in red with a circular contour.'
    assert '073_01_A_01' not in prose('<p>A glyph drawn in red with a circular contour. 073_01_A_01.</p>')
    assert 'MC2_B_031' not in prose('<p>A glyph drawn in red with a circular contour. MC2_B_031.</p>')
    assert 'X.100.A.27' not in prose('<p>A glyph drawn in red with a circular contour. X.100.A.27.</p>')
    assert 'MC2-B-031' not in prose('<p>A glyph drawn in red with a circular contour. MC2-B-031.</p>')
    assert plain('<script>bad</script><p>good</p>')=='good'
    assert visual_sentences('Le glyphe représente une tête. Son histoire est longue.')=='Le glyphe représente une tête.'
    a=torch.tensor([[2.,0.],[0.,2.]]);alpha,_=choose_alpha(a,-a,torch.tensor([0,1]),['p1','p2']);assert alpha==0.
    assert torch.allclose(standardize(a).mean(1),torch.zeros(2));print('CHECK passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['sources','encode','evaluate','verify','check'],required=True);p.add_argument('--offline',action='store_true');p.add_argument('--model',choices=['clip','siglip2'],default='clip');args=p.parse_args()
    if args.phase=='sources':sources(args.offline)
    elif args.phase in ['encode','evaluate']:globals()[args.phase](args.model)
    else:globals()[args.phase]()
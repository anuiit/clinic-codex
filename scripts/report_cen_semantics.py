"""Render the completed CEN ablations; never change experiment inputs."""
from __future__ import annotations
import base64, collections, hashlib, html, io, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend/training_runs/cen-semantic-report-deps'),str(ROOT/'backend/training_runs/cen-semantic-deps'),str(ROOT),str(ROOT/'backend')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from scripts import research_recognition_mechanisms as mechanism
R=ROOT/'backend/training_runs/20261004-cen-semantics-v2'
OUT=ROOT/'reports/cen-semantic-20261004-v2/report'
OUT.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text())
source=load(OUT.parent/'class-text.json');names=[e['name'] for e in source['entries']]
reports={m:load(R/f'{m}-results.json') for m in ['clip','siglip2']}
verification=load(R/'verification.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path,digest in verification['artifact_sha256'].items():assert sha(Path(path))==digest
for path,digest in load(R/'preregistration.json')['input_sha256'].items():assert sha(Path(path))==digest
KINDS=['name','meaning','gdn','notice','visual']
TITLES={'name':'Noms seuls','meaning':'Sens courts CEN','gdn':'Définitions GDN','notice':'Notices Tlachia','visual':'Phrases de forme extraites'}

def paired(rows,reference):
    old={r['row_id']:r for r in reference};assert len(old)==len(reference)
    pages=collections.defaultdict(lambda:[0,0]);classes=collections.Counter()
    for r in rows:
        delta=int(r['prediction']==r['label'])-int(old[r['row_id']]['prediction']==r['label'])
        pages[r['group']][0]+=delta;pages[r['group']][1]+=1;classes[names[r['label']]]+=delta
    values=np.array(list(pages.values()));rng=np.random.default_rng(20261004)
    samples=values[rng.integers(0,len(values),(10000,len(values)))].sum(1)
    intervals=np.quantile(samples[:,0]/samples[:,1]*100,[.025,.975])
    return dict(delta_pp=float(values[:,0].sum()/values[:,1].sum()*100),cluster_bootstrap95_pp=intervals.tolist(),page_count=len(values),page_net=list(pages.values()),class_net=[dict(name=c,net=n) for c,n in classes.most_common() if n])

summary={'coverage':source['coverage'],'comparisons':{},'active_model_unchanged':True,'encoder_training':False,'dev_scored':False,'locked_test_scored':False}
for model,view,method in [('clip','raw','fusion_gdn'),('siglip2','mask','fusion_notice')]:
    d=reports[model];reference=d['predictions'][view]['dino_historical'];rows=d['predictions'][view][method]
    summary['comparisons'][model+'_'+view]=paired(rows,reference)
    summary['comparisons'][model+'_'+view+'_recomputed']=paired(rows,d['predictions'][view]['dino_recomputed'])
for model,d in reports.items():
    for view,methods in d['predictions'].items():
        for method,rows in methods.items():
            correct=sum(r['prediction']==r['label'] for r in rows);agg=d['aggregate'][view][method]
            assert correct==agg['correct'] and len(rows)==agg['n'] and len({r['row_id'] for r in rows})==len(rows)
            assert abs(correct/len(rows)-agg['accuracy'])<1e-6
summary['metrics']={m:d['aggregate'] for m,d in reports.items()}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
methods=[('DINO actuel','clip','dino_historical'),('DINO références recalculées','clip','dino_recomputed'),('DINO + sens / CLIP','clip','fusion_meaning'),('DINO + GDN / CLIP','clip','fusion_gdn'),('DINO + GDN / SigLIP2','siglip2','fusion_gdn'),('DINO + notices / SigLIP2','siglip2','fusion_notice')]
for ax,view,title in zip(axes,['raw','mask'],['Découpes annotées : 343 glyphes, 99 classes','Découpes SAM appariées : 254 glyphes, 74 classes']):
    vals=[reports[m]['aggregate'][view][method]['correct']/reports[m]['aggregate'][view][method]['n']*100 for _,m,method in methods]
    colors=['#394b59','#8599aa','#d19c44','#29826e','#578dce','#815cc1']
    ax.barh(range(len(vals)),vals,color=colors,height=.63);ax.set_yticks(range(len(vals)),[x[0] for x in methods]);ax.invert_yaxis();ax.set_xlim(0,80);ax.set_xlabel('Premier nom prédit correct (%)');ax.set_title(title,pad=15,fontsize=11);ax.set_axisbelow(True);ax.grid(axis='x',alpha=.18)
    for i,(v,(_,m,method)) in enumerate(zip(vals,methods)):
        metric=reports[m]['aggregate'][view][method];ax.text(v+.8,i,f'{v:.1f} %  ({metric["correct"]}/{metric["n"]})',va='center',fontsize=9)
fig.suptitle('Les textes CEN ajoutent un signal faible au modèle visuel',fontsize=15)
fig.supxlabel('Cinq plis sur 17 planches déjà explorées • encodeurs figés • aucune modification des boîtes ou du modèle actif',fontsize=9)
fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)

cache=torch.load(R/'clip-features.pt',weights_only=True,map_location='cpu');queries={(q['view'],q['row_id']):q for q in cache['queries']}
fixture=None

def crop(q):
    global fixture
    if q['view']=='raw':image=Image.open(q['path']).convert('RGB')
    else:
        if fixture is None:fixture=torch.load(mechanism.RUN/'crop-fixture.pt',weights_only=True,map_location='cpu')
        j=q['fixture_index'];image=Image.fromarray(mechanism.render_views(fixture['rgb'][j].numpy(),fixture['masks'][j].numpy())['mask'])
    image.thumbnail((220,180));b=io.BytesIO();image.save(b,format='PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

def examples(model,view,method):
    rows=reports[model]['predictions'][view][method];old={r['row_id']:r for r in reports[model]['predictions'][view]['dino_historical']}
    chosen=[]
    for label,want_fixed in ([('tlacatl',True),('calli',True),('cozamatl',False)] if view=='raw' else [('pantli',True),('tlacatl',True),('coltic',False)]):
        found=[r for r in rows if names[r['label']]==label and ((r['prediction']==r['label'] and old[r['row_id']]['prediction']!=r['label']) if want_fixed else (r['prediction']!=r['label'] and old[r['row_id']]['prediction']==r['label']))]
        assert found;chosen.append(found[0])
    cards=[]
    for r in chosen:
        e=source['entries'][r['label']];b=old[r['row_id']];fixed=r['prediction']==r['label'];definition=' '.join(((e['meanings']['fr'] or e['meanings']['es'] or [e['name']])[0]).split()[:20])
        cards.append(f'<article class="example"><img alt="Découpe du glyphe {html.escape(e["name"])}" src="{crop(queries[view,r["row_id"]])}"><strong>{"Corrigé" if fixed else "Dégradé"} · {html.escape(e["name"])}</strong><p>{html.escape(names[b["prediction"]])} → {html.escape(names[r["prediction"]])}</p><p class="muted">{html.escape(definition[:190])}</p><a href="{html.escape(e["gdn_url"],quote=True)}">Entrée CEN</a></article>')
    return ''.join(cards)

fmt=lambda x:f'{x*100:.1f}'.replace('.',',')+' %'
rows=[]
for model in ['clip','siglip2']:
    d=reports[model]
    for kind in KINDS:
        cells=[]
        for view in ['raw','mask']:
            met=d['aggregate'][view];x=met['fusion_'+kind];sh=met['fusion_'+kind+'_shuffled'];text=met['text_'+kind]
            cells+= [f'<td>{fmt(text["accuracy"])}</td>',f'<td>{fmt(x["accuracy"])}<small>{x["correct"]}/{x["n"]} · +{x["fixed"]}/−{x["broken"]}</small></td>',f'<td>{fmt(sh["accuracy"])}</td>']
        rows.append('<tr><th>'+('CLIP' if model=='clip' else 'SigLIP2')+' · '+TITLES[kind]+'</th>'+''.join(cells)+'</tr>')
perclass={}
for view,model,method in [('raw','clip','fusion_gdn'),('mask','siglip2','fusion_notice')]:
    d=reports[model]['aggregate'][view];baseline={r['name']:r for r in d['dino_historical']['per_class']};recomputed={r['name']:r for r in d['dino_recomputed']['per_class']}
    perclass[view]=[dict(name=r['name'],n=r['n'],base=baseline[r['name']]['correct'],visual=recomputed[r['name']]['correct'],semantic=r['correct'],delta=r['correct']-baseline[r['name']]['correct']) for r in d[method]['per_class']]
alpha_rows=[]
for model in ['clip','siglip2']:
    for kind in KINDS:
        cells=['<td>'+', '.join(str(f['views'][v]['selections'][kind]['alpha']) for f in reports[model]['folds'])+'</td>' for v in ['raw','mask']]
        alpha_rows.append('<tr><th>'+model+' · '+TITLES[kind]+'</th>'+''.join(cells)+'</tr>')
image_uri='data:image/png;base64,'+base64.b64encode((OUT/'comparison.png').read_bytes()).decode()
ci_raw=summary['comparisons']['clip_raw']['cluster_bootstrap95_pp'];ci_mask=summary['comparisons']['siglip2_mask']['cluster_bootstrap95_pp']
body='''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CEN et reconnaissance des glyphes · 4 octobre 2026</title>
<style>:root{font:16px/1.55 system-ui;color:#20313b;background:#f5f7f8}body{max-width:1150px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.2}h2{font-size:23px;margin-top:36px}p{max-width:950px}a{color:#176d62}small,.muted{font-size:13px;color:#546771}small{display:block}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{text-align:left;padding:11px;border-bottom:1px solid #dce3e7}thead{background:#e5edef}img.chart{width:100%;height:auto}section.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.example{background:white;border:1px solid #dde4e7;border-radius:12px;padding:16px}.example img{display:block;max-width:100%;height:180px;object-fit:contain;margin:auto}.example strong{display:block;margin-top:12px}.example p{margin:6px 0}.notice{padding:18px;background:#e4f0e9;border-left:4px solid #2d806a;border-radius:4px}.scroll{overflow:auto}label{display:inline-flex;gap:8px;align-items:center;margin:8px 15px 12px 0}select,input{font:inherit;padding:7px;border:1px solid #b6c4cc;border-radius:5px}footer{margin-top:40px;font-size:13px;color:#536873}@media(max-width:700px){body{padding:16px}section.cards{grid-template-columns:1fr}th,td{padding:8px}}</style>
<h1>Ce que les textes CEN apportent aux glyphes</h1><p>Expérience terminée le 4 octobre 2026. Référence linguistique : <a href="https://cen.sup-infor.com/home/hellow">CEN</a>, avec les définitions GDN et les notices pictographiques Tlachia.</p>
<div class="notice"><strong>Un signal utile, mais trop faible pour changer le modèle actif.</strong><p>Sur les découpes annotées, CLIP + GDN ajoute 12 bonnes réponses (54,2 % → 57,7 %), dont un gain net de 10 pour <em>tlacatl</em>. Les références visuelles recalculées atteignent déjà 57,1 %. Sur SAM, SigLIP2 + notices ajoute 3 bonnes réponses (63,0 % → 64,2 %), tandis que la moyenne par classe passe de 59,9 % à 59,5 %.</p></div>
<h2>Comparaison à images identiques</h2><img class="chart" alt="Précision des méthodes de reconnaissance sur les découpes annotées et SAM" src="__CHART__">
<p>À gauche : le glyphe est fourni par sa boîte annotée. À droite : le glyphe a déjà été trouvé et apparié à une annotation par SAM. Le test mesure uniquement le <strong>premier nom prédit, sans seuil de rejet</strong>. Les 89 annotations sans découpe SAM appariée ne figurent pas dans le dénominateur de droite. Ni localisation, ni rappel complet de la détection ne sont améliorés dans cet essai.</p>
<h2>Les textes utilisés</h2><p>Les 286 noms de classes sont reliés exactement aux entrées du catalogue Tlachia. Couverture : <strong>283 sens courts, 275 lemmes GDN, 286 notices et 260 extractions de phrases de forme</strong>. Les variantes peuvent partager une définition de lemme tout en conservant leur identifiant de classe distinct. Les variantes de sens sont conservées dans l’audit ; le sens le plus fréquent dans le catalogue est utilisé dans le prompt court.</p>
<p>Les notices sont débarrassées des listes d’occurrences et de leurs identifiants. Les phrases de forme sont extraites par mots-clés, avec un repli vers les autres textes lorsqu’elles manquent ; elles ne sont pas des descriptions validées par un expert. Les définitions GDN proviennent des réponses dont le lemme correspond exactement à la recherche. Les 566 captures HTTP utilisées ont été vérifiées par empreinte.</p>
<h2>Comment le test fonctionne</h2><ol><li>La même découpe entre dans l’encodeur d’image CLIP ou SigLIP2 ; son texte CEN entre dans l’encodeur de texte du même modèle.</li><li>Les similarités image–texte classent les 286 éléments. Les noms, sens courts, définitions, notices et phrases de forme sont testés séparément.</li><li>Une fusion ajoute ce score au score DINO actuel : (1 − α) × score visuel + α × score texte. Les deux scores sont normalisés par découpe avant la fusion.</li><li>Le poids α est choisi sur les autres planches d’entraînement, avec exclusion des groupes, composants de sources et pixels tenus à l’écart. Cinq plis donnent une prédiction par découpe.</li><li>Un témoin mélange les textes entre les classes et bénéficie de la même calibration. Un autre recalcule les références DINO à partir des seuls exemples autorisés dans chaque pli.</li></ol>
<p><strong>Les encodeurs restent figés.</strong> Aucune descente de gradient, aucun apprentissage image–texte sur nos glyphes, aucune génération de nouvelles boîtes. Le corpus historique sert au comparateur visuel recalculé ; il n’a pas servi à entraîner CLIP ou SigLIP2 dans cet essai.</p>
<h2>Tous les essais</h2><p>« Texte seul » signifie une comparaison image–texte sans DINO. « Fusion » combine DINO et texte CEN. Le témoin utilise des textes permutés. +/− compte les bonnes réponses ajoutées et perdues face au DINO actuel.</p><div class="scroll"><table><thead><tr><th rowspan="2">Modèle et texte</th><th colspan="3">Découpes annotées · n = 343</th><th colspan="3">Découpes SAM · n = 254</th></tr><tr><th>Texte seul</th><th>Fusion</th><th>Témoin</th><th>Texte seul</th><th>Fusion</th><th>Témoin</th></tr></thead><tbody>__RESULTS__</tbody></table></div>
<p>Les modèles image–texte seuls atteignent au mieux <strong>4,1 %</strong> sur les découpes annotées et <strong>2,8 %</strong> sur SAM. CLIP + GDN corrige 15 erreurs et en crée 3 ; SigLIP2 + notices corrige 4 erreurs et en crée 1. Le témoin GDN permuté CLIP reste à 54,2 % sur les découpes annotées ; le témoin notices SigLIP2 atteint 62,2 % sur SAM. Un seul tirage de permutation a été testé.</p>
<h2>Exemples de décisions modifiées</h2><p>Découpes annotées · DINO + GDN / CLIP.</p><section class="cards">__RAW_EXAMPLES__</section><p>Découpes SAM · DINO + notices / SigLIP2.</p><section class="cards">__MASK_EXAMPLES__</section>
<h2>Classe par classe</h2><p>Le tableau contient seulement les classes présentes dans les annotations évaluées : 99 pour les découpes annotées, 74 pour SAM. Une classe absente ne peut pas recevoir une précision mesurée. Les pourcentages avec peu d’exemples sont fragiles.</p><label>Découpes <select id="view"><option value="mask">SAM · fusion notices SigLIP2</option><option value="raw">Annotées · fusion GDN CLIP</option></select></label><label>Classe <input id="search" placeholder="Rechercher un nom"></label><label>Minimum d’exemples <input id="minimum" type="number" min="1" value="1" style="width:70px"></label><div class="scroll"><table><thead><tr><th>Classe</th><th>n</th><th>DINO actuel</th><th>DINO recalculé</th><th>DINO + CEN</th><th>Gain net</th></tr></thead><tbody id="classes"></tbody></table></div>
<h2>Ce que l’on peut conclure</h2><p>Les définitions ajoutent un signal mesurable à certaines décisions, surtout <em>tlacatl</em> dans le test CLIP. Elles ne produisent pas une progression large entre classes. Sur SAM, trois corrections de <em>pantli</em> et une de <em>tlacatl</em> compensent une erreur nouvelle de <em>coltic</em>.</p>
<p>Le bootstrap descriptif des 17 planches donne un intervalle à 95 % de <strong>__RAW_CI__ points</strong> pour le gain CLIP GDN et de <strong>__MASK_CI__ points</strong> pour SigLIP2 notices. Il rééchantillonne les prédictions déjà calculées, sans refaire la calibration. Ces intervalles ne corrigent ni la sélection du meilleur essai parmi les variantes, ni l’exploration répétée des mêmes planches.</p>
<p>La projection DINO historique a une provenance d’entraînement incomplète ; ce lot n’est pas un test aveugle inédit. Les annotations de boîtes sont partielles. Seules quatre classes évaluées ont 1 à 4 images d’entraînement distinctes selon le pli : ce sous-groupe est trop petit pour valider une amélioration des classes rares. Face au DINO recalculé, le gain CLIP GDN se limite à +0,58 point, avec un intervalle descriptif de −0,81 à +1,96 point. Les nombres détaillés sont conservés dans summary.json.</p>
<h2>La suite que je recommande</h2><p>Conserver CEN comme référence des noms et des définitions. Pour tester un réel apprentissage sémantique, aligner nos images étiquetées avec ces textes par une petite adaptation contrastive, en gardant une supervision des identifiants de classe pour les variantes proches. Comparer cette adaptation à DINO recalculé sur les mêmes exclusions. Cela reste un essai à faire ; les résultats ci-dessus concernent uniquement des encodeurs figés.</p><p>En parallèle, compléter les boîtes métier : les textes ne fournissent pas de supervision de placement. Le modèle actif reste inchangé.</p>
<details><summary>Poids texte choisis dans chacun des cinq plis</summary><div class="scroll"><table><thead><tr><th>Texte</th><th>Découpes annotées</th><th>Découpes SAM</th></tr></thead><tbody>__ALPHAS__</tbody></table></div></details>
<footer>Sources : <a href="https://cen.sup-infor.com/home/hellow">CEN</a> · <a href="https://huggingface.co/openai/clip-vit-base-patch32">CLIP officiel</a> · <a href="https://huggingface.co/google/siglip2-base-patch16-224">SigLIP2 officiel</a>. Rapport dérivé du run 20261004-cen-semantics-v2 ; inputs et modèle actif vérifiés inchangés. Le brouillon v1 a été remplacé avant toute évaluation.</footer>
<script>const data=__CLASSES__;function render(){const view=document.getElementById('view').value,q=document.getElementById('search').value.toLowerCase(),minimum=Math.max(1,Number(document.getElementById('minimum').value)||1);const target=document.getElementById('classes');target.replaceChildren();for(const r of data[view].filter(x=>x.n>=minimum&&x.name.toLowerCase().includes(q)).sort((a,b)=>b.n-a.n||a.name.localeCompare(b.name))){const tr=document.createElement('tr');const pct=x=>(100*x/r.n).toFixed(1).replace('.',',')+' % ('+x+'/'+r.n+')';for(const value of [r.name,r.n,pct(r.base),pct(r.visual),pct(r.semantic),(r.delta>0?'+':'')+r.delta]){const td=document.createElement('td');td.textContent=value;tr.append(td)}target.append(tr)}}for(const id of ['view','search','minimum'])document.getElementById(id).addEventListener('input',render);render();</script></html>'''
ci=lambda x:' à '.join(f'{v:+.2f}'.replace('.',',') for v in x)
for key,value in {'__CHART__':image_uri,'__RESULTS__':''.join(rows),'__RAW_EXAMPLES__':examples('clip','raw','fusion_gdn'),'__MASK_EXAMPLES__':examples('siglip2','mask','fusion_notice'),'__CLASSES__':json.dumps(perclass,ensure_ascii=False).replace('<','\\u003c'),'__ALPHAS__':''.join(alpha_rows),'__RAW_CI__':ci(ci_raw),'__MASK_CI__':ci(ci_mask)}.items():body=body.replace(key,value)
assert '__CHART__' not in body and body.count('<article class="example">')==6
(OUT/'index.html').write_text(body,encoding='utf-8')
print('REPORT',OUT,flush=True);print('COMPARISONS',json.dumps(summary['comparisons'],ensure_ascii=False),flush=True)
"""Render auditable comparison of both fixed YOLO pilot durations."""
import collections,json,statistics
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from scripts import research_synthetic_detector as s
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=s.ROOT;REPORTS=ROOT/'reports'
LONG=ROOT/'backend/training_runs/20261004-yolo-synthetic-1000-v1'
READOUTS={200:s.RUN/'recognition/results.json',1000:LONG/'recognition/results.json'}
DETECTORS={200:s.RUN/'detection-results.json',1000:LONG/'detection-results.json'}


def per_class(pages,view=None):
    counts=collections.defaultdict(lambda:dict(known=0,detected=0,correct=0))
    for p in pages:
        evaluation=p['methods']['gated']['evaluations']['rare_branch'] if view is None else p['views'][view]['rare_branch']
        matched={r['gt_index']:r for r in evaluation['matches']}
        for index,gt in enumerate(p['gt']):
            row=counts[gt['class_name']];row['known']+=1;row['detected']+=index in matched;row['correct']+=index in matched and matched[index]['correct']
    return dict(counts)


def localization(pages):
    values=[]
    for p in pages:
        chosen=[r for r in p['predictions'] if r['detector_confidence']>=p['selected_threshold']]
        for gt in p['gt']:
            values.append(max([s.f.box_iou(s.f.box(gt['bbox'])['box'],s.f.box(r['bbox'])['box']) for r in chosen],default=0.))
    return dict(known=len(values),best_candidate_iou_ge025=sum(v>=.25 for v in values),best_candidate_iou_ge05=sum(v>=.5 for v in values),best_candidate_iou_ge075=sum(v>=.75 for v in values),mean_best_iou=statistics.mean(values),note='Diagnostic best overlap, not one-to-one recall; one prediction may cover several GT. No threshold selection.')


def pct(value):return f'{100*value:.1f} %'


def main():
    results={steps:s.read(p) for steps,p in READOUTS.items()};detectors={steps:s.read(p) for steps,p in DETECTORS.items()}
    branch=s.read(s.BRANCH);base=results[200]['reference'];basepages=[p for p in branch['outer_pages'] if p['seed']==42]
    classstats={'sam':per_class(basepages)};local={};fitstats={};geotable=[];rectable=[]
    for steps,out in results.items():
        det=detectors[steps];s.verify(out['protocol']);s.verify(det['protocol'])
        assert len(det['fits'])==10 and det['optimizer_updates']==10*steps
        for arm in ['real','real_plus_synthetic']:
            fs=[f for f in det['fits'] if f['arm']==arm];ps=[p for f in out['fits'] if f['arm']==arm for p in f['pages']]
            assert len(ps)==17 and sum(len(p['gt']) for p in ps)==343
            geo=out['geometry'][arm];geotable.append(dict(steps=steps,arm=arm,**geo))
            for view in ['raw','mask']:
                classstats[f'{steps}_{arm}_{view}']=per_class(ps,view)
                for policy in ['current_035','rare_branch']:
                    z=out['recognition'][arm][view][policy];c=z['counts']
                    assert c['correct']<=c['matched']<=343 and c['scored_predictions']==c['matched']+c['unmatched_predictions']
                    rectable.append(dict(steps=steps,arm=arm,view=view,policy=policy,**z))
            local[f'{steps}_{arm}']=localization([dict(p,selected_threshold=f['selection']['threshold']) for f in fs for p in f['outer_pages']])
            fitstats[f'{steps}_{arm}']=dict(seconds=sum(f['fit_seconds'] for f in fs),real_exposures=sum(f['real_image_exposures'] for f in fs),synthetic_exposures=sum(f['synthetic_image_exposures'] for f in fs),mean_first50_real_loss=statistics.mean(row['real_loss'] for f in fs for row in f['history'][:50]),mean_last50_real_loss=statistics.mean(row['real_loss'] for f in fs for row in f['history'][-50:]),thresholds=[f['selection']['threshold'] for f in fs],fold_detected=[sum(p['geometry']['counts']['matched'] for p in f['outer_pages']) for f in fs])
    summary=dict(reference=base,geometry=geotable,recognition=rectable,per_class=classstats,localization_diagnostic=local,fit=fitstats,run_sha256={str(p):s.sha(p) for p in list(READOUTS.values())+list(DETECTORS.values())},detector_optimizer_updates=12000,activated=False,locked_test_scored=False)
    s.write(REPORTS/'yolo-synthetic-summary-20261004.json',summary)
    labels=['SAM']+[f'YOLO {z["steps"]}\n'+('réel' if z['arm']=='real' else 'réel + synth.') for z in geotable]
    g=[base['sam_geometry']]+geotable
    fig,axes=plt.subplots(1,2,figsize=(14,5));colors=['#44546a','#247c98','#78afbc','#9b6f24','#d4ae6a']
    for metric,title,ax in [('detection_recall','Glyphes avec une bonne boîte',axes[0]),('detection_f1','F1 des boîtes (précision + rappel)',axes[1])]:
        vals=[100*z['metrics'][metric] for z in g];bars=ax.bar(labels,vals,color=colors);ax.set_ylim(0,100);ax.set_ylabel('%');ax.set_title(title)
        for bar,value in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.1f}',ha='center')
    fig.suptitle('17 planches déjà examinées · 343 glyphes · seed 42 · IoU ≥ 0,5',fontsize=15);fig.tight_layout()
    fig.savefig(REPORTS/'yolo-synthetic-geometry-20261004.png',dpi=160);plt.close(fig)
    chosen=[z for z in rectable if z['policy']=='rare_branch'];labels=['SAM candidat']+[f'{z["steps"]} '+('réel' if z['arm']=='real' else '+ synth.')+'\n'+('crop' if z['view']=='raw' else 'masque SAM') for z in chosen]
    fig,ax=plt.subplots(figsize=(13,5));values=[100*base['sam_branch']['metrics']['exact_f1']]+[100*z['metrics']['exact_f1'] for z in chosen]
    bars=ax.bar(labels,values,color=['#44546a']+['#247c98' if z['arm']=='real' else '#d4ae6a' for z in chosen]);ax.set_ylim(0,60);ax.set_ylabel('F1 exact (%)');ax.set_title('Bonne boîte + bon nom · seuils sémantiques de la branche historique conservés')
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.5,f'{value:.1f}',ha='center')
    fig.tight_layout();fig.savefig(REPORTS/'yolo-synthetic-recognition-20261004.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for steps,ax in zip([200,1000],axes):
        for arm,color in [('real','#247c98'),('real_plus_synthetic','#d4ae6a')]:
            histories=[f['history'] for f in detectors[steps]['fits'] if f['arm']==arm]
            means=[statistics.mean(h[i]['real_loss'] for h in histories) for i in range(steps)]
            smooth=[statistics.mean(means[max(0,i-19):i+1]) for i in range(steps)]
            ax.plot(range(1,steps+1),smooth,color=color,label='réel' if arm=='real' else 'réel + synthétique')
        ax.set_title(f'{steps} mises à jour · moyenne 5 partitions');ax.set_xlabel('Mise à jour');ax.set_ylabel('Perte sur batch réel (moyenne mobile20)');ax.legend()
    fig.tight_layout();fig.savefig(REPORTS/'yolo-synthetic-loss-20261004.png',dpi=160);plt.close(fig)
    data=s.read(s.RUN/'dataset.json');sp=data['splits'][0];preview=Image.new('RGB',(1536,812),'white')
    draw=ImageDraw.Draw(preview);font=ImageFont.load_default(size=19)
    for slot,row in enumerate([sp['real'][0],sp['synth'][0]]):
        im=Image.open(row['path']).convert('RGB');pen=ImageDraw.Draw(im)
        for b in row['ignore_boxes']:pen.rectangle(b,outline='#dd9700',width=2)
        for b in row['boxes']:pen.rectangle(b,outline='#bb1924',width=2)
        preview.paste(im,(slot*768,44))
    draw.text((15,10),'Vraie tuile : rouge = cibles ; orange = ignorees / autre tuile',fill='#222222',font=font)
    draw.text((785,10),'Synthetique : crops legacy + fond reel pauvre en encre',fill='#222222',font=font)
    preview.save(REPORTS/'yolo-synthetic-data-legend-20261004.png')
    # Preselected illustration: second held page in fold0, no choice based on its score.
    group=detectors[200]['fits'][0]['outer_pages'][1]['group'];page=detectors[200]['fits'][0]['outer_pages'][1]
    source=Image.open(page['image']).convert('RGB');scale=min(520/source.width,850/source.height);size=(round(source.width*scale),round(source.height*scale))
    entries=[('Annotation métier',page['gt'])]
    for steps in [200,1000]:
        for arm in ['real','real_plus_synthetic']:
            fit=next(f for f in detectors[steps]['fits'] if f['fold']==0 and f['arm']==arm);p=next(p for p in fit['outer_pages'] if p['group']==group)
            entries.append((f'{steps} '+('réel' if arm=='real' else '+ synth.'),[r for r in p['predictions'] if r['detector_confidence']>=fit['selection']['threshold']]))
    example=Image.new('RGB',(5*(size[0]+12),size[1]+45),'white');pen=ImageDraw.Draw(example)
    for j,(title,rows) in enumerate(entries):
        im=source.resize(size);d=ImageDraw.Draw(im)
        for r in rows:
            x,y,w,h=r['bbox'];d.rectangle([x*scale,y*scale,(x+w)*scale,(y+h)*scale],outline='#13644c' if j==0 else '#bd2430',width=2)
        example.paste(im,(j*(size[0]+12),35));pen.text((j*(size[0]+12)+4,10),title,fill='#222222',font=font)
    example.save(REPORTS/'yolo-synthetic-boxes-20261004.png')
    best=max((z for z in rectable if z['policy']=='rare_branch'),key=lambda z:z['metrics']['exact_f1'])
    verdict=('Aucun variant testé ne dépasse le candidat SAM en F1 exact.' if best['metrics']['exact_f1']<=base['sam_branch']['metrics']['exact_f1'] else 'Un variant dépasse le candidat SAM en F1 exact sur ce readout exploratoire ; les compromis de rappel et les limites restent à examiner.')
    diagnostic=s.read(ROOT/'backend/training_runs/20261004-yolo-train-diagnostic-v1/results.json')
    summary['verdict']=verdict;summary['geometry_only_note']='Name fields are placeholders in geometry-only readout; evaluate matched/missed and detection metrics there. Recognition results contain actual class predictions.'
    summary['train_diagnostic']=diagnostic; s.write(REPORTS/'yolo-synthetic-summary-20261004.json',summary)
    lines=['# Expériences YOLO : vraies annotations et crops synthétiques','',verdict,'',
           'Deux comparaisons figées ont été terminées : **20 entraînements de détecteur, 12 000 mises à jour au total**. Une seule graine (42), cinq partitions identiques par comparaison. Aucune activation et aucune évaluation dev/test dans cette campagne.','',
           'Les « 400 et quelques annotations » sont des boîtes d’objets, pas 400 planches annotées. Ce readout utilise 343 glyphes connus sur 17 planches et 7 régions à ignorer. Les 9 259 anciens crops sans coordonnées peuvent superviser la reconnaissance ; ils ne donnent pas directement les positions dans une vraie page.','',
           '## Recette et inférence','',
           'YOLO11n COCO préentraîné, une classe « glyphe », tous les paramètres appris ; statistiques BatchNorm figées. AdamW, pic 5e-4, fin 5e-5, warmup20, cosine, poids decay5e-4, clip5. Deux vraies tuiles par mise à jour dans les deux bras. Le bras mixte ajoute deux images synthétiques et la moitié de leur perte moyenne : même exposition réelle, davantage de calcul.','',
           'Chaque partition possède 64 pages synthétiques de16 objets (1 024 positions connues), construites exclusivement avec les crops legacy du train autorisé : groupes, composants et pixels de calibration/évaluation exclus avant la construction. Les 286 classes sont présentes dans les pools éligibles. Fond réel sans boîte et pauvre en encre, masques estimés par couleur/contraste : cela produit de nouvelles coordonnées synthétiques, pas les anciennes coordonnées perdues.','',
           'À l’inférence : tuiles768/stride512 + page entière → YOLO → NMS0,5 → seuil de confiance choisi uniquement sur les trois pages de calibration de la partition → rectangle brut ou masque SAM demandé avec cette boîte → DINOv2 figé → projection paired propre à la partition, prototypes historiques et garde des classes rares → seuil sémantique0,35 ou branche historique. Aucune boîte vérité terrain dans les prompts ou les crops. SAM n’est pas réentraîné dans cette campagne.','',
           'Le premier brouillon synthétique contenait des montages external sous une seule boîte et du texte agrandi dans les fonds. Il a été écarté avant toute optimisation. Le générateur conservé utilise seulement legacy et des fonds pauvres en encre. Un bug technique du chargement YOLO a également été corrigé avant le premier update ; source et protocole antérieurs conservés.','',
           '## Boîtes seules','',
           '| Détecteur | Bonnes boîtes /343 | Sorties évaluées | Précision | Rappel | F1 |','|---|---:|---:|---:|---:|---:|']
    for label,z in [('SAM automatique',base['sam_geometry'])]+[(f'YOLO {z["steps"]} — {z["arm"]}',z) for z in geotable]:
        c=z['counts'];m=z['metrics'];lines.append(f'| {label} | {c["matched"]} | {c["scored_predictions"]} | {pct(m["detection_precision"])} | {pct(m["detection_recall"])} | {pct(m["detection_f1"])} |')
    lines+=['','![Boîtes](yolo-synthetic-geometry-20261004.png)','','## Bonne boîte et bon nom','','Les seuils sémantiques existants sont conservés. Leur calibration avait été faite sur les propositions SAM ; ce transfert vers YOLO peut être imparfait. Tous les variants prévus sont publiés, sans choisir un vainqueur à partir des pages évaluées.','',
            '| Détecteur | Crop | Filtre | Corrects /343 | Précision exacte | Rappel exact | F1 exact |','|---|---|---|---:|---:|---:|---:|']
    for label,policy,z in [('SAM candidat','0,35',base['sam_gated_035']),('SAM candidat','branche',base['sam_branch'])]:
        m=z['metrics'];lines.append(f'| {label} | masque | {policy} | {z["counts"]["correct"]} | {pct(m["exact_precision"])} | {pct(m["exact_recall"])} | {pct(m["exact_f1"])} |')
    for z in rectable:
        m=z['metrics'];lines.append(f'| YOLO {z["steps"]} {z["arm"]} | {z["view"]} | {z["policy"]} | {z["counts"]["correct"]} | {pct(m["exact_precision"])} | {pct(m["exact_recall"])} | {pct(m["exact_f1"])} |')
    lines+=['','![Reconnaissance](yolo-synthetic-recognition-20261004.png)','','## Par classe','','Corrects après filtre branche ; masque SAM pour les variantes YOLO ci-dessous. Un faible effectif donne une estimation très instable.','',
            '| Classe | Objets | SAM candidat | YOLO200 réel | YOLO200+synth | YOLO1000 réel | YOLO1000+synth |','|---|---:|---:|---:|---:|---:|---:|']
    names=sorted(classstats['sam'],key=lambda name:(-classstats['sam'][name]['known'],name))
    for name in names[:20]:
        keys=['sam','200_real_mask','200_real_plus_synthetic_mask','1000_real_mask','1000_real_plus_synthetic_mask']
        values=[classstats[key][name]['correct'] for key in keys];n=classstats['sam'][name]['known'];lines.append('| '+name+' | '+str(n)+' | '+' | '.join(f'{v}/{n}' for v in values)+' |')
    lines+=['','## Entraînement et limites','','![Pertes](yolo-synthetic-loss-20261004.png)','','| Essai | Temps de fit (5 folds) | Expositions réelles | Expositions synthétiques | Seuils par fold |','|---|---:|---:|---:|---|']
    for key,z in fitstats.items():lines.append(f'| {key} | {z["seconds"]:.1f} s | {z["real_exposures"]} | {z["synthetic_exposures"]} | {z["thresholds"]} |')
    lines+=['','## Vérification train / pages exclues sur la partition0','','Diagnostic postérieur, partition0 fixée avant lecture ; mêmes seuils sélectionnés sur calibration. Les cohortes train et outer ont une composition différente, cette comparaison documente un écart entre train et pages exclues et ne constitue pas une nouvelle validation.','','| Essai | Train : bonnes boîtes | Rappel train | Pages exclues : bonnes boîtes | Rappel exclu |','|---|---:|---:|---:|---:|']
    for z in diagnostic['results']:
        t=z['train'];o=z['outer'];lines.append(f'| {z["steps"]} {z["arm"]} | {t["counts"]["matched"]}/{t["counts"]["known"]} | {pct(t["metrics"]["detection_recall"])} | {o["counts"]["matched"]}/{o["counts"]["known"]} | {pct(o["metrics"]["detection_recall"])} |')
    lines+=['','Le prolongement à1 000 est adaptatif : il a été décidé après le pilote200, puis gelé dans un nouveau dossier. Il modifie également l’horizon du cosine ; ce n’est pas une ablation causale de la durée seule. Les deux résultats restent publiés.','',
            'Ces 17 pages ont déjà été examinées dans les recherches précédentes : partitions exclues des adaptations, mais aucune nouvelle validation aveugle. Provenance historique des projections/prototypes imparfaitement connue. Une seule graine de détecteur, petite diversité de vraies pages, masques synthétiques approximatifs, calcul supplémentaire du bras synthétique, et recette de fit spécifique. Ces résultats évaluent cette recette ; ils ne démontrent pas que toute architecture YOLO serait inefficace.','',
            'Les vérifications couvrent les hashes des données/sources/poids, initialisation égale des bras, plans de batches, exclusions, vrai forward/backward GPU (gradients ignorés nuls et positifs conservés), checkpoints, recomptage des matches et fichiers actifs inchangés. Le SHA du poids YOLO est un pin local téléchargé depuis la release officielle ; l’amont ne publie pas de digest.','',
            '![Données](yolo-synthetic-data-legend-20261004.png)','','![Exemple de boîtes](yolo-synthetic-boxes-20261004.png)','',
            'Sources de méthode : [YOLO11](https://docs.ultralytics.com/models/yolo11), [Cut, Paste and Learn](https://arxiv.org/abs/1708.01642). Ces sources motivent le test ; elles ne prédisent pas les performances sur nos glyphes.','']
    (REPORTS/'yolo-synthetic-experiments-20261004.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(geometry=geotable,fit=fitstats),indent=2))

if __name__=='__main__':main()





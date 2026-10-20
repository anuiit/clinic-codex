"""Verify completed crop-compatible fits and publish all fixed checkpoints."""
import collections,csv,json,statistics,hashlib
from pathlib import Path
import torch
from scripts.experiments import research_crop_compatible as driver
from scripts.experiments import summarize_recognition_mechanisms as s
ROOT=driver.core.ROOT
RUN=driver.RUN
OUT=ROOT/'reports/crop-compatible-results-20261004.json'
ABLATION=ROOT/'backend/training_runs/20261004-crop-compatible-no-kl-v2'
METHODS=driver.METHODS+(['paired_no_kl'] if (ABLATION/'results.json').exists() else [])

def checkpoint(method,fold,seed,step):
    folder=ABLATION if method=='paired_no_kl' else RUN
    name='paired' if method=='paired_no_kl' else method
    return folder/f'{name}_fold{fold}_seed{seed}_step{step}.pt'


def guards(entries,method,baseline,step,candidate_summary,baseline_summary):
    deltas={str(f):statistics.mean(e['methods'][method]['milestones'][str(step)]['evaluation']['raw']['top1_micro']-(e['baseline'] if baseline=='historical' else e['methods'][baseline]['milestones'][str(step)]['evaluation'])['raw']['top1_micro'] for e in entries if e['fold']==f) for f in range(5)}
    checks=dict(page_mean_improves=candidate_summary['page_mean_joint']>baseline_summary['page_mean_joint'],rare_macro_no_regression=candidate_summary['rare_macro']>=baseline_summary['rare_macro'],every_fold_raw_drop_at_most2pp=min(deltas.values())>=-.02)
    return dict(checks=checks,eligible=all(checks.values()),fold_mean_raw_deltas=deltas)



@torch.no_grad()
def crop_shift(entries,cache):
    torch.set_num_threads(4)
    f=torch.load(driver.m.RUN/'crop-fixture.pt',weights_only=True,map_location='cpu')
    init=torch.load(driver.core.RUNTIME/'weights/projection.pt',weights_only=True,map_location='cpu')
    proto=driver.core.aligned_prototypes(torch.load(driver.core.RUNTIME/'weights/prototypes.pt',weights_only=True,map_location='cpu'),cache['class_names'])
    out={};model=driver.core.ProjectionHead();rawmeta={r['row_id']:r for r in csv.DictReader((driver.core.SNAPSHOT/'metadata.csv').open())}
    for method in ['historical']+METHODS:
        records={}
        for seed in driver.SEEDS:
            distances=[];counts=collections.Counter();rare_known=collections.Counter();rare_correct=collections.Counter();rare_matched=0
            for e in entries:
                if e['seed']!=seed:continue
                ids=[i for i,r in enumerate(f['rows']) if r['split']=='train' and r['source_group'] in e['held']]
                raw=[f['rows'][i]['snapshot_index'] for i in ids];labels=cache['labels'][raw]
                state=init if method=='historical' else torch.load(checkpoint(method,e['fold'],seed,1000),weights_only=True,map_location='cpu')['projection']
                model.load_state_dict(state);model.eval()
                a=model(cache['features'][raw].float());b=model(f['features']['mask'][ids].float())
                distances.extend((1-(a*b).sum(1)).tolist())
                pa=(a@proto.T).argmax(1);pb=(b@proto.T).argmax(1)
                ca=pa==labels;cb=pb==labels
                expected=e['baseline'] if method=='historical' else e['methods'][method]['milestones']['1000']['evaluation']
                assert int(cb.sum())==expected['masked']['top1_correct']
                train_counts=s.support_counts([rawmeta[cache['row_ids'][i]] for i in e['selected_raw_ids']])
                for r in expected['raw_predictions']:
                    if 1<=train_counts.get(r['label'],0)<=4:rare_known[r['label']]+=1
                for j,label in enumerate(labels.tolist()):
                    if 1<=train_counts.get(label,0)<=4:
                        rare_matched+=1
                        if bool(cb[j]):rare_correct[label]+=1

                counts.update(dict(matched=len(ids),matched_raw_correct=int(ca.sum()),mask_correct=int(cb.sum()),both_correct=int((ca&cb).sum()),raw_correct_mask_wrong=int((ca&~cb).sum()),raw_wrong_mask_correct=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),prediction_changes=int((pa!=pb).sum())))
            assert counts['matched']==254
            records[str(seed)]=dict(counts,mean_cosine_distance=statistics.mean(distances),rare_known_count=sum(rare_known.values()),rare_matched_count=rare_matched,rare_sam_correct=sum(rare_correct.values()),rare_sam_macro_joint=statistics.mean(rare_correct[c]/n for c,n in rare_known.items()))
        out[method]=dict(seeds=records,mean_cosine_distance=statistics.mean(e['mean_cosine_distance'] for e in records.values()),mean_raw_correct_mask_wrong=statistics.mean(e['raw_correct_mask_wrong'] for e in records.values()))
    return out


def main():
    p=s.read(RUN/'preregistration.json');driver.verify(p)
    data=s.read(RUN/'results.json');entries=data['folds']
    assert len(entries)==15 and {(e['fold'],e['seed']) for e in entries}=={(f,seed) for f in range(5) for seed in driver.SEEDS}
    if 'paired_no_kl' in METHODS:
        q=s.read(ABLATION/'preregistration.json');driver.verify(q)
        ab=s.read(ABLATION/'results.json')['folds'];assert len(ab)==15
        for e in entries:
            r=next(z for z in ab if z['fold']==e['fold'] and z['seed']==e['seed'])
            assert r['held']==e['held'] and r['selected_raw_ids']==e['selected_raw_ids'] and r['baseline']==e['baseline']
            e['methods'].update(r['methods'])
            assert all(h['weighted_kl']==0 for h in e['methods']['paired_no_kl']['history'])
            for step in ['300','1000']:assert e['methods']['paired_no_kl']['milestones'][step]['extra_pair_forwards']==e['methods']['paired']['milestones'][step]['extra_pair_forwards']
    cache=torch.load(driver.core.CACHE,weights_only=True,map_location='cpu');index=cache['row_ids'];labels=cache['labels'].tolist()
    meta={r['row_id']:r for r in csv.DictReader((driver.core.SNAPSHOT/'metadata.csv').open())}
    coverage=[];times=[]
    for e in entries:
        hashes={r['batch_sha256'] for r in e['methods'].values()};assert len(hashes)==1
        old=s.read(driver.m.RUN/'contrast-results.json')
        prior=next(r for r in old['folds'] if r['fold']==e['fold'] and r['seed']==e['seed'])['methods']['ce_fixed']['evaluation']
        assert e['methods']['ce']['milestones']['300']['evaluation']==prior
        for method,r in e['methods'].items():
            assert len(r['history'])==1000
            if method=='ce':assert r['reference300_reproduction']['predictions_and_metrics_equal']
            for step,mile in r['milestones'].items():
                cp=checkpoint(method,e['fold'],e['seed'],step)
                assert s.sha(cp)==mile['checkpoint_sha256']
                assert mile['exposure']==e['methods']['ce']['milestones'][step]['exposure']
                assert abs(sum(mile['ce_class_mass'].values())-int(step))<.01
                assert not mile['extra_pair_forwards'] or method in ['paired','paired_no_kl']
            raw=e['selected_raw_ids'];counts=s.support_counts([meta[index[i]] for i in raw]);last=r['milestones']['1000']
            seen=[int(i) for i in last['exposure']];assert {labels[i] for i in seen if i<len(index)}==set(range(286))
            assert r['unique_rows_seen']==len(seen)<=r['eligible_rows']
            assert set(e['anchor_row_ids'])<=set(r['train_row_ids']) and not set(e['held']).intersection(meta[row]['source_group'] for row in e['anchor_row_ids'])
            rare=[c for c,n in counts.items() if 1<=n<=4]
            coverage.append(dict(fold=e['fold'],seed=e['seed'],method=method,eligible_rows=r['eligible_rows'],unique_rows_seen=r['unique_rows_seen'],ce_forwards=sum(last['exposure'].values()),
                pair_forwards=sum(last['extra_pair_forwards'].values()),rare_classes=len(rare),mean_rare_class_weight_mass=statistics.mean(last['ce_class_mass'].get(str(c),0) for c in rare),
                teacher_anchor_row_ids_sha256=hashlib.sha256('\n'.join(sorted(e['anchor_row_ids'])).encode()).hexdigest(),teacher_correct_parent_objects=len(e['anchor_row_ids']),teacher_correct_parent_classes=len({int(meta[row]['class_label']) for row in e['anchor_row_ids']})))
            times.append(r['elapsed_seconds'])
    assert len(list(RUN.glob('*_step1000.pt')))==60 and len(list(RUN.glob('*_step300.pt')))==60
    assert len(times)==15*len(METHODS)
    if 'paired_no_kl' in METHODS:assert len(list(ABLATION.glob('*_step1000.pt')))==len(list(ABLATION.glob('*_step300.pt')))==15
    historical=s.across_seeds([s.aggregate([(e,e['baseline']) for e in entries if e['seed']==seed],meta,index) for seed in driver.SEEDS])
    results={}
    for step in driver.MILESTONES:
        methods={method:s.across_seeds([s.aggregate([(e,e['methods'][method]['milestones'][str(step)]['evaluation']) for e in entries if e['seed']==seed],meta,index) for seed in driver.SEEDS]) for method in METHODS}
        for method,value in methods.items():
            value['paired_vs_historical']=s.paired_pages(value,historical);value['paired_vs_ce']=s.paired_pages(value,methods['ce'])
            value['guard_vs_historical']=guards(entries,method,'historical',step,value,historical)
            value['guard_vs_ce']=guards(entries,method,'ce',step,value,methods['ce'])
            h=[row for e in entries for row in e['methods'][method]['history'][step-20:step]]
            value['last20_losses']={k:statistics.mean(row[k] for row in h) for k in ['ce','kl','weighted_kl','pair','loss']}
            g=[e['methods'][method]['history'][step-1] for e in entries]
            value['gradient_ratios']={k:statistics.mean(row.get(k,0)/row['ce_gradient_norm'] for row in g) for k in ['weighted_kl_gradient_norm','pair_gradient_norm']}
        results[str(step)]=methods
    assert results['300']['ce']['masked_correct']==[170,171,172]
    verification=dict(input_and_active_hashes_unchanged=True,checkpoint_hashes_passed=True,ce300_metrics_and_predictions_reproduced_15_fits=True,ce_batch_and_exposure_equal=True,all286_classes_supervised=True,held_groups_pixels_components_excluded=True)
    output=dict(schema_version='crop-compatible-summary.v2',run=str(RUN),primary_checkpoint='fixed_step1000',fit_count=len(times),optimizer_updates=1000*len(times),adaptive_no_kl_included='paired_no_kl' in METHODS,ablation_run=str(ABLATION) if 'paired_no_kl' in METHODS else None,ablation_protocol_sha256=s.sha(ABLATION/'preregistration.json') if 'paired_no_kl' in METHODS else None,retained_training_seconds=sum(times),
        historical=historical,checkpoints=results,coverage=coverage,crop_shift=crop_shift(entries,cache),crop_shift_scope='Posthoc exploratory diagnostic at fixed1000: matched raw/SAM embeddings and rare SAM recall; excluded from selection guards',verification=verification,activated=False,locked_test_scored=False,
        limitations=p['limitation'],pair_exposure_note=p['pair_exposure'],protocol_sha256=s.sha(RUN/'preregistration.json'),analysis_source_sha256=s.sha(Path(__file__)))
    s.write(OUT,output)
    names={'ce':'CE de référence','balanced':'CE équilibrée par classe','preserve':'Équilibrage + conservation','paired':'Conservation + paires de crops','paired_no_kl':'Équilibrage + paires, sans KL'}
    tables={}
    for step,methods in results.items():
        tables[step]='\n'.join('| '+names[method]+' | '+' / '.join(map(str,e['masked_correct']))+' | '+' / '.join(map(str,e['raw_correct']))+f" | {100*e['page_mean_joint']:.2f} % | {100*e['rare_macro']:.2f} % | "+('Oui' if e['guard_vs_historical']['eligible'] else 'Non')+' |' for method,e in methods.items())
    paired=results['1000']['paired'];shift=output['crop_shift'];compare=paired['paired_vs_historical']
    rare_table='\n'.join('| '+('Historique figé' if k=='historical' else names[k])+' | '+' / '.join(str(v['rare_sam_correct']) for v in e['seeds'].values())+f" | {100*statistics.mean(v['rare_sam_macro_joint'] for v in e['seeds'].values()):.2f} % |" for k,e in shift.items())
    table_header='| Recette | SAM correct /254, graines 42 /43 /44 | Crop annoté correct /343 | Moyenne par planche | Macro rare annotée | Ancien garde-fou annoté |\n|---|---:|---:|---:|---:|---|\n'
    lines=[
        '# Reconnaissance compatible avec les crops SAM — 4 octobre 2026',
        '**Résultat principal.** Entraîner sur les vues utilisées à l’inférence améliore le nombre total de glyphes reconnus. Pourtant, la recette avec conservation et paires dégrade fortement les glyphes rares sur les crops SAM : 3/10 contre 8/10 pour l’historique. Les premiers garde-fous, mesurés sur les crops annotés, étaient insuffisants. Un correctif d’inférence sans nouvel entraînement récupère ces réponses rares ; son résultat et ses limites sont détaillés ci-dessous. Le modèle actif reste inchangé.',
        f'**Budget.** {len(times)} fits retenus, {1000*len(times):,} mises à jour, {sum(times)/60:.2f} minutes cumulées dans les boucles d’entraînement, hors préparation et évaluation. Comparaison initiale : 5 partitions × 3 graines × 4 bras, soit 60 fits. Puis 15 fits supplémentaires retirent la KL du bras avec paires. Cette ablation est enregistrée après inspection des 60 premiers résultats : elle reste adaptative et exploratoire. Le pas 1 000 est le critère principal fixé avant les fits ; le pas 300 est un diagnostic sans sélection.',
        '**Recette commune.** DINOv2 et SAM figés, mêmes caractéristiques et prototypes historiques fixes, même initialisation historique, AdamW à 10⁻⁴, clipping à 1, sans scheduler. Seule la projection de reconnaissance est entraînée. Les quatre bras reçoivent exactement les mêmes batches : 64 classes de crops annotés, jusqu’à deux objets par classe, et 16 classes de crops SAM masqués, jusqu’à quatre objets. Les 286 classes reçoivent une supervision directe ; toutes les images ne sont pas nécessairement tirées.',
        '**Pertes comparées.** CE de référence : moyenne par crop. CE équilibrée : moyenne au sein de chaque classe présente, puis moyenne des classes ; cela change les poids sans augmenter la fréquence de tirage des classes rares. Conservation : KL vers les réponses historiques correctes sur les objets parent train, détachées, température 0,2 et coefficient 4 sans facteur T². Paires : distance cosinus entre crop annoté et crop SAM du même objet, coefficient 1, sans dropout. Cette cohérence de deux vues est un terme positif ; elle ne constitue pas une perte contrastive avec des négatifs entre classes. Les passages supplémentaires sont comptés séparément.',
        '**Mesure.** Les 17 planches métier sont exclues à tour de rôle, avec fermeture de leurs composants et exclusion de leurs pixels. Les paires sont jointes par objet et groupe. Aucun dev/test ne sert aux gradients ou à la sélection du checkpoint. Les planches ont déjà été examinées lors des recherches précédentes, et la provenance d’entraînement historique reste inconnue : cette comparaison ne constitue pas un test indépendant à l’aveugle.',
        'Les métriques portent sur 343 glyphes connus, dont 254 appariés à SAM et 89 absents ou mal découpés. Le top-1 sans rejet sur ces propositions mesure la reconnaissance des objets connus, pas la précision ou le F1 sur toutes les propositions. Le groupe rare contient dix objets de quatre classes, ayant 1 à 4 crops aux pixels distincts éligibles à l’entraînement par classe dans la partition considérée. Tous ces dix objets sont appariés à SAM. La macro moyenne les scores des classes ; elle diffère du pourcentage global de bonnes réponses.',
        '**Principal, pas 1 000.** Historique : 160/254 sur SAM, 186/343 sur crops annotés, 49,75 % de moyenne par planche et 31,25 % de macro rare annotée. Les moyennes par planche comptent aussi les objets non appariés comme manqués.',
        table_header+tables['1000'],
        f'Le bras avec conservation et paires gagne en moyenne {statistics.mean(paired["masked_correct"])-160:.2f} bonnes reconnaissances SAM sur l’historique, soit {100*(statistics.mean(paired["masked_correct"])-160)/254:.2f} points sur les 254 crops appariés. Le gain moyen par planche est de {100*compare["mean_delta"]:.2f} points : {compare["wins"]} planches progressent, {compare["ties"]} restent à égalité et {compare["losses"]} reculent. L’intervalle descriptif par rééchantillonnage des planches, [{100*compare["bootstrap_95_interval"][0]:.2f} ; {100*compare["bootstrap_95_interval"][1]:.2f}] points, contient zéro. Les trois graines ne remplacent pas de nouvelles planches.',
        '**Limite critique : les rares en entrée SAM.** Diagnostic exploratoire calculé après les fits au checkpoint fixé, exclu des garde-fous initiaux. L’ancien garde-fou rare portait uniquement sur les crops annotés : 2/10 bonnes réponses historiques, macro 31,25 %, également préservées par le bras paired. Sur les crops réellement utilisés à l’inférence, le même historique reconnaît 8/10 objets rares et paired seulement 3/10. Passer le garde-fou annoté ne préserve donc pas les réponses rares en inférence.',
        '| Recette | Rares SAM corrects /10, graines 42 /43 /44 | Macro rare SAM |\n|---|---:|---:|\n'+rare_table,
        f'La distance cosinus moyenne entre vues diminue de {shift["historical"]["mean_cosine_distance"]:.3f} à {shift["paired"]["mean_cosine_distance"]:.3f}. C’est cohérent avec l’effet recherché par la perte de paires, mais cela ne garantit pas la reconnaissance des rares. L’ablation sans KL conserve un total SAM proche, tout en perdant une réponse rare annotée pour la graine 43. La conservation a donc un effet limité sur le total et utile sur ce petit groupe ; elle ne suffit pas à protéger les vues SAM.',
        '![Résultats globaux et glyphes rares sur SAM](crop-compatible-20261004.png)',
    ]
    readout_path=ROOT/'reports/rare-readout-results-20261004.json'
    if readout_path.exists():
        readout=s.read(readout_path);readout_run=Path(readout['run']);driver.verify(s.read(readout_run/'preregistration.json'))
        assert readout['optimizer_updates']==0 and not readout['activated'] and all(readout['verification'].values())
        gated=readout['families']['gated'];effect=gated['paired_vs_historical'];counts=[]
        for seed in driver.SEEDS:
            es=[e for e in readout['folds'] if e['seed']==seed]
            counts.append({k:sum(e['details']['mask'][k] for e in es) for k in ['fallback_rows','corrected_vs_paired','broken_vs_paired']})
        assert all(c==dict(fallback_rows=14,corrected_vs_paired=5,broken_vs_paired=1) for c in counts)
        assert readout['guard']['eligible'] and gated['masked_correct']==[180,178,180]
        lines.extend([
            '**Correctif d’inférence sans entraînement.** Après cette découverte, un protocole séparé est enregistré avant son évaluation. On calcule le nombre de crops train uniques de chaque classe après exclusion de la partition. Si la réponse top-1 historique désigne une classe avec 1 à 4 supports, on conserve cette réponse ; sinon, on utilise la nouvelle projection paired au pas 1 000. La règle utilise la classe prédite et les données train, jamais l’étiquette de l’objet évalué. Aucun seuil n’a été balayé, aucun poids réentraîné, aucune activation effectuée. Il s’agit d’une sélection entre deux projections à l’inférence avec les mêmes prototypes.',
            '| Variante | SAM correct /254, graines 42 /43 /44 | Rares SAM corrects /10 | Moyenne par planche |\n|---|---:|---:|---:|\n| Historique | 160 / 160 / 160 | 8 / 8 / 8 | 49,75 % |\n| Conservation + paires | 176 / 174 / 176 | 3 / 3 / 3 | 52,07 % |\n| Paires + recours historique pour les rares | 180 / 178 / 180 | 8 / 8 / 8 | 52,99 % |',
            'La règle appelle la réponse historique sur 14 des 254 crops SAM par graine : elle corrige cinq réponses de paired et en dégrade une, soit un gain net de quatre. Les réponses brutes sur crops annotés restent à 209 / 206 / 207 sur 343. Les quatre garde-fous passent, y compris celui ajouté sur la macro rare SAM. Les réponses rares historiques sont retrouvées, sans amélioration supplémentaire des deux rares encore incorrects.',
            f'Face à l’historique, le correctif apporte {statistics.mean(gated["masked_correct"])-160:.2f} bonnes réponses SAM en moyenne, soit {100*(statistics.mean(gated["masked_correct"])-160)/254:.2f} points conditionnels sur les crops appariés et {100*(statistics.mean(gated["masked_correct"])-160)/343:.2f} points en comptant les 343 objets connus. Le gain moyen par planche est de {100*effect["mean_delta"]:.2f} points : {effect["wins"]} progressent, {effect["ties"]} restent à égalité et {effect["losses"]} reculent. L’intervalle descriptif [{100*effect["bootstrap_95_interval"][0]:.2f} ; {100*effect["bootstrap_95_interval"][1]:.2f}] points contient encore zéro. Le correctif lui-même a été imaginé sur les erreurs de ces planches ; ses gains ne sont pas une validation indépendante.',
            '**Mon avis.** Le simple ajout d’annotations dans le long entraînement de base ne résolvait pas le changement de vue entre crop annoté et crop SAM. Apprendre ces deux vues améliore ici la reconnaissance globale. Mais la recette paired seule n’est pas un remplacement acceptable : elle oublie des réponses rares sur SAM. Le recours à l’historique est le candidat le plus intéressant de cette série, car il récupère ces réponses et améliore aussi le total avec une règle courte. Je conserverais le modèle actif et validerais cette règle, figée, sur de nouvelles planches exhaustivement annotées avant activation. Il faut y mesurer également les fausses propositions et le rejet, ainsi que les erreurs par classe rare. SAM reste figé dans cette série : les 89 objets manqués ou mal découpés restent un problème distinct et plafonnent le résultat.',
        ])
    lines.extend([
        '**Diagnostic au pas 300, sans sélection.** La référence reproduit les anciennes 170 / 171 / 172 réponses SAM, toutes les prédictions et métriques des quinze comparaisons ; les poids passent le contrôle numérique. Le pas 300 ne remplace pas le critère principal à 1 000.',
        table_header+tables['300'],
        '**Garde-fous initiaux.** Gain de moyenne par planche, absence de baisse de macro rare sur crop annoté, et baisse brute moyenne d’au plus deux points dans chaque partition. Les graines sont moyennées sans choisir la meilleure. Ces critères n’incluaient pas les rares SAM ; la colonne « ancien garde-fou annoté » ne suffit donc pas à conclure à une préservation en inférence. Le readout séparé ajoute explicitement la non-régression de macro rare SAM.',
        '**Vérification et traçabilité.** Hashes des sources, caches, prototypes, checkpoints et fichiers actifs vérifiés ; quinze reproductions à 300 pas ; mêmes batches et expositions CE entre bras ; exclusions des groupes, composants et pixels. Les couvertures, poids cumulés des classes rares et passages supplémentaires figurent dans le JSON. Deux lancements techniques ont chacun écarté un fit de 1 000 pas : une assertion exigeait à tort tous les crops, puis une comparaison opposait des clés entières en mémoire à des clés texte du JSON dans l’ablation. Sources et motifs sont archivés, corrections appliquées sans changement de loss ni d’hyperparamètre. Ces 2 000 mises à jour techniques sont exclues des 75 fits retenus.',
        '**Revue indépendante.** La règle, les exclusions et les agrégations ont été vérifiées en lecture seule. Une lacune de provenance a été relevée : le helper qui calcule les garde-fous du readout n’était pas inclus dans son préenregistrement de hashes. Les quatre critères ont donc été recalculés directement depuis les métriques sauvegardées, et leur code de vérification est archivé après coup dans le run. Cela confirme les valeurs publiées sans rendre ce contrôle rétrospectif aveugle ni autoriser une promotion.',
        '[Mesures et couverture des 75 fits](crop-compatible-results-20261004.json). [Mesures et décisions du correctif](rare-readout-results-20261004.json). Scripts : `scripts/research_crop_compatible.py`, `scripts/research_crop_compatible_no_kl.py`, `scripts/research_rare_readout.py`. Runs : `backend/training_runs/20261004-crop-compatible-v2`, `backend/training_runs/20261004-crop-compatible-no-kl-v2`, `backend/training_runs/20261004-rare-readout-v1`.',
    ])
    (ROOT/'reports/crop-compatible-research-20261004.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    s.write(RUN/'summary-verification.json',dict(summary_sha256=s.sha(OUT),**verification))
    print(json.dumps({step:{method:dict(sam=e['masked_correct'],raw=e['raw_correct'],page=e['page_mean_joint'],rare=e['rare_macro'],guard=e['guard_vs_historical']) for method,e in methods.items()} for step,methods in results.items()},indent=2))


if __name__=='__main__':main()

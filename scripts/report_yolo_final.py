"""Final accounting on confirmed partial annotations; no fitting or selection."""
import collections,json
from pathlib import Path
from scripts import report_yolo_synthetic as old
from scripts import research_synthetic_detector as s
import matplotlib.pyplot as plt

ROOT=s.ROOT;OUT=ROOT/'reports';RUNS=ROOT/'backend/training_runs'
PART=RUNS/'20261004-yolo-partial-label-v2'
FINAL='yolo-experiments-final-20261004'


def measure(result,geometry=False):
    c=result['counts'];n=c['known'];assert n==343 and c['correct']<=c['matched']<=n
    row=dict(known_records=n,matched_known=c['matched'],known_box_recall=c['matched']/n,
             retained_outputs=c['accepted'],ignored_outputs=c['ignored_predictions'],
             unmatched_to_export=c['unmatched_predictions'])
    if not geometry:
        row.update(correct_known=c['correct'],known_joint_recall=c['correct']/n,
                   conditional_naming=c['correct']/c['matched'] if c['matched'] else None)
    assert c['scored_predictions']==c['matched']+c['unmatched_predictions']
    return row


def pct(value):return f'{100*value:.1f} %'


def main():
    results={steps:s.read(path) for steps,path in old.READOUTS.items()}
    detectors={steps:s.read(path) for steps,path in old.DETECTORS.items()}
    partial=s.read(PART/'results.json');pd=s.read(PART/'detection-results.json')
    confirmation=OUT/'yolo-annotation-coverage-confirmation-20261004.json'
    audit=OUT/'yolo-annotation-coverage-audit-20261004.json'
    assert s.read(confirmation)['original_audit_sha256']==s.sha(audit)
    assert s.read(confirmation)['export_is_exhaustive'] is False
    verified={};fits=[]
    for steps,det in detectors.items():
        assert len(det['fits'])==10 and det['optimizer_updates']==10*steps
        fits.extend(det['fits'])
        for name,protocol in [('detector',det['protocol']),('recognition',results[steps]['protocol'])]:
            s.verify(protocol);verified[f'{steps}_{name}']=len(protocol['input_sha256'])
        assert results[steps]['feature_sha256']==s.sha(old.READOUTS[steps].parent/'features.pt')
    for name,protocol in [('partial_detector',pd['protocol']),('partial_recognition',partial['protocol'])]:
        s.verify(protocol);verified[name]=len(protocol['input_sha256'])
    assert len(pd['fits'])==5 and pd['optimizer_updates']==1000
    assert partial['feature_sha256']==s.sha(PART/'features.pt')
    fits.extend(pd['fits']);assert len(fits)==25
    for fit in fits:
        assert s.sha(fit['checkpoint'])==fit['checkpoint_sha256']
        assert len(fit['history']) in [200,1000]
    updates=sum(len(fit['history']) for fit in fits);assert updates==13000
    original=s.read(s.RUN/'dataset.json');pdata=s.read(PART/'dataset.json')
    assert [p['real_plan'] for p in original['splits']]==[p['real_plan'] for p in pdata['splits']]
    for f in pd['fits']:
        control=next(z for z in detectors[200]['fits'] if z['arm']=='real' and z['fold']==f['fold'])
        assert f['initial_state_sha256']==control['initial_state_sha256']
        assert f['selection']['threshold']==control['selection']['threshold']
    active=s.read(OUT/'yolo-synthetic-verification-20261004.json')['active_runtime_sha256']
    assert all(s.sha(ROOT/'backend/codex_model'/path)==value for path,value in active.items())
    base=results[200]['reference']
    geometry=[dict(label='SAM automatique',**measure(base['sam_geometry'],True))]
    naming=[dict(label='SAM',view='mask',policy=policy,**measure(base[key])) for policy,key in [('current_035','sam_gated_035'),('rare_branch','sam_branch')]]
    branch=s.read(s.BRANCH);classes={'sam':old.per_class([p for p in branch['outer_pages'] if p['seed']==42])}
    for steps,result in results.items():
        for arm in ['real','real_plus_synthetic']:
            label=f'YOLO {steps} '+('réel' if arm=='real' else 'réel + synthétique')
            geometry.append(dict(label=label,**measure(result['geometry'][arm],True)))
            pages=[p for f in result['fits'] if f['arm']==arm for p in f['pages']]
            assert len(pages)==17 and sum(len(p['gt']) for p in pages)==343
            for view in ['raw','mask']:
                classes[f'{steps}_{arm}_{view}']=old.per_class(pages,view)
                for policy in ['current_035','rare_branch']:
                    naming.append(dict(label=label,view=view,policy=policy,**measure(result['recognition'][arm][view][policy])))
    geometry.append(dict(label='YOLO 200 annotations partielles',**measure(partial['geometry'],True)))
    assert len(partial['pages'])==17 and sum(len(p['gt']) for p in partial['pages'])==343
    for view in ['raw','mask']:
        classes[f'partial_{view}']=old.per_class(partial['pages'],view)
        for policy in ['current_035','rare_branch']:
            naming.append(dict(label='YOLO 200 annotations partielles',view=view,policy=policy,**measure(partial['recognition'][view][policy])))
    assert [z['matched_known'] for z in geometry]==[254,174,165,134,150,219]
    assert partial['recognition']['raw']['rare_branch']['counts']['correct']==134
    for values in classes.values():assert sum(v['known'] for v in values.values())==343
    diagnostic=s.read(RUNS/'20261004-yolo-train-diagnostic-v1/results.json')
    train=[dict(steps=z['steps'],arm=z['arm'],train_known=z['train']['counts']['known'],train_matched=z['train']['counts']['matched'],outer_known=z['outer']['counts']['known'],outer_matched=z['outer']['counts']['matched']) for z in diagnostic['results']]
    source_paths=list(old.READOUTS.values())+list(old.DETECTORS.values())+[PART/'results.json',PART/'detection-results.json',confirmation,audit]
    proof=dict(total_detector_fits=25,detector_optimizer_updates=updates,frozen_input_hash_counts=verified,all_frozen_inputs_and_25_checkpoints_unchanged=True,partial_init_plans_and_cuts_match_control=True,feature_caches_unchanged=True,active_runtime_sha256=active,native_independent_review='annotation_fix_review: five fits, known matches, naming denominators and3938 detector hashes independently checked; no execution or exclusion issue found.',activated=False,locked_test_scored=False)
    s.write(OUT/'yolo-experiments-verification-final-20261004.json',proof)
    summary=dict(metric_policy='Confirmed partial labels. Known-record recall and conditional naming only. Unmatched outputs are not all false positives; precision and F1 are not exhaustive quality metrics.',known_records=343,pages=17,seed=42,total_detector_fits=25,detector_optimizer_updates=updates,geometry=geometry,naming=naming,per_class=classes,train_diagnostic=train,source_sha256={str(p):s.sha(p) for p in source_paths},verification=proof,activated=False,locked_test_scored=False)
    s.write(OUT/f'{FINAL}.json',summary)
    chosen=[naming[1]]+[z for z in naming if z['policy']=='rare_branch' and z['label'] in ['YOLO 200 réel','YOLO 200 annotations partielles']]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    labels=['SAM','200 réel','200 + synth.','1000 réel','1000 + synth.','200 partiel']
    for ax,vals,names,title in [(axes[0],[100*z['known_box_recall'] for z in geometry],labels,'Boîtes retrouvées sur les annotations connues'),(axes[1],[100*z['known_joint_recall'] for z in chosen],['SAM\nmasque','200 réel\nrectangle','200 réel\nmasque','200 partiel\nrectangle','200 partiel\nmasque'],'Boîte et nom corrects sur les annotations connues')]:
        colors=['#465469']+['#27859a']*(len(vals)-2)+['#bd751a']
        if len(vals)==5:colors=['#465469','#27859a','#77b2ba','#bd751a','#dfac62']
        bars=ax.bar(names,vals,color=colors);ax.set_ylim(0,100);ax.set_ylabel('Rappel connu (%)');ax.set_title(title,fontsize=12)
        for bar,value in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.1f}',ha='center')
        ax.tick_params(axis='x',labelsize=9)
    fig.suptitle('Annotations partielles · 343 enregistrements · 17 planches · seed 42 · IoU ≥ 0,5',fontsize=13)
    fig.text(.5,.015,'Le rappel réel et la précision globale ne sont pas mesurables avec cet export incomplet.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/f'{FINAL}.png',dpi=160);plt.close(fig)
    lines=['# Expériences glyphes : bilan corrigé après confirmation des annotations partielles','',
           '**25 entraînements de détecteur, 13 000 mises à jour.** Le correctif des annotations partielles récupère 45 boîtes connues supplémentaires à recette, batches et seuils identiques. SAM retrouve encore davantage de boîtes connues. Aucun modèle activé.','',
           '## Ce qui change le diagnostic','',
           'Tu as confirmé que la tête orange de S14 doit être annotée. Elle est absente de toutes les décisions dans l’archive source : ce cas ne provient pas d’une perte silencieuse lors de l’import. Le contrôle YOLO 200 la détectait déjà et son rectangle était correctement reconnu comme tlacatl, mais l’évaluation la comptait parmi les sorties sans correspondance. La précision et les F1 précédemment présentés ne mesurent donc pas la qualité exhaustive : certaines erreurs comptées sont des glyphes correctement trouvés mais non annotés.','',
           '![Audit visuel](yolo-annotation-coverage-20261004.png)','',
           'Une autre tête possède deux boîtes tlacatl presque identiques (IoU 0,816). C’est un candidat à révision métier, sans suppression automatique. Le dénominateur343 correspond à des enregistrements connus, pas strictement à 343 objets physiques distincts. Les 400 et quelques annotations sont des boîtes réparties sur environ 20 planches physiques, pas 400 images de planches indépendantes. Cette campagne évalue 343 enregistrements sur 17 planches, avec 7 régions explicitement ignorées.','',
           '## Essais terminés','',
           '| Comparaison | Fits | Mises à jour | But |','|---|---:|---:|---|',
           '| Réel contre réel + synthétique,200 updates | 10 |2000|Tester des positions synthétiques provenant des anciens crops|',
           '| Même comparaison,1000 updates | 10 |10000|Tester une recette prolongée, avec horizon cosine prolongé|',
           '| Réel avec zones non annotées ignorées,200 updates | 5 |1000|Éviter certaines pénalisations négatives injustifiées|','',
           'Tous utilisent YOLO11n préentraîné COCO, une classe glyphe, seed 42 et cinq partitions. Même exposition réelle entre les deux bras ; le synthétique ajoute du calcul. Les exemples synthétiques utilisent uniquement les anciens crops legacy du train admissible et des fonds réels pauvres en encre. Les crops historiques sans coordonnées supervisent les noms et peuvent créer des positions synthétiques ; ils ne révèlent pas leurs anciennes positions dans de vraies planches.','',
           'Le dernier essai conserve exactement les pixels, cibles positives, 200 batches, initialisation, optimiseur, horizon LR, BatchNorm et seuils du contrôle réel. Il ajoute des rectangles d’ignorance sur les seules pages propres au fit : propositions SAM sans recouvrement IoU≥0,5 avec une boîte connue ou explicitement ignorée. 351/444/510/459/403 rectangles supplémentaires dans les tuiles selon la partition. Aucun pseudo label positif. Les gradients de classification négative sont annulés dans ces régions ; les positifs, pertes box et DFL sont conservés.','',
           '**Limite du correctif :** SAM peut manquer des glyphes ou proposer du fond. Ses rectangles ne couvrent même pas le centre de la tête confirmée de S14. Ce mécanisme réduit certaines pénalisations, sans compléter les annotations.','',
           '## Boîtes retrouvées','',
           '| Détecteur | Boîtes connues retrouvées | Rappel connu | Sorties conservées | Sans correspondance dans l’export |','|---|---:|---:|---:|---:|']
    for z in geometry:lines.append(f'| {z["label"]} | {z["matched_known"]}/343 | {pct(z["known_box_recall"])} | {z["retained_outputs"]} | {z["unmatched_to_export"]} |')
    lines+=['','Les sorties ignorées sont retirées du matching. Les sorties sans correspondance peuvent être des glyphes absents de l’export, du fond ou des doublons ; leur nombre ne permet pas de calculer une précision réelle. Le correctif améliore les cinq partitions :51/68/17/16/22 →62/76/32/25/24 objets connus retrouvés. Gain agrégé :+45, soit+13,1 points. Le nombre de sorties conservées passe cependant de 597 à 950.','',
            f'![Résultats corrigés]({FINAL}.png)','','## Boîte et nom : séparer rappel et justesse conditionnelle','',
            'Rappel joint = bons noms associés à une bonne boîte /343 annotations connues. Justesse conditionnelle = bons noms /annotations connues associées à une boîte conservée après filtre. Les cohortes de boîtes associées changent entre variantes ; une baisse conditionnelle ne prouve pas que le classifieur lui-même se dégrade.','',
            '| Détecteur | Crop | Filtre | Boîte + nom corrects | Rappel joint connu | Justesse des noms sur boîtes associées | Sorties conservées |','|---|---|---|---:|---:|---:|---:|']
    for z in naming:
        lines.append(f'| {z["label"]} | {z["view"]} | {z["policy"]} | {z["correct_known"]}/343 | {pct(z["known_joint_recall"])} | {z["correct_known"]}/{z["matched_known"]} = {pct(z["conditional_naming"])} | {z["retained_outputs"]} |')
    lines+=['','Avec le filtre branche et les rectangles : 112→134 bons résultats joints (+22), mais justesse conditionnelle 74,2→71,3%. Avec les masques : 108→119 (+11), mais 78,3→68,8%. Le gain concerne le rappel joint connu, sans amélioration démontrée de la reconnaissance des noms. Le classifieur et les prototypes restent figés. Le candidat SAM atteint 156/343 et 156/194=80,4% sur ses correspondances conservées.','',
            '## Par classe','',
            'Les 99 classes présentes couvrent 343 annotations ; 187 des 286 classes n’ont aucun exemple ici. 46 classes n’ont qu’un exemple, 20 en ont deux, 11 trois, 6 quatre, 16 au moins cinq. Une classe à 1/1 ne constitue pas une performance fiable à 100 %. Le JSON joint contient les 99 classes pour toutes les variantes.','',
            '| Classe | Annotations connues | SAM branche | YOLO 200 rectangle | Partiel rectangle | Partiel masque |','|---|---:|---:|---:|---:|---:|']
    names=sorted(classes['sam'],key=lambda n:(-classes['sam'][n]['known'],n))
    assert len(names)==99
    for name in names[:25]:
        n=classes['sam'][name]['known'];values=[classes[k][name]['correct'] for k in ['sam','200_real_raw','partial_raw','partial_mask']]
        lines.append('| '+name+' | '+str(n)+' | '+' | '.join(f'{v}/{n}' for v in values)+' |')
    lines+=['','## Chemin d’inférence','',
            'Planche → tuiles 768/stride512 + page entière → YOLO → fusion NMS 0,5 → seuil détecteur propre à la partition → rectangle brut ou masque SAM obtenu avec cette boîte → crop avec marge 5 → DINOv2 figé → projection propre à la partition + prototypes historiques → garde des classes rares → filtre sémantique 0,35 ou branche. Aucune boîte de référence n’entre dans les prompts SAM ou les crops évalués. SAM n’est pas réentraîné.','',
            'Le test synthétique ne change que l’apprentissage des positions. Le dernier correctif ne change que les régions soumises à la perte négative. Les deux réutilisent le travail de reconnaissance déjà entraîné dans les expériences antérieures.','',
            '![Données d’entraînement](yolo-synthetic-data-legend-20261004.png)','',
            '## Entraînement prolongé et limites','',
            '![Pertes du pilote et du prolongement](yolo-synthetic-loss-20261004.png)','',
            'Les courbes montrent la perte sur les vrais batches pour les 20 premiers fits ; elles ne mesurent pas la qualité sur une nouvelle planche. Le prolongement à 1000 est adaptatif, décidé après lecture du pilote 200, puis figé dans un nouveau dossier. Il change aussi l’horizon cosine. Il ne démontre ni un bénéfice de durée seule, ni l’inefficacité générale de YOLO.','',
            '| Diagnostic partition 0 | Train : boîtes connues | Pages exclues : boîtes connues |','|---|---:|---:|']
    for z in train:lines.append(f'| {z["steps"]} {z["arm"]} | {z["train_matched"]}/{z["train_known"]} | {z["outer_matched"]}/{z["outer_known"]} |')
    lines+=['','Ce diagnostic postérieur documente un écart entre cohortes train et exclue de composition différente ; ce n’est pas une preuve causale de surapprentissage.','',
            'Les 17 planches ont déjà été examinées, une seule graine est utilisée, la provenance historique des projections/prototypes reste imparfaitement connue. Les partitions excluent les pages de l’adaptation actuelle, sans constituer un nouveau lot aveugle. Aucun score dev/test ni activation dans cette campagne. Les seuils initiaux ont été calibrés avec l’hypothèse désormais réfutée d’annotations exhaustives ; le dernier essai les réutilise pour contrôler la comparaison, sans les retoucher après lecture. Les seuils sémantiques historiques ont été calibrés sur les propositions SAM et peuvent se transférer imparfaitement vers YOLO.','',
            '## Avis et prochaine priorité','',
            '**Conserver SAM comme référence opérationnelle.** Le correctif est prometteur pour récupérer des annotations connues, mais il produit plus de sorties et reste derrière SAM sur le rappel connu des boîtes et des résultats joints. Sans annotation exhaustive, on ne sait pas chiffrer sa précision réelle ni arbitrer un remplacement.','',
            '**Priorité : terminer et vérifier les annotations des planches existantes**, en validant avec le métier les glyphes manquants et le quasi doublon, puis en gelant explicitement leur état de complétude. Les propositions des modèles peuvent servir de suggestions à vérifier ; elles ne deviennent pas automatiquement une vérité terrain. La boîte orange de ce rapport demeure un repère visuel, sans modification automatique du corpus.','',
            'Après cet audit, réévaluer les variantes sur des partitions par planche et publier rappel, précision, erreurs de noms et résultats par classe. Les définitions textuelles n’apporteraient pas les positions manquantes ; ce serait une expérience distincte de reconnaissance. Il n’y a pas encore de preuve ici qu’elles résoudraient le problème.','',
            '## Vérifications et traçabilité','',
            'Empreintes des sources, données et poids vérifiées pour les six protocoles ; 25 checkpoints et caches de features intacts ; batches et seuils du correctif égaux au contrôle ; trois fichiers du modèle actif inchangés. Une revue native indépendante confirme les résultats et les dénominateurs. Les champs précision/F1 restent dans les JSON bruts historiques comme comptabilité sur un export incomplet ; ils sont exclus des conclusions corrigées.','',
            '[Données corrigées](yolo-experiments-final-20261004.json) · [Vérification finale](yolo-experiments-verification-final-20261004.json) · [Audit source](yolo-annotation-coverage-audit-20261004.json) · [Confirmation métier](yolo-annotation-coverage-confirmation-20261004.json)','',
            'Sources de méthode : [YOLO11](https://docs.ultralytics.com/models/yolo11), [Cut, Paste and Learn](https://arxiv.org/abs/1708.01642). Ces sources motivent les expériences ; elles ne prédisent pas nos résultats.','']
    (OUT/f'{FINAL}.md').write_text('\n'.join(lines),encoding='utf-8')
    previous=OUT/'yolo-synthetic-experiments-20261004.md';archive=OUT/'yolo-synthetic-experiments-initial-20261004.md'
    if not archive.exists():archive.write_bytes(previous.read_bytes())
    warning='> **Rapport initial remplacé.** La confirmation métier a établi que les annotations sont partielles : précision et F1 ci-dessous ne mesurent pas la qualité exhaustive. Lire le [bilan corrigé](yolo-experiments-final-20261004.md), couvrant 25 fits et 13 000 updates.\n\n'
    previous.write_text(warning+archive.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(dict(fits=25,updates=updates,geometry=geometry,classes_top=[dict(name=n,**{k:v[n] for k,v in classes.items() if k in ['sam','200_real_raw','partial_raw','partial_mask']}) for n in names[:10]],verified=verified),ensure_ascii=False,indent=2))

if __name__=='__main__':main()


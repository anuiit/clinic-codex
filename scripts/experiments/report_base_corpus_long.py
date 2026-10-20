"""Write the local research report from verified completed experiment summaries."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(name):return json.loads((ROOT/'reports'/name).read_text())
w=read('base-corpus-long-results-20261004.json');c=read('base-init-control-results-20261004.json');v=read('base-inference-views-results-20261004.json')
assert c['primary_all_folds'] is not None
cv=c['primary_all_folds'];diag=w['training_crop_diagnostic']['candidates']['combined']
rows=[]
for name,e in [('Historique figé',w['historical']),('Poids historiques, corpus ancien',w['primary']['parent']),('Poids historiques, corpus enrichi',w['primary']['combined']),('Tête neuve, corpus ancien',cv['parent']),('Tête neuve, corpus enrichi',cv['combined'])]:
 rows.append(f"| {name} | {e['raw']['correct']}/343 | {e['masked_correct']}/254 | {100*e['page_mean_joint']:.2f} % | {100*e['natural_support_bins']['rare1_4']['macro']:.2f} % |")
viewrows=[]
for name,key in [('Historique figé','historical'),('Poids historiques, ancien','warm_parent'),('Poids historiques, enrichi','warm_combined'),('Tête neuve, ancien','fresh_parent'),('Tête neuve, enrichi','fresh_combined')]:
 vs=v['families'][key];viewrows.append('| '+name+' | '+' | '.join(str(vs[k]['correct'])+'/254' for k in ['mask','rectangle','dilated3','half_background'])+' |')
seedrows=[]
for arm,name in [('parent','Ancien'),('combined','Enrichi')]:
 for r in w['full_corpus_secondary'][arm]:
  e=r['fixed_epoch100'];seedrows.append(f"| {name} | {r['seed']} | {e['application']['train']['top1_correct']}/254 | {e['raw']['locked_test']['top1_correct']}/146 | {e['application']['locked_test']['top1_correct']}/51 | {r['best_dev_epoch']} |")
paired=v['paired_warm_rectangle_vs_historical_mask'];lo,hi=paired['descriptive_bootstrap95']
text=f'''# Ajout des annotations à la recette de base — résultats du 4 octobre 2026

L’ajout direct des annotations est utile, mais son résultat dépend fortement de la recette et du crop présenté à l’inférence. Le réentraînement long à partir des poids historiques apprend presque toutes les entrées annotées : **{diag['annotation_raw_correct']}/343** sur les planches utilisées pour l’entraînement. Sur les planches exclues, il reconnaît **139/254** crops SAM masqués, contre **104/254** pour le témoin long et **160/254** pour l’historique figé. En lui présentant les rectangles SAM bruts, le même modèle enrichi obtient **167/254**.

Ces résultats justifient de conserver les annotations et de travailler la cohérence entre entraînement et inférence. Ils ne justifient pas de remplacer le modèle actif. L’écart de sept glyphes face à l’historique combine le réentraînement enrichi et le changement de format ; il reste variable par planche. Le classement des classes rares sur boîtes métier régresse. Aucun modèle, seuil ou chemin d’inférence actif n’a été modifié.

**Ce qui a réellement été entraîné.** La tête de projection du modèle de base, sur les caractéristiques DINOv2 S/14 précalculées. DINO et MobileSAM sont figés. La configuration indique `freeze_blocks: 9`, mais le chemin `train.py` utilisé ne charge pas le backbone : ce champ ne provoque aucun apprentissage de DINO ici. Les boîtes et leur placement ne sont pas des cibles de cette loss.

La recette reproduite fait 100 époques × 500 épisodes, soit **50 000 mises à jour** par entraînement, avec 20 classes, 3 supports et 5 requêtes, AdamW à 0,001, cinq époques de montée du taux puis décroissance cosinus, température 0,1. Le corpus ancien et l’enrichi ont la même initialisation dans chaque comparaison. Le contrôle supplémentaire réinitialise seulement la tête, en conservant les caractéristiques DINO préentraînées.

**Corpus.** Le parent contient 9 266 images, dont 9 128 en train. Le lot enrichi contient 9 716 images, dont 9 471 en train. Il ajoute 450 annotations actives : **343 train, 43 dev, 64 test**, couvrant 99 classes dans les 17 planches train. Le compteur exact de 433 annoncé initialement n’a pas été retrouvé ; l’expérience porte sur le lot exploitable identifié dans l’archive téléchargée.

La recette exige huit exemples distincts pour une classe épisodique. Elle entraîne directement 250 classes sur 286 dans le parent, 252 dans l’enrichi. **337 des 343 ajouts participent à la loss ; six restent utilisés uniquement pour les prototypes** : trois `tenchilnahuayo`, un `yacametztli`, deux `chimalli`. Toutes les classes disposent néanmoins d’un prototype, sans fallback historique.

**Mesure principale : planches exclues de l’adaptation.** Les 17 planches métier sont réparties en cinq partitions, graine 42. Pour chaque partition, ses images sont retirées des épisodes et de la construction des prototypes. Les affectations dev/test sont conservées. Le score principal vient exclusivement du checkpoint final à l’époque 100 ; le meilleur-dev est un diagnostic secondaire.

| Départ et corpus | Classe sur boîte métier | Classe sur crop SAM masqué | Boîte + classe, moyenne par planche | Macro rare sur boîtes métier |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Les 343 glyphes connus produisent 254 propositions SAM appariées ; 89 absents ou mal découpés restent des échecs. Le score par planche donne le même poids à chaque planche ; un total de glyphes donne plus de poids aux grandes planches. Il s’agit du top-1 sans politique de rejet, d’un rappel sur objets connus, pas d’une précision de détection ou d’un F1 sur toutes les propositions.

Le groupe rare ne contient que dix objets de quatre classes. Sa macro pondère les classes également et ne doit pas être interprétée comme une estimation précise sur tous les glyphes rares. Les deux départs enrichis échouent au garde-fou de non-régression rare face à l’historique. Le départ neuf passe les garde-fous face à son propre témoin ancien, mais échoue face à l’enrichi sur poids historiques à cause de baisses de plus de deux points sur les boîtes métier de certaines partitions. Les deux enrichis conservent un gain par rapport à leur témoin long, tout en restant moins bons que l’historique avec l’entrée masquée actuelle.

![Courbe des entraînements à partir des poids historiques](base-corpus-long-20261004.png)

**Pourquoi une planche déjà corrigée peut encore échouer.** Sur les planches vues par l’enrichi, à la graine 42 et au checkpoint final, {diag['annotation_raw_correct']}/343 crops annotés reçoivent la bonne classe. Parmi les 254 glyphes retrouvés par SAM, {diag['matched_raw_correct']}/254 sont corrects sur le crop annoté et seulement {diag['matched_mask_correct']}/254 sur le crop SAM. **{diag['raw_correct_mask_wrong']} reconnaissances correctes se perdent lors du changement de crop**. Les 89 glyphes non appariés constituent un autre obstacle, indépendant de la tête de reconnaissance.

Cette comparaison change ensemble la marge, le masque et parfois le découpage. Elle prouve la sensibilité au crop ; elle n’isole pas à elle seule une cause unique parmi ces changements. C’est une mesure sur données vues pendant l’entraînement, destinée à expliquer le symptôme, pas à estimer la généralisation.

![Exemples de prédictions différentes sur les deux crops](base-crop-examples-20261004.png)

**Même boîte SAM, format de crop différent.** Les quatre variantes viennent du même cache RGB/masque audité, sans nouvelle extraction, nouveau DINO, réentraînement ou changement de propositions. Ces comparaisons d’inférence sont exploratoires et utilisent les checkpoints fixes à l’époque 100.

| Modèle et corpus | Masque blanc | Rectangle brut | Masque dilaté de 3 px | Fond blanchi à moitié |
|---|---:|---:|---:|---:|
{chr(10).join(viewrows)}

Pour l’enrichi sur poids historiques, le rectangle atteint **167/254**, soit 167/343 avec les ratés de découpage, et **52,00 %** en moyenne par planche. La comparaison à l’historique avec masque change à la fois le modèle entraîné et le format : elle n’isole pas l’effet du rectangle. Dans cette comparaison, il gagne sur {paired['wins']} planches, perd sur {paired['losses']} et reste égal sur {paired['ties']}. L’écart moyen vaut **{100*paired['mean_page_delta']:.2f} points**, avec un intervalle descriptif par rééchantillonnage de planches de **[{100*lo:.2f} ; {100*hi:.2f}] points**. Ce rééchantillonnage n’est pas une validation indépendante et l’intervalle inclut zéro.

L’historique préfère ici le masque (160 contre 125 rectangles) ; le nouveau modèle appris sur crops bruts préfère les rectangles (167 contre 139 masques). Le résultat du modèle historique figé ne permettait donc pas, à lui seul, de juger le format à utiliser après réentraînement.

**Trois graines sur le corpus complet : diagnostics secondaires.** Les planches train sont alors vues pendant l’entraînement du corpus enrichi. La partition nommée `locked_test` a déjà été consultée lors des recherches précédentes ; elle n’est pas un nouveau test aveugle et n’a servi ni aux gradients ni à choisir le résultat principal. Le meilleur-dev est affiché à titre de traçabilité ; les nombres ci-dessous restent ceux de l’époque 100.

| Corpus | Graine | SAM masqué sur planches train | Boîtes métier, ancien test | SAM masqué, ancien test | Époque meilleur-dev |
|---|---:|---:|---:|---:|---:|
{chr(10).join(seedrows)}

**Contrôle d’initialisation.** La tête neuve a été examinée sur la première partition, puis étendue aux cinq parce que l’écart du témoin dépassait le seuil de 10 points enregistré avant tout fit neuf. Le seuil a été défini après inspection du premier résultat historique : le contrôle reste exploratoire. Les hashes des plans d’épisodes et les expositions de chaque ligne sont identiques entre départ historique et départ neuf pour chaque corpus. Le DINO préentraîné reste le même. Sur les cinq partitions, repartir à neuf donne 213 contre 220 bonnes classes sur boîtes métier, 124 contre 139 sur crops SAM masqués, et 155 contre 167 sur rectangles. Sa moyenne par planche masquée est légèrement meilleure (40,45 contre 39,05 %), mais son total est inférieur et ses résultats varient : il n’apporte pas de gain supplémentaire convaincant.

**Avis et suite proposée.** Je garderais le modèle et les annotations comme point de départ. Je travaillerais d’abord une recette qui voit les crops utilisés à l’inférence, avec conservation des acquis historiques et supervision des petites classes. La recette complète actuelle laisse 34 classes sans gradient épisodique direct. La régression observée des petites classes motive une supervision explicite ; ces mesures ne permettent pas d’attribuer à cette seule exclusion toute la baisse. Pour le placement des boîtes, il faut poursuivre une loss de localisation sur les planches complètes ; ajouter des crops à cette tête ne peut pas lui enseigner ce placement.

L’option rectangle constitue une piste concrète, mais son avantage reste modeste et variable sur ce corpus. Le précédent pilote CE avec crops bruts et masqués reconnaissait 170–172/254 glyphes selon la graine ; le nouveau résultat de 167 n’apporte pas encore un remplacement plus convaincant. Aucun changement d’architecture ni dégel de DINO n’est nécessaire pour tester ces deux corrections de recette.

**Vérifications et coût.** 11 entraînements complets sur départ historique et {c['distinct_fit_count']} contrôles complets à tête neuve, soit **{w['distinct_fit_count']+c['distinct_fit_count']} entraînements distincts et {format(w['total_optimizer_updates']+c['optimizer_updates'], ',').replace(',', ' ')} mises à jour** dans les runs retenus. Les comparaisons des cinq témoins partagent leurs poids lorsqu’elles ont exactement les mêmes données et la même graine, puis reconstruisent chaque évaluation séparément. Le premier lancement technique v1 a été interrompu pour supprimer ces calculs identiques ; sa source et son motif sont archivés, ses résultats ne sont pas utilisés.

Le contrôle CPU/CUDA reproduit la collation, la loss, les logits, les gradients et une mise à jour AdamW de la recette originale ; erreur maximale de gradient GPU 2,24×10⁻⁸. Les hashes des entrées, checkpoints, pixels des exemples et fichiers actifs passent les vérifications. Les expositions confirment 8 millions de tirages d’images par fit et l’absence des planches exclues. Les revues natives indépendantes n’ont trouvé aucun blocage méthodologique dans ces contrôles.

La provenance d’entraînement du modèle historique reste inconnue et le corpus a été examiné à plusieurs reprises. Ces expériences mesurent une adaptation contrôlée sur le lot local retrouvé, sans établir une généralisation aveugle ou une amélioration sur l’instance métier en ligne.

Artefacts : [mesures principales](base-corpus-long-results-20261004.json), [contrôle de départ](base-init-control-results-20261004.json), [variantes de crops](base-inference-views-results-20261004.json). Les manifests, poids et résultats détaillés sont dans `/home/sina/omxpro/clinic-codex/backend/training_runs/20261004-base-corpus-long-v2` et `20261004-base-init-control-v1`.
'''
(ROOT/'reports/base-corpus-long-research-20261004.md').write_text(text,encoding='utf-8')
print(ROOT/'reports/base-corpus-long-research-20261004.md')

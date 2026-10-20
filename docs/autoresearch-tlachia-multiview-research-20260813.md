# P4a — recherche Tlachia multi-vue

Ce harness teste si les glyphes, zones et planches Tlachia peuvent améliorer la tête de projection DINOv2-S/14 sans toucher au test métier ni au locked-test.

## Contrat gelé

- Entrées structurelles : plan `pilot-v1` validé, 219 JPG, 24 groupes de planches et une évidence descendante couvrant 286 classes runtime.
- Entrées éléments : cache scellé contenant uniquement 3 763 lignes train et 465 lignes dev.
- Backbone : DINOv2-S/14 local, source et poids vérifiés par SHA-256.
- Split structurel : six groupes entiers en dev, choisis par couverture gloutonne des classes puis hash stable ; les dix-huit autres groupes restent en train.
- Cache : `float16`, cinq vues par image (`base`, deux légères, deux fortes), avec identités des lignes, rôles, groupes, labels descendants, split et provenance.
- Recherche : 16 specs gelées avant scoring — contrôles historique/B1 et 14 challengers VICReg, dont trois hybrides avec un refresh épisodique supervisé limité au train éléments.
- Initialisation : chaque challenger repart indépendamment de la projection historique. Seul le contrôle B1 charge B1.

## Évaluation et sélection

Le primaire est la moyenne par planche de l’AP échantillon sur les glyphes du dev structurel. Le harness mesure également rappel@5, AP échantillon, métriques par groupe, ainsi que micro/macro/logloss sur le dev éléments.

Les 16 specs passent d’abord au seed 42. Ce seed sert exclusivement à sélectionner les trois meilleurs challengers respectant `macro éléments >= contrôle - 0,005`. Il est exclu de toutes les statistiques du gate final. Les challengers sélectionnés sont ensuite confirmés aux seeds frais 73, 101 et 202 avec le contrôle historique.

La décision agrège uniquement les trois confirmations fraîches : moyenne des deltas primaires, moyenne des macros éléments et moyenne par classe. L’intervalle apparié est un bootstrap hiérarchique déterministe qui rééchantillonne d’abord les seeds, puis les groupes de planches à l’intérieur de chaque seed.

Un candidat ne passe que si, sur cette agrégation fraîche :

- le gain primaire atteint 0,01 ;
- la borne basse à 95 % du bootstrap apparié par planche est positive ;
- la macro éléments reste dans la garde de 0,005 ;
- aucune classe éléments observée ne chute de plus de 0,20 ;
- au moins deux des trois seeds frais ont un gain primaire positif.

Tous les artefacts restent research-only et `promotion_eligible=false`.

## Commandes

```bash
backend/.venv/bin/python scripts/autoresearch_tlachia_multiview.py prepare --device cuda --batch-size 16
backend/.venv/bin/python scripts/autoresearch_tlachia_multiview.py run-all --device cuda
backend/.venv/bin/python scripts/autoresearch_tlachia_multiview.py verify
```

`prepare` réalise l’extraction GPU seulement si le cache immuable n’existe pas. Le cache et sa provenance sont préparés dans un répertoire de staging puis publiés ensemble par renommage atomique ; ils ne deviennent jamais visibles séparément. `run-all` est volontairement séparé : la préparation et la revue peuvent donc être terminées avant toute campagne.

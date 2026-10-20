# Résultats modèles sur le test final métier — 2026-08-12

## Décision

Le lot de 10 nouvelles pages directement placé dans `annotations/` est désormais la
référence métier pour comparer les modèles. Il est figé séparément des données
d'entraînement et n'est utilisé ni pour l'apprentissage, ni pour le choix des
hyperparamètres.

Sur ce lot, le candidat **S/14 B1 obtient des scores descriptifs au niveau du modèle
historique**. Il est légèrement meilleur en top-1 et en macro-classe, mais l'écart
n'est pas statistiquement démontré avec seulement 10 pages. Il ne faut donc revendiquer
ni amélioration, ni équivalence statistiquement établie. C'est un signal favorable de
non-régression, à confirmer à mesure que le test métier grandit.

## Test final figé

- Identifiant : `business-final-ea447440c1ae1d12e7b4`
- Empreinte de contenu : `ea447440c1ae1d12e7b47d2d8774911593b6328de7181a011823fa564f8cf652`
- 303 annotations évaluables, 87 classes de la taxonomie runtime, 10 pages
- 12 annotations exclues : 4 rejetées par le métier et 8 hors taxonomie runtime
- Aucun doublon exact de page ou de crop dans le lot
- Aucun crop pixel-identique avec les 9 266 exemples du snapshot d'entraînement
- Classe la plus représentée : `tlacatl`, 79/303 (26,07 %)
- Page la plus représentée : 45/303 (14,85 %)

Deux crops absents du disque ont été reconstruits depuis la page source et leur bbox;
les 313 crops déjà présents ont tous été vérifiés pixel à pixel contre la page source.

## Résultats

Le protocole principal `artifact` mesure le comportement exact du package modèle :
projection et prototypes livrés avec chaque version.

| Modèle | Top-1 micro | Top-3 micro | Top-1 macro | Top-3 macro |
|---|---:|---:|---:|---:|
| Historique | 47,85 % (145/303) | 64,69 % (196/303) | 49,12 % | 59,95 % |
| S/14 B0, ancre historique | 47,19 % (143/303) | 63,70 % (193/303) | 50,74 % | 60,16 % |
| **S/14 B1, warm-start** | **48,18 % (146/303)** | 64,36 % (195/303) | **51,32 %** | **60,32 %** |

Comparaison B1 contre l'historique :

- top-1 micro : +0,33 point, 7 gains et 6 pertes individuels;
- IC 95 % bootstrap pairé **par page** : −1,47 à +1,97 points;
- top-1 macro : +2,20 points;
- top-3 micro : −0,33 point;
- top-3 macro : +0,37 point.

Le protocole secondaire `train_recomputed`, qui recalcule les prototypes des modèles
sur le même train figé, donne 48,18 % pour B1 contre 47,19 % pour l'historique
(+0,99 point; IC 95 % groupé par page −0,34 à +2,33). B0 et l'historique sont strictement identiques
dans ce protocole, ce qui confirme l'alignement des features et de la projection.

## Portée de la conclusion

Le test est la vérité métier disponible et doit remplacer l'ancien holdout technique
de 82 exemples pour les résultats produit. Il reste toutefois partiel : 87 classes sur
286 et seulement 10 pages. Il montre que B1 est au niveau de l'historique sur ces pages,
sans démontrer une équivalence statistique ni une supériorité globale sur la taxonomie.

L'indépendance démontrée est limitée aux identités RGB exactes de pages et de crops.
Le snapshot historique ne porte pas de clé de lignée source comparable; une proximité
visuelle ou une même source acquise autrement ne peuvent donc pas être exclues par ce
contrôle. Les intervalles par crop sont conservés comme diagnostic exploratoire; seul
le bootstrap groupé par page est utilisé pour l'incertitude principale.

Le package historique reste intact et la promotion reste manuelle. Avant promotion,
la prochaine étape est une revue des erreurs par classe et par page sur ce rapport,
puis un smoke E2E applicatif de B1. Les futures pages métier devront être ajoutées à
une nouvelle version immutable du test; elles ne doivent jamais être absorbées dans
le train.

## Artefacts et reproductibilité

- Manifeste test local : `backend/training_corpus/evaluation/business-final-ea447440c1ae1d12e7b4/test_manifest.json`
- Rapport machine : `reports/business-final-model-evaluation-20260812.json`
- SHA sémantique du rapport : `36083003ba7a6eb002513debd1d9eb6101bae2cb215d9290b5d9b3a3a99cb928`
- Référence : `20260810T143106Z-7e40c6d-original`
- B0 : `20260811-s14-snapshot-anchor-b0`
- B1 : `20260811-s14-snapshot-warmstart-b1`
- Backbone commun : DINOv2 ViT-S/14 épinglé localement

Le gel, le cache de features et l'évaluation contrôlent les empreintes des pages,
crops, ordres de classes, snapshots d'entraînement, poids, prototypes, configurations
et checkpoints. Toute divergence fait échouer l'évaluation.

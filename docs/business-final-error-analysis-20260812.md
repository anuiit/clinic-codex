# Analyse d'erreurs du test métier final — 2026-08-12

Cette analyse est descriptive. Le jeu métier final est figé et n'est utilisé ni pour choisir les variantes, ni pour régler des hyperparamètres.

## Résultats observés

Le jeu contient 303 crops issus de 10 pages et couvre 87 des 286 classes. Les top-1 des artefacts packagés sont : historique 145/303 (47,85 %), B0 143/303 (47,19 %) et B1 146/303 (48,18 %). B1 ne gagne donc qu'un exemple net sur l'historique, sans démonstration statistique de non-régression.

Les erreurs sont très concentrées par source. Trois pages haute résolution représentent 87 des 158 erreurs du modèle historique : `ee53bd19` (35/45), `6e099900` (32/43) et `c4062dfe` (20/30). Leurs résolutions sont proches de 1 500 × 2 500, contre environ 700–750 × 1 100 pour plusieurs pages plus faciles. La similarité médiane avec la vraie classe y est nettement plus faible (0,320 à 0,461, contre 0,798 sur la page `12d6dd17`). Le shift de source, d'échelle ou de rendu est donc l'hypothèse dominante.

La confusion principale est `tlacatl → ixtlilli` : 20 erreurs pour l'historique, 21 pour B0 et 20 pour B1. `tlacatl` occupe 79/303 exemples et n'est correctement classée en top-1 que 31 fois par l'historique et 26 fois par B1. Plusieurs classes restent à 0 en top-1 pour les trois modèles, notamment `cuahuitl_1` (0/4), `macuilli` (0/3), `coyotl` (0/3) et `xacalli` (0/4).

Le nombre d'exemples d'entraînement n'explique pas seul les erreurs : la corrélation de Pearson entre support d'entraînement et taux d'erreur métier est d'environ 0,053. Le shift visuel/source est plus compatible avec les observations que la seule rareté des classes.

## Lecture de B1

B1 modifie très peu la frontière de décision. Face à l'historique packagé, il compte 7 corrections et 6 régressions. Dans le protocole équitable où les prototypes sont recalculés depuis le même train, B0 et l'historique sont identiques à 143/303, tandis que B1 atteint 146/303 (4 corrections, 1 régression).

L'historique utilise un mapping legacy bijectif par noms de classes, alors que B0/B1 utilisent le mapping canonique 0–285. Les comparaisons `artifact` mesurent la fidélité runtime ; les comparaisons `train-recomputed` isolent mieux l'effet de la projection.

## Conclusion actionnable

Les prochains gains crédibles dépendent surtout d'un instrument de développement couvrant davantage de pages, de résolutions et de classes, puis d'expériences groupées par source. Le test métier final ne doit pas être relu entre les itérations. La campagne autoresearch S/14 v2 a donc été menée exclusivement sur un cache physique train+dev scellé ; aucun challenger n'a dépassé B1 selon la règle pré-enregistrée.

Sources locales : `reports/business-final-model-evaluation-20260812.json` et `reports/autoresearch-s14-20260812-v2/results.json`.

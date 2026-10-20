# P5b — Politique de split fail-closed

P5b durcit la construction des futurs snapshots sans modifier le snapshot
`snapshot-471c74f801b749a9ad6e` ni le runtime historique.

## Contrat

- La politique devient `locked-source-group.v3`.
- Le plancher train est explicite et manifesté (`2` lignes par classe par
  défaut). Il est vérifié séparément de l'éligibilité au holdout.
- Les cibles `dev` et `locked_test` sont calculées avant matérialisation. Une
  cible non atteinte provoque un échec avec les déficits exacts.
- `--allow-underfilled-holdouts` est une dérogation nommée, réservée aux
  snapshots de recherche. La dérogation et les déficits restent liés au
  manifeste; elle ne rend pas le snapshot éligible à une promotion.
- Chaque classe expose son nombre de composantes, ses groupes et la raison
  d'une éventuelle inéligibilité structurelle au holdout. Le manifeste sépare
  cette borne théorique de la mobilité opérationnelle : une composante marquée
  `assignment_origin=parent` n'est jamais listée comme déplaçable.
- Les assignations d'un parent restent immuables; les connexions de doublons
  entre splits restent bloquantes.

Le dry-run strict du corpus actuel échoue volontairement avec `dev deficit=870`
et `locked_test deficit=844`. Le dry-run avec dérogation reproduit les comptes
9 128 / 56 / 82, mais génère un nouveau plan de politique v3 et ne matérialise
rien.

## Commandes

```bash
# Gate normal : doit échouer tant que les sources indépendantes manquent.
backend/.venv/bin/python scripts/build_training_snapshot.py \
  --dry-run --exclude-conflicts --json

# Diagnostic/recherche uniquement; aucune matérialisation avec --dry-run.
backend/.venv/bin/python scripts/build_training_snapshot.py \
  --dry-run --exclude-conflicts --allow-underfilled-holdouts --json
```

Une future extension cumulative doit fournir le snapshot courant via
`--parent-manifest`, ajouter des sources indépendantes, puis repasser le gate
strict sans dérogation avant matérialisation.

## Conséquence pour la collecte

L'audit P5a prouve qu'aucune composante train existante n'est déplaçable. Le
minimum informationnel pour le seul gate locked est donc 518 nouvelles lignes
sur 93 classes absentes. Il ne garantit pas le résultat du splitter. Avec les
deux splits à 600 lignes, la borne idéale monte à 1 062; restaurer 10 % / 10 %
demande au moins 2 144 lignes si toutes sont utilisables en heldout.

La collecte opérationnelle doit pré-enregistrer les groupes sources, simuler
le splitter sur les identités et labels seulement, puis figer le snapshot sans
examiner les performances du locked test. Le plan initial vise environ 1 000 à
1 100 lignes candidates indépendantes couvrant au moins 100 classes, avec une
marge supplémentaire pour les doublons, conflits et groupes atomiques. Cette
fourchette est une capacité de collecte, pas une garantie de promotion.

Voir [l'audit P5a](p5a-split-feasibility-20260811.md) pour les bornes et les
classes prioritaires.

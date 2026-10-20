# P5a — Audit de faisabilité du split

Snapshot : `snapshot-471c74f801b749a9ad6e` (`b7c8fd8eb197e976a5e32381b2c7455e5dcea1bb5ef1ff55e37d7473092a13da`).

## Verdict

**impossible_with_current_corpus_under_constraints**

Le locked test actuel contient **82 lignes / 7 classes / 4 groupes**, contre un gate de 600 lignes / 100 classes.
La borne supérieure optimiste du corpus est de **15 classes / 696 lignes**; elle n'est pas présentée comme simultanément réalisable et suppose une répartition depuis zéro.
Avec les assignations existantes immuables, aucune composante train n'est déplaçable : il faut au minimum **518 nouvelles lignes couvrant 93 classes supplémentaires** dans des sources indépendantes.

## Split actuel

| Split | Lignes | Cible | Classes | Groupes | Plus grosse classe | Plus gros groupe |
|---|---:|---:|---:|---:|---:|---:|
| dev | 56 | 926 | 8 | 8 | 57.1% | 57.1% |
| locked_test | 82 | 926 | 7 | 4 | 87.8% | 87.8% |

Épuisement du splitter avant les fractions demandées : **true** (0 composante train encore déplaçable individuellement).

## Bornes de faisabilité

- Composantes individuellement cédables : 26
- Classes avec au moins deux composantes : 15
- Classes avec au moins trois composantes : 0
- Construction gloutonne réalisable : 459 lignes / 15 classes

## Collecte minimale à planifier (contrat append-only)

Borne inférieure : **518 nouvelles lignes indépendantes**, avec au moins 93 nouvelles incidences classe-composante. Cette borne est nécessaire, pas suffisante : dédoublonnage, conflits et atomicité des pages peuvent augmenter la collecte réelle.
Si dev et locked_test doivent tous deux atteindre 600 lignes, la borne idéale monte à 1062 lignes; restaurer strictement les fractions 10 % / 10 % demanderait au moins 2144 lignes si chaque nouvelle ligne était utilisable en heldout. Le hash réel ne garantit aucune de ces bornes.

Classes absentes du locked test, triées par support train croissant :

| Classe | Lignes | Composantes | Groupes | Besoin pour 1 holdout |
|---|---:|---:|---:|---:|
| tlacotl | 34 | 2 | 2 | 0 |
| tlamamallahuiztli | 2 | 1 | 1 | 1 |
| ameyalli | 5 | 2 | 2 | 0 |
| teocomitl | 4 | 1 | 1 | 1 |
| xicolli | 4 | 1 | 1 | 1 |
| tenchilnahuayo | 4 | 1 | 1 | 1 |
| cáliz | 4 | 1 | 1 | 1 |
| tlalpilli_2 | 4 | 1 | 1 | 1 |
| amacalli | 4 | 1 | 1 | 1 |
| yacametztli | 4 | 1 | 1 | 1 |
| texcalli | 10 | 2 | 2 | 0 |
| cozcapetlatl | 4 | 1 | 1 | 1 |
| nextli | 4 | 1 | 1 | 1 |
| nacayotl | 4 | 1 | 1 | 1 |
| tzontli_2 | 4 | 1 | 1 | 1 |
| ocotl | 4 | 1 | 1 | 1 |
| matlalin | 4 | 1 | 1 | 1 |
| quimilli | 4 | 1 | 1 | 1 |
| cuahuacalli | 4 | 1 | 1 | 1 |
| cemixtlapalli | 4 | 1 | 1 | 1 |
| tepetla | 6 | 1 | 1 | 1 |
| tlapexohuiloni | 6 | 1 | 1 | 1 |
| pochotl | 6 | 1 | 1 | 1 |
| camotli | 6 | 1 | 1 | 1 |
| mapilli | 6 | 1 | 1 | 1 |
| xoxoqui | 6 | 1 | 1 | 1 |
| tenextli | 6 | 1 | 1 | 1 |
| ilhuicatl | 6 | 1 | 1 | 1 |
| quetzalmiyahuayotl | 6 | 1 | 1 | 1 |
| acalli | 6 | 1 | 1 | 1 |

## Décision P5b

Ne pas réassigner le snapshot existant ni entraîner pour promotion. Le prochain lot doit rendre le splitter fail-closed lorsque les cibles sont irréalisables, déclarer les classes mono-composante `holdout_ineligible`, puis définir un plan de collecte par source indépendante.

Audit SHA-256 : `4138211757c103d9894e4b90593f512cd2a0529598ed9a2b0ba696679d8b1794`.

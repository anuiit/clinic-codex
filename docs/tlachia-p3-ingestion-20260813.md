# Tlachia P3 — mapping, acquisition bornée et simulation

Date de gel : 2026-08-13.

## Mise à jour clean-room (autoritative)

La politique append-only décrite plus bas pour le pilote 391 a été remplacée pour la
reconstruction globale. Le corpus Tlachia est désormais simulé **sans consulter ni dédupliquer
contre l'ancien dataset d'entraînement**. L'ancien snapshot ne sert donc ni de source, ni de
filtre. Seul le test métier final reste une quarantaine obligatoire.

Artefacts globaux exécutés :

- plan borné : `global-training-source-exact-v4.json`, SHA sémantique
  `516a7ea65bc1e1ed005fa539cf6d72610ff4a7ab9288efc5450e47ccdb2f11d5` ;
- téléchargement complet : 4 459 BMP, 8 918 requêtes Range, 141 105 342 octets
  décompressés et 61 534 514 pixels ;
- manifeste média : `global-exact-manifest-v3.json`, SHA sémantique
  `2d27f3166a6eaaa3f53dfe85cc2d6d11d9e65b8b683cc888dde1c364abeea6d1` ;
- simulation clean-room : `global-standalone-v1.json`, SHA sémantique
  `a57df53f1ba1493b3898674e2d5940b2e4584253140d97ef98af12d07ad4bc42`.

Le downloader exige ETag, Content-Range total exact, tailles, CRC-32, SHA-256 octets,
SHA-256 RGB canonique et limites de décodage. La simulation relit ensuite les 4 459 fichiers et
recalcule leurs hashes. Elle recalcule également le hash sémantique et les hashes pixels/octets
des 303 crops et 10 pages du test métier final.

Résultat de simulation autonome :

- 4 389 lignes retenues après quarantaine ;
- 269/286 classes couvertes ;
- split proposé 3 548 train / 423 dev / 418 locked test ;
- zéro groupe source partagé entre splits ;
- 18 groupes de doublons exacts cohérents, soit 20 lignes redondantes retirées ;
- 20 groupes pixels exacts multi-classe, soit 41 fichiers quarantainés ;
- 4 composants perceptuels multi-classe, soit 8 fichiers quarantainés ;
- un quasi-doublon du test métier (`t_030.zip:01_02_11.bmp`, distance dHash 4), quarantainé ;
- une classe couverte mais sous le plancher de deux lignes train : `tlamamallahuiztli`.

Les 17 classes absentes sont toutes des familles suffixées : `cuahuitl_1/2`, `ehuatl_2`,
`maitl_1/2`, `omitl_1/2`, `tepoztli/2`, `tlalpilli/2`, `tzontli/2` et
`xihuitl_1/2/3/4`. Elles correspondent à 18 lignes de mapping et 2 089 occurrences. La feuille
métier prioritaire est
`reports/tlachia-p3-20260813/mapping/global-business-review-priority-17-classes-v1.csv`.

Verdict : **NO-GO matérialisation et entraînement**. Il faut d'abord faire approuver ou rejeter
ces 18 mappings, reconstruire le plan, télécharger seulement les références additionnelles,
puis obtenir une simulation 286 classes sans conflit ni recouvrement métier. Aucun runtime,
registre modèle ou poids n'a été modifié.

## Décision de conception

Le graphe Tlachia et les ZIP n'expriment pas la même unité :

- l'API décrit les occurrences `codex → planche → zone → glyphe → élément` ;
- un BMP de thème dans un ZIP est une référence physique réutilisée par plusieurs occurrences.

Le pilote 391 contient 142 occurrences élément mais seulement 60 BMP de thème. Une occurrence
API n'est donc jamais multipliée en ligne d'entraînement. La clé physique est
`(archive, entry_path)` ; les occurrences restent des preuves relationnelles.

## Table Tlachia → taxonomie runtime

Le générateur `scripts/tlachia_p3.py mapping` produit :

- les 286 classes runtime, toujours présentes dans la table ;
- `runtime_class_label` 0–285, qui est l'ABI du modèle ;
- `legacy_class_index` 1–286, réservé aux anciens noms de dossiers ;
- une ligne par couple `(libellé Tlachia, thème)` avec les occurrences API qui la prouvent ;
- un hash sémantique qui lie la table au catalogue API et à `backend/codex_model/config.json`.

Politique du pilote : un libellé NFC exactement égal à une classe runtime est accepté
automatiquement seulement s'il n'appartient pas à une famille suffixée. Toute correspondance
par `valeur`, tout libellé absent et toute famille comme `xihuitl_1`/`xihuitl_3` reste soumise au
métier.

Résultat codex 391 :

- 286 classes runtime représentées dans la table ;
- 60 couples libellé/thème issus de 142 occurrences ;
- 48 couples acceptés exactement, couvrant 47 classes runtime ;
- 12 couples à revoir par le métier.

La feuille à remplir est
`reports/tlachia-p3-20260813/mapping/codex-391-business-review-v1.csv`. Les colonnes
`business_status`, `business_runtime_class_name`, `business_note`, `business_reviewer`,
`business_reviewed_at` et `business_ticket` sont réservées à la décision métier. Les statuts
acceptés sont `approved`, `rejected` et `quarantined`. La commande `apply-review` refuse un CSV
qui ne porte pas le hash exact de la proposition, un reviewer unique et une date unique.

Les douze lignes pilote sont :

| Libellé | Thème | Occurrences | Candidat | Motif |
|---|---:|---:|---|---|
| `?_05` | `05.99.99` | 1 | `cuetlaxtli` | candidat tiré de `valeur`, jamais automatique |
| `ahuehuetl` | `03.01.17` | 2 | — | absent de la taxonomie |
| `chichiquilli` | `05.13.08` | 1 | — | absent |
| `cuahuitl_1` | `03.01.01` | 2 | `cuahuitl_1` | famille suffixée |
| `huaxin` | `03.04.04` | 1 | — | absent |
| `quiltototl` | `02.01.56` | 1 | — | absent |
| `teponaztli` | `05.13.21` | 2 | — | absent |
| `tepoztli` | `05.10.12` | 1 | `tepoztli` | famille runtime apparentée |
| `tzontli` | `01.02.20` | 1 | `tzontli` | famille runtime apparentée |
| `tzopilotl` | `02.01.54` | 1 | — | absent |
| `xihuitl_1` | `03.02.16` | 1 | `xihuitl_1` | famille suffixée |
| `xihuitl_3` | `04.04.05` | 11 | `xihuitl_3` | famille suffixée |

## Téléchargement pilote

`scripts/tlachia_p3.py plan-download` produit un plan immuable avant tout octet image. Il exige
des limites explicites de fichiers, octets compressés, octets décompressés, requêtes et pixels.
Le downloader :

1. ne lit que le local header et le payload compressé de l'entrée choisie par HTTP Range ;
2. exige HTTP 206, `Content-Range` exact et ETag identique à l'inventaire ;
3. refuse chiffrement, data descriptor, ZIP64 implicite et compression autre que store/deflate ;
4. vérifie nom brut, tailles, CRC-32 et SHA-256 des octets ;
5. décode avec des limites 10 000 px par côté / 80 Mpx ;
6. calcule le même SHA-256 RGB canonique que le snapshot builder ;
7. refuse les conflits où un même pixel approuvé porte plusieurs classes.

Pilote 391 exécuté comme preuve de revue : 60 fichiers, 5 873 259 octets compressés,
6 812 104 octets décompressés, 120 requêtes Range, 2 265 254 pixels, 60 SHA d'octets et
60 SHA de pixels uniques, zéro conflit. Aucun ZIP complet n'a été téléchargé.

## Simulation historique du pilote 391 (supersédée pour le clean-room global)

La vue entraînable automatique ne contient que les 48 références exactes. La simulation est
append-only par rapport à `snapshot-471c74f801b749a9ad6e`, conserve ses affectations de split,
compare le test métier final par hash de pixels puis applique dHash64 avec distance de Hamming
≤ 4 contre les 9 266 lignes du parent et les 313 images du test métier (303 crops et 10 pages).

Résultat :

- 45/48 fichiers sont pixel-identiques à des lignes du parent, avec la même classe ;
- les trois autres (`tlantli`, `mazatl`, `tlalli`) sont des quasi-doublons perceptuels du parent
  (distances dHash 0, 0 et 1) avec la même classe ;
- zéro ajout net, zéro conflit de label, zéro collision exacte avec le test métier final ;
- splits projetés inchangés : 9 128 train / 56 dev / 82 locked test.

Le gate est donc **NO-GO entraînement Tlachia pour ce pilote** : le mapping métier n'est pas
terminé et, même avant cette revue, le codex 391 n'apporte aucun pixel indépendant au snapshot.
Entraîner dessus ne testerait pas l'apport Tlachia ; cela ne ferait que réentraîner sur les mêmes
images.

## Résultat global des 50 codex

Le crawl élémentaire global a terminé en quatre lots reprenables, à débit limité : 41 077
réponses API ont été capturées. La fusion hashée contient 50 codex, 1 351 planches, 2 000 zones,
37 674 glyphes/personnes et 70 050 occurrences élémentaires. Les 3 928 anomalies API sont des
doublons sémantiques explicitement conservés (3 874 éléments et 54 glyphes), pas des lignes
supplémentaires d'entraînement.

La table globale contient les 286 classes runtime et 896 couples `(libellé Tlachia, thème)` :

- 290 couples exacts, 57 389 occurrences et 269 classes runtime sont automatiquement
  exploitables au niveau sémantique ;
- 606 couples représentant 12 661 occurrences requièrent le métier ; 37 ont un candidat et
  569 restent sans correspondance automatique ;
- la feuille de revue est
  `reports/tlachia-p3-20260813/mapping/global-business-review-v1.csv` ;
- l'artefact lié est `global-runtime-v1.json`, SHA sémantique
  `b11becb985ea1804855c86a130cf1e3e02acb1a0c475d11c65f141758e287e04`.

La jointure ZIP produit 110 050 relations : 69 935 références d'élément, 36 985 glyphes,
1 888 zones et 1 242 planches. Elle signale 888 médias absents, 112 relations attachées à
l'archive `t_028.zip` absente et 25 ambiguïtés exactes point/underscore. Aucune jointure fuzzy
n'est appliquée.

Le plan global de revue porte sur 6 106 BMP physiques seulement — preuve que 69 935 relations
ne constituent pas autant d'images indépendantes — pour 157 686 597 octets compressés et
216 804 716 octets décompressés. Il n'a pas été exécuté. Le plan entraînable fail-closed v3
retient 4 459 références ayant une seule classe approuvée et aucune correspondance pendante :
269 classes, 57 170 occurrences, 104 329 547 octets compressés et 141 105 342 octets
décompressés. Il met en quarantaine 1 647 références physiques / 12 765 occurrences : 1 646
portent encore au moins une correspondance métier pendante et une porte deux classes exactes.

Deux cas illustrent cette barrière : `t_374.zip:05_05_01.bmp` associe `tilmatli` accepté à
`tlacatl` encore en revue ; `t_tepech.zip:08_01_08.bmp` associe exactement `coztic` et
`matlalin`. Les deux sont bloqués. La feuille physique exhaustive est
`global-physical-business-review-v2.csv`, liée par
`global-physical-review-v2.json` (SHA sémantique
`3a7a586eb71cf023ede7829cc4bdc55e2c3962ff6e9129e797a25e3295b504f6`). Le plan de
téléchargement entraînable est `global-training-source-exact-v3.json` (SHA sémantique
`7d186767a92eb3cf537e735dcf5bdaf81252906181c364c27f3f74546e45691c`). Aucun de ces
4 459 fichiers n'a encore été téléchargé.

## Décision historique avant reconstruction globale

Cette section reflète l'état antérieur à l'exécution clean-room globale. La décision autoritative
est celle de la mise à jour en tête de document : téléchargement exact terminé, simulation
autonome terminée, entraînement bloqué par les gates mesurés.

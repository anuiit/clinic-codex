# Spike de fusion glyphique run_50 + B1 (2026-08-13)

Ce harness est un sanity spike autonome, reproductible et fail-closed. Il compare
le checkpoint historique EfficientNet-B3 `run_50` à la baseline B1 sur les
glyphes JPG du lot Tlachia P4. Il ne constitue pas un gate élément, ne lance
aucun entraînement et ne peut promouvoir aucun modèle. Son verdict terminal est
toujours `INSUFFICIENT_FOR_ELEMENT_GATE`.

## Pourquoi le protocole est glyphique

Le benchmark BMP élément est contaminé et trop petit après déduplication :
104/119 BMP dev recouvrent le snapshot B1 et il ne reste que 31 couples dev,
12 classes, dont une seule commune avec `run_50`. En revanche, P4 fournit des
relations glyphe-vers-éléments exactes utilisables pour un diagnostic
multilabel. Un benchmark élément plus large, frais et disjoint en provenance
reste obligatoire avant toute décision globale.

## Protocole gelé

- Entrées P4 : plan, manifeste, vérification et corpus sous
  `reports/p4-multiview-20260813` et
  `backend/training_corpus/external/tlachia-multiview-pilot-v1`.
- Modèle historique réel :
  `/mnt/f/CODEX/AI_clinic_class/runs/run_50/best_model.pt`, SHA-256
  `9cfd73f37655bce2773f5b45b90bed66415144c16db72e0560a76685e95eefcd`.
  L'architecture EfficientNet-B3 et la tête 1536→768→512→256→33 sont
  reconstruites d'après le code et la configuration épinglés; les labels du
  checkpoint sont prioritaires.
- Baseline réelle : package B1
  `20260811-s14-snapshot-warmstart-b1`, projection SHA-256
  `52507c66ae568380c150a4ff94da78c41f4d07848c46cb366ba5a60c9f716d03`.
- Profil par défaut `whole_glyph_direct_b1` : chaque JPG glyphe entier passe
  directement dans `CodexClassifier.classify_batch(top_k=286)`. Il utilise
  donc exactement le preprocessing runtime (resize proportionnel LANCZOS,
  dimensions tronquées à l'entier, padding blanc et normalisation ImageNet),
  le backbone, la projection et les prototypes déployés. Les confiances cosine
  brutes des 286 classes sont la baseline. Le batch B1 est gelé à 16, lié au
  protocole/manifeste et reporté avec le nombre de batches. La latence inclut
  décodage JPG, preprocessing et forward complet, mais pas la construction du
  modèle.
- Profil diagnostique non défaut `whole_glyph_cached_b1` : il valide le cache P4
  `multiview.features.pt` (SHA-256
  `c2c3ecbd38e870ac55bbaf5f517b60f8d2489c11ccd5666dcddba351be641506`)
  et sa provenance, mappe les lignes strictes par SHA pixel, prend uniquement
  la vue entière `base`, puis exécute réellement la projection et les
  prototypes B1. L'extraction DINO déjà gelée n'est ni relancée ni incluse dans
  la latence; celle-ci mesure seulement le forward projection/prototypes B1 et
  l'inférence réelle `run_50`. Son preprocessing historique BILINEAR+arrondi
  n'est pas équivalent au runtime B1 LANCZOS+troncature : ce profil ne peut donc
  jamais être présenté comme baseline production.
- Profil conservé mais non défaut `mobile_sam_b1` : MobileSAM segmente chaque
  JPG, B1 score les crops, puis le maximum par classe agrège les scores au
  niveau du glyphe. Le budget est gelé à 32 segments : il n'est pas modifiable
  par la CLI et fait partie du manifeste du cache.
  Si MobileSAM ne fournit aucun crop valide, la politique gelée
  `whole_glyph_b1` classifie exactement une fois le JPG entier avec B1 et garde
  ses 286 scores; le glyphe n'est jamais exclu adaptativement. Chaque ligne
  conserve son mode et un `segment_count=0` pour ce fallback. Le compte, les
  IDs concernés et les latences du fallback sont liés au manifeste et reportés
  dans les résultats.
- Checkpoint MobileSAM :
  `/home/sina/.cache/mobile_sam/mobile_sam.pt`, 40 728 226 octets, SHA-256
  `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`.
  Le fichier homonyme incomplet sous `backend/` n'est jamais utilisé. Le
  preflight exige un chargement sûr `weights_only=True`, exactement 439
  tenseurs et l'inventaire de clés épinglé avant toute construction du modèle.
- Taxonomie : 31 classes partagées, 255 classes B1-only protégées bit à bit,
  et deux classes historiques seulement (`?`, `barra`).
- Split déterministe et page-disjoint : 18 planches train / 6 planches dev.
- Fuites : exclusion des 19 glyphes pixel-identiques au corpus `run_50`
  (dont 3 dev), exclusion dHash-64 à distance de Hamming ≤4 (aucun cas
  supplémentaire dans les entrées épinglées), plus exclusion des identités BMP
  traversant train/dev.
- AP stricte : seuls les glyphes dont tous les descendants sont résolus sont
  évalués. Les entrées business, annotations et locked-test sont interdites.

La baseline `b1` est une copie strictement identique des scores cosine bruts
produits par le profil direct runtime : aucune température ni recalibration
post-cache ne lui est appliquée.
`b1_calibrated` est une variante descriptive distincte et non candidate. Les
variantes restantes sont `legacy`, `prior_global`, `cascade` et un `oracle`
descriptif non candidat. Les paramètres des variantes sont ajustés sur train
uniquement, puis le dev reste réservé à l'évaluation; tous les gains et gates
sont comparés au `b1` brut.
Les mesures sont AP macro et sample, Recall@5, set-F1, ECE-10, Brier multilabel,
latence par glyphe et bootstrap apparié par planche. Pour les truths contenant
une classe B1-only, AP sample, Recall@5 et top-1 sont aussi reportés dans
le ranking global des 286 classes. Ainsi, remonter une classe partagée devant
une truth B1-only est bien compté comme une dégradation, même si les 255 logits
protégés restent inchangés.

Le sanity gate indicatif emploie un seul estimand, la moyenne non pondérée des
AP par planche : gain d'au moins 1 point et borne basse de l'IC bootstrap
strictement positive. Il exige aussi une latence sérielle d'au plus
250 ms/glyphe, l'identité bit à bit des 255 logits B1-only et aucune dégradation
de leurs AP, Recall@5 ou top-1 conditionnels. Même réussi, il ne change ni le
verdict, ni l'interdiction d'entraînement global automatique, ni
l'inéligibilité à la promotion.

## Utilisation sûre

Depuis la racine du dépôt :

```bash
# Lecture seule : hashes, relations, split et fuites.
backend/.venv/bin/python scripts/autoresearch_legacy_fusion.py verify --inputs-only

# Prévisualisation seulement; ne charge pas les modèles et n'écrit rien.
backend/.venv/bin/python scripts/autoresearch_legacy_fusion.py prepare

# Action explicite et coûteuse : inférence réelle, cache immuable write-once.
# Le profil par défaut exécute le classifier B1 runtime directement sur les JPG.
backend/.venv/bin/python scripts/autoresearch_legacy_fusion.py prepare \
  --device cuda --batch-size 16 --apply

# Profil features-cache diagnostique, jamais baseline production.
backend/.venv/bin/python scripts/autoresearch_legacy_fusion.py prepare \
  --profile whole_glyph_cached_b1 --device cuda --batch-size 16 --apply

# Profil MobileSAM conservé, uniquement sur demande explicite.
backend/.venv/bin/python scripts/autoresearch_legacy_fusion.py prepare \
  --profile mobile_sam_b1 --device cuda --batch-size 16 --apply

# Évalue les variantes depuis le cache; résultats immuables write-once.
backend/.venv/bin/python scripts/autoresearch_legacy_fusion.py run-all

# Vérifie aussi le cache s'il existe.
backend/.venv/bin/python scripts/autoresearch_legacy_fusion.py verify
```

Les scores sont stockés en NPZ avec `allow_pickle=False`. Avant toute lecture
des tenseurs, le harness valide le hash du NPZ et les bindings du manifeste :
checkpoint `run_50`, projection/prototypes B1, sources d'inférence et imports,
plan/manifeste P4, IDs, pixels, truths, planches et split.

Les artefacts sont confinés à `autoresearch/legacy-fusion-20260813/`. Une
réexécution identique est idempotente; toute tentative d'écraser un artefact
avec un contenu différent échoue.

## Résultats du spike direct

La campagne `whole_glyph_direct_b1` a été exécutée le 13 août 2026 sur GPU.
Après exclusion des 19 glyphes déjà vus pixel-identiquement par `run_50`, le
benchmark contient 157 glyphes stricts : 112 sur 18 planches de calibration et
45 sur 6 planches de validation. Aucun test métier, ensemble d'annotations ou
`locked_test` n'a été lu.

| Variante | AP sample | AP macro | Recall@5 | AP moyenne par planche | Delta vs B1 | IC95 du delta |
|---|---:|---:|---:|---:|---:|---:|
| B1 direct | 17,32 % | 28,25 % | 18,66 % | 16,94 % | référence | — |
| Legacy `run_50` | 7,10 % | 25,73 % | 7,88 % | 7,26 % | -9,68 pts | [-26,14 ; +1,95] pts |
| B1 + prior global | 16,28 % | 31,03 % | 19,66 % | 15,99 % | -0,95 pt | [-4,71 ; +1,60] pts |
| B1 + cascade | 16,25 % | 30,88 % | 19,66 % | 15,97 % | -0,98 pt | [-4,73 ; +1,62] pts |
| Oracle descriptif | 20,44 % | 29,73 % | 22,06 % | 20,14 % | +3,20 pts | [+1,26 ; +5,57] pts |

Le prior global et la cascade échouent au sanity gate : leur delta primaire
est négatif, leur IC traverse zéro et le classement global des classes B1-only
est dégradé. Aucun `sanity_winner` n'est retenu; le téléchargement global et
l'entraînement d'un nouveau classifieur restent donc volontairement non
déclenchés. L'oracle confirme seulement qu'un potentiel de complémentarité
existe si une règle de sélection plus discriminante peut être apprise sur un
benchmark frais et beaucoup plus grand.

Latence GPU mesurée sur ce spike : 6,47 ms/glyphe pour B1 direct, 8,17 ms pour
`run_50`, soit 14,64 ms/glyphe en exécution sérielle. Les preuves immuables sont
`autoresearch/legacy-fusion-20260813/whole_glyph_direct_b1/results.json`
(SHA-256 `b7b80a16c14ff13bc51a3b740aa83583dbb647c12217dd93054cab7354a2d17e`)
et `final.json`; le verdict demeure `INSUFFICIENT_FOR_ELEMENT_GATE` et
`automatic_global_training_allowed=false`.

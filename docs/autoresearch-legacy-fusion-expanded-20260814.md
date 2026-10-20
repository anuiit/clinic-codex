# Benchmark élargi de fusion legacy + B1

Ce lot étend le spike de 157 glyphes sans ouvrir le test métier. Le plan est
figé à 2 000 glyphes JPG Tlachia, au moins 200 planches, 50 planches de dev et
10 glyphes maximum par planche. Seuls les glyphes dont tous les éléments
descendants disposent d'une correspondance `accepted_exact` ou
`approved_manual` sont admissibles.

Les quatre catalogues historiques sont liés par leur SHA-256 de fichier exact,
car leur ancienne canonicalisation sémantique diffère du helper courant. Le
nouveau plan, lui, utilise le schéma et le hash canonique courants. Tout drift
de fichier source bloque avant le téléchargement.

Budgets : 2 000 fichiers, 4 000 requêtes HTTP Range, 80 MB compressés,
100 MB décompressés et 1 Gpixel. Aucun fallback ZIP entier n'est autorisé.
Le téléchargement vérifie ETag, Content-Range, CRC32, tailles ZIP, SHA-256 des
octets et des pixels.

```bash
PYTHONPATH=backend backend/.venv/bin/python \
  scripts/autoresearch_legacy_fusion_expanded.py plan \
  --output reports/legacy-fusion-expanded-20260814/plan-v2.json

PYTHONPATH=backend backend/.venv/bin/python \
  scripts/autoresearch_legacy_fusion_expanded.py download \
  --plan reports/legacy-fusion-expanded-20260814/plan-v2.json \
  --capture-dir reports/legacy-fusion-expanded-20260814/raw \
  --output-root backend/training_corpus/external/legacy-fusion-expanded-20260814 \
  --manifest-output reports/legacy-fusion-expanded-20260814/manifest-v1.json \
  --apply
```

Après le téléchargement, l'évaluation unique réutilise les scoreurs directs
B1/run_50 existants. Elle refait les identités d'octets/pixels, exclut
dynamiquement les recouvrements pixel/dHash de `run_50`, puis applique les
variantes B1, legacy, prior, cascade et oracle. Les paramètres de fusion sont
choisis sur les planches train uniquement.

Si l'exclusion `run_50` vide une planche de validation, le split propre garde
les planches dev survivantes puis complète jusqu'à 50 avec les planches train
au plus petit hash stable `expanded-clean-dev-v1`. Cette règle est appliquée
avant tout score modèle et consignée dans le résultat.

```bash
PYTHONPATH=backend backend/.venv/bin/python \
  scripts/autoresearch_legacy_fusion_expanded.py evaluate \
  --plan reports/legacy-fusion-expanded-20260814/plan-v2.json \
  --manifest reports/legacy-fusion-expanded-20260814/manifest-v1.json \
  --output-root backend/training_corpus/external/legacy-fusion-expanded-20260814 \
  --device cuda --batch-size 16 --apply \
  --output reports/legacy-fusion-expanded-20260814/evaluation-v1.json
```

Le gate reste préenregistré : +1 point d'AP moyenne par planche, borne basse
IC95 positive, aucune régression B1-only et latence série ≤250 ms/glyphe. Il
reste strictement de recherche : aucune promotion; un passage de gate autorise
uniquement l'itération d'entraînement légère suivante.

ECE/Brier restent descriptifs : une sigmoïde est appliquée aux similarités
cosinus B1, sans prétendre produire des probabilités calibrées.

L'itération protégée réordonne les 31 classes communes uniquement dans leurs
positions de score B1 existantes. Les 255 scores B1-only et leurs rangs globaux
restent donc inchangés par construction.

Le rapport lie aussi les lignes finalement notées par un SHA canonique de
`row_id`, pixel, planche et split, avec les comptes par split et pixels uniques.

## Résultat

Après exclusion de 562 glyphes pixel-identiques au corpus `run_50`, le lot
propre contient 1 438 glyphes : 1 226 train sur 333 planches et 212 dev sur 50
planches. La cascade libre gagne 0,09899 AP moyenne par planche, IC95
[0,04370 ; 0,15627], mais dégrade le ranking B1-only. La cascade protégée
préserve exactement B1-only mais ne gagne que 0,00404, IC95
[-0,02949 ; 0,03293]. Verdict : plateau, aucun entraînement léger automatique.

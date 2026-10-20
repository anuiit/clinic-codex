# Tlachia P4 — pilote multi-vue borné

Ce pilote prépare l'acquisition reproductible des vues `glyph_occurrence`,
`zone` et `planche` en vue d'une adaptation auto-supervisée. Il ne télécharge
jamais une archive ZIP complète : chaque image est liée à une entrée exacte du
catalogue ZIP et à son CRC, puis récupérée par deux requêtes HTTP Range bornées.

## Contrat de données

- Le groupe de séparation est toujours `codex + planche`.
- Chaque image reste `unlabeled_ssl`.
- Les classes runtime approuvées trouvées parmi les descendants API sont de
  l'évidence de couverture. Pour un glyphe, elles peuvent devenir une cible
  auxiliaire multi-label explicite. Elles ne sont jamais une vérité terrain
  single-label de crop.
- En l'absence de boîtes englobantes, une zone, un glyphe ou une planche ne
  prouve pas la localisation exacte d'un élément. Ces vues ne doivent donc pas
  être injectées comme crops supervisés.
- Les annotations métier et le locked-test sont absents des entrées et des
  critères de sélection.
- Le résultat est un pilote de recherche et porte toujours
  `promotion_allowed=false`.

La sélection est déterministe et group-aware : elle maximise d'abord les
nouvelles classes runtime observées, puis utilise un hash stable. Les caps par
planche sont par défaut de 8 glyphes, 2 zones et 1 planche. Les gates demandent
au moins 20 planches et, lorsque le catalogue le permet, 100 classes runtime.

## Commandes

Construire l'artefact immuable après revue des budgets :

```bash
backend/.venv/bin/python scripts/tlachia_p4_multiview.py plan \
  --max-files 300 --max-groups 24 \
  --max-compressed-bytes 250000000 \
  --max-uncompressed-bytes 250000000 \
  --max-requests 600 --max-pixel-count 1000000000 \
  --output reports/tlachia-p4-20260813/multiview-plan.json
```

Le téléchargement n'est autorisé qu'après inspection du plan et avec
`--apply`, plus trois destinations explicites :

```bash
backend/.venv/bin/python scripts/tlachia_p4_multiview.py download \
  --plan reports/tlachia-p4-20260813/multiview-plan.json \
  --capture-dir reports/tlachia-p4-20260813/http-capture \
  --output-root backend/training_corpus/tlachia-multiview-pilot \
  --manifest-output reports/tlachia-p4-20260813/multiview-manifest.json \
  --apply
```

Enfin, la vérification recalcule les SHA-256 des fichiers et pixels, contrôle
les CRC, les budgets, les caps, les gates et refuse tout doublon pixel traversant
deux groupes :

```bash
backend/.venv/bin/python scripts/tlachia_p4_multiview.py verify \
  --plan reports/tlachia-p4-20260813/multiview-plan.json \
  --manifest reports/tlachia-p4-20260813/multiview-manifest.json \
  --output-root backend/training_corpus/tlachia-multiview-pilot \
  --output reports/tlachia-p4-20260813/multiview-verification.json
```

Le téléchargement réel n'a pas été lancé pendant l'implémentation.

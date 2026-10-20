# Catalogue Tlachia P0-P2

Date : 2026-08-13. Statut : métadonnées uniquement, aucun import ML et aucun
téléchargement massif d'images.

## Décision

Tlachia est l'autorité sémantique. Les ZIP publiés sous
`https://sup-infor.com/Marc/Images/` sont l'autorité des fichiers image. Une
relation n'est acceptée que par les identifiants API (`codex_id`, `cote`,
`theme`) et un nom ZIP exact normalisé. Les labels textuels ne sont jamais
utilisés pour une jointure approximative.

Le test métier final reste hors apprentissage. Son manifeste, ses 303 crops et
ses 10 pages sont liés à un artefact de quarantaine par leurs hashes déjà
publiés. L'absence actuelle de `codex_id` et de `cote` est enregistrée comme
`unmapped`; elle n'est pas devinée depuis les classes ou les chemins.

## Livrables

- `scripts/tlachia_catalog.py` : CLI reprenable `preflight`, `inventory-api`,
  `inventory-zips`, `join`, `verify-quarantine`.
- `backend/codex_pipeline/data/tlachia/` : canonicalisation, transport capturé,
  graphe API, lecteur de répertoire central ZIP et quarantaine.
- `backend/tests/test_tlachia_catalog.py` : tests sans réseau.
- `reports/tlachia-p0-p2-20260813/` : captures et catalogues générés, ignorés
  par Git et reproductibles offline.

## Protocole de sécurité

- allowlist exacte des deux hôtes et chemins;
- HTTPS, redirects interdits, budgets de requêtes et de bytes;
- chaque réponse brute est hashée et rejouable sans réseau;
- une requête Range doit répondre `206`; une réponse `200` est refusée avant
  lecture d'une archive complète;
- ZIP multi-disque et ZIP64 refusés; noms décodés selon le bit UTF-8 ou CP437,
  nom brut hashé, collisions NFC/casefold signalées;
- aucune commande P0-P2 ne sait extraire ou télécharger les fichiers image;
- les artefacts dérivés portent un hash sémantique canonique NFC sans float;
- les sorties finales sont write-once; un verrou de capture concurrente reste
  fail-closed après incident pour inspection manuelle.
- `preflight-v2.json` atteste que le snapshot d'entraînement réel a été comparé
  à la quarantaine métier; l'ancien `preflight.json` reste une preuve v1
  immuable et n'est pas réécrit.

## Exécution validée

```bash
backend/.venv/bin/python scripts/tlachia_catalog.py --json preflight

backend/.venv/bin/python scripts/tlachia_catalog.py --json inventory-api \
  --store reports/tlachia-p0-p2-20260813 --capture codex-391-v1 \
  --codex-id 42 --include-elements --max-requests 300 \
  --output reports/tlachia-p0-p2-20260813/catalog/api-codex-391-v4.json

backend/.venv/bin/python scripts/tlachia_catalog.py --json inventory-zips \
  --store reports/tlachia-p0-p2-20260813 --capture zip-all-v2-hardened \
  --max-requests 200 \
  --output reports/tlachia-p0-p2-20260813/catalog/zip-all-v3.json
```

Pilote codex 391 : 1 planche, 3 zones, 141 glyphes/personnes, 142 occurrences
d'éléments, 236 entrées ZIP et 287 relations API/ZIP exactes (planche, zones,
glyphes/personnes et références d'éléments); aucune anomalie
de jointure. L'API contient trois doublons sémantiques stricts d'éléments avec
des variantes de traduction; ils sont conservés comme `raw_variants` et
signalés.

Inventaire ZIP global : 49 archives, 47 251 entrées et 2 146 051 901 octets
référencés. Le protocole Range a lu 3 212 293 octets de fins d'archives puis
2 316 234 octets de ranges additionnels; 4 540 142 octets correspondent aux
répertoires centraux. Les tails peuvent contenir quelques octets compressés
précédant le répertoire : aucune image n'a été extraite ni décodée, mais il ne
faut pas qualifier l'opération de « zéro octet pixel ». Aucune collision de nom
normalisé n'a été observée.

Le graphe structurel global contient 50 codex, 1 351 planches, 2 000 zones et
37 674 glyphes/personnes. Sa jointure avec les ZIP donne 40 115 relations
exactes, aucune ambiguïté et 910 absences : 49 descendants du codex `028` dont
l'archive `t_028.zip` manque, plus 861 JPG référencés par l'API mais absents des
ZIP. Ces absences restent des anomalies bloquantes pour un corpus exhaustif;
elles ne sont pas réparées par correspondance approximative.

## Portée et suite

P0-P2 produisent un catalogue, pas un dataset. Avant P3+ :

1. terminer/rejouer offline le graphe API global et quantifier les anomalies;
2. exécuter les relations élémentaires exhaustives par lots bornés et
   reprenables;
3. faire valider la correspondance Tlachia vers les 286 classes runtime;
4. planifier le téléchargement image par lot, avec CRC, SHA-256 bytes/pixels,
   décodage et déduplication perceptuelle;
5. simuler un child snapshot sans jamais réaffecter les groupes du parent;
6. seulement après revue, matérialiser un corpus d'entraînement distinct.

Le `source_group` Tlachia est toujours
`tlachia:codex:<id>:planche:<cote>`. Les fallbacks `source-path:` et
`image-pixels:` sont interdits pour ces données. Toute composante reliée au
test métier, toute collision de classe ou toute jointure ambiguë est mise en
quarantaine et bloque l'export ML.

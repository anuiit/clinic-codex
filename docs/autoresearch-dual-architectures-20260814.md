# Recherche d'architectures sur les corpus historique et Tlachia — 2026-08-14

## Décision

Le meilleur remplacement **from scratch** est `v1_ln_mlp` (LayerNorm + MLP), mais son avantage sur le contrôle froid est fragile et il reste sous B1. L'adaptateur résiduel B1 conserve l'ancien domaine et améliore le nouveau, mais son gain moyen de +0,581 point reste sous la gate de +1 point : il est rejeté pour l'instant.

Aucun modèle n'est promu et aucun test métier, registre ou runtime de production n'a été modifié. Les caches sources contiennent des lignes `locked_test`; elles sont chargées lors de la désérialisation puis filtrées et ne sont jamais utilisées, prototypées, sélectionnées ou évaluées.

## Données et protocole

- Ancien corpus : 9 128 lignes train, 56 dev, 8 classes observées dans dev.
- Nouveau corpus Tlachia : 3 763 lignes train, 465 dev, 208 classes observées dans dev.
- Entraînement combiné : 12 891 lignes après garde anti-doublon; aucun conflit ni chevauchement exact avec dev détecté.
- Évaluation : prototypes recalculés depuis train uniquement; top-1 micro/macro et top-3 sur chaque dev.
- Confirmation : seeds 42, 73 et 101.
- Les métriques du petit dev historique ont une variance élevée et ne suffisent pas pour une promotion.

## Architectures

| Version | Architecture | Paramètres | Statut |
|---|---|---:|---|
| B1 warm | Projection B1 existante, référence | 197 120 | Référence produit |
| V0 | `384 → 384 GELU Dropout → 128` | 197 120 | Contrôle froid |
| V1 | `LayerNorm(384) → MLP V0` | 197 888 | Meilleur modèle froid |
| V2 | `LayerNorm(384) → Linear(128)` | 49 920 | Trop faible, screening seulement |
| V3 | Résiduel projeté `384 → 128` | 247 168 | Meilleur nouveau dev froid, mais instable sur l'ancien |
| Adaptateur B1 | B1 gelé + `LayerNorm → Linear(128)` résiduel, gate scalaire | 50 049 entraînables | Candidat exploratoire principal |

## Résultats

Moyennes des seeds 42/73/101, sauf V2 (screening seed 42). Les valeurs sont des pourcentages.

| Modèle | Ancien top-1 micro | Ancien top-1 macro | Nouveau top-1 micro | Nouveau top-1 macro | Nouveau top-3 |
|---|---:|---:|---:|---:|---:|
| B1 warm combiné | **92,86** | **85,71** | 93,12 | 93,25 | 94,62 |
| V0 froid combiné | 84,52 | 67,87 | 97,85 | 97,49 | 98,35 |
| V1 LayerNorm+MLP | 85,71 | 70,65 | 97,71 | 97,49 | 98,28 |
| V2 LayerNorm+Linear (1 seed) | 64,29 | 51,84 | 86,67 | 88,26 | 95,70 |
| V3 résiduel froid | 85,12 | 69,26 | 97,92 | 97,65 | 98,28 |
| Adaptateur résiduel B1 | **92,86** | **85,71** | 94,62 | 94,41 | 96,56 |

### Comparaisons appariées

- V1 contre V0 froid : +1,389 point de macro moyen, mais le gain vient surtout du seed 42 sur l'ancien dev; les deux autres seeds sont nuls sur ce domaine et le nouveau domaine est pratiquement inchangé.
- V3 contre V0 froid : +0,775 point en moyenne, avec une régression de 4,17 points sur l'ancien dev au seed 73; rejeté par la gate.
- Adaptateur contre B1 : aucun changement macro sur l'ancien dev; +1,362, +1,122 et +1,002 points macro sur le nouveau dev. Gain macro moyen sur les deux domaines : +0,581 point.
- L'adaptateur est `DISCARD` : il manque la gate de +1 point et son LR/budget avaient en plus été choisis après observation de ces mêmes dev sets.

## Lecture du Council

Le Council a recommandé de comparer des têtes simples avant LoRA/VICReg, d'utiliser un contrôle froid apparié et de privilégier un adaptateur résiduel initialisé exactement sur B1. Le partenaire Opus a répondu; le partenaire GLM a expiré deux fois. La synthèse est donc explicitement partielle, sans consensus artificiel.

Session persistante : `adv_20260811T223304_22869a68`; synthèse : `C:\Users\anmou\.argos\workspace\.omx\argos-council\codex-subagent-panel\turns\009\synthesis.md`.

## Artefacts vérifiables

- Comparatif corrigé : `reports/autoresearch-dual-architectures-20260814-v4/results.json`.
- Adaptateur corrigé : `reports/autoresearch-b1-residual-adapter-20260814-v2/results.json`.
- Les campagnes `dual-architectures-v3` et `b1-residual-adapter` sans suffixe v2 sont conservées mais invalidées : la référence B1 y avait son Dropout actif. Le correctif force désormais `eval()` dans l'évaluateur commun et maintient la base B1 en `eval()` pendant l'entraînement de l'adaptateur.

## Suite minimale fiable

Ne pas multiplier les architectures. B1 reste la référence. La seule suite justifiée serait de geler l'adaptateur actuel et de l'évaluer sur un nouveau holdout page/source-disjoint qui n'a servi ni au probe ni au choix du LR. S'il dépasse alors +1 point avec un IC95 inférieur positif et sans régression de domaine, seulement préparer un package candidat et une évaluation métier unique.

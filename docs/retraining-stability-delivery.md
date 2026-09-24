# Livraison réentraînement fiable — 17 septembre 2026

## Périmètre livré

Le [plan détaillé](retraining-stability-plan.md) regroupe les six lots validés. Travail dans la branche `codex/retrainable-release`. Aucun push/merge, aucune activation de modèle et aucune migration de données utilisateur pendant cette passe.

| Demande | Comportement livré |
|---|---|
| Analyse → annoter la région | Conservation de l'index original, focus après montage de la liste et chargement de l'image ; inspector scrollable. |
| Validations absentes de « Validés » | Filtre corrigé et décisions persistées dans SQLite ; une décision périmée n'est pas entraînable. |
| Images Dataset lentes | Chargement différé/décodage asynchrone natifs ; lookup média limité à l'analyse et à l'élément demandé, sans rescanner toute la file. Cache privé revalidé, autorisation toujours vérifiée. |
| Revenir sur une annotation | Remettre à vérifier, corriger, consulter les révisions, restaurer une version comme « à vérifier ». Conflit 409 au lieu d'écraser une modification concurrente. |
| Nom original | Nom conservé dans la soumission, la revue, le Dataset, les snapshots et la comparaison. Les noms identiques ne servent jamais d'identifiants. Un ancien nom non enregistré ne peut pas être reconstitué. |
| `unknown base class: tzapotl` | Classe proposée visible comme non confirmée ; confirmation explicite dans Classes ; ID stable ajouté après les IDs historiques ; apprentissage dans un candidat, jamais modification implicite du modèle actif. |
| Retour admin → app | Lien « Retour à l’analyse », garde sur correction non sauvegardée. |
| Classes lisibles | Recherche, statut actif/à confirmer/confirmé, ID et compteurs de validation/utilisabilité. |
| Interface entraînement | Une action principale : vérifier → créer un candidat → comparer. Options et logs repliés ; pas de perte/score inventés. Modification des données ou paramètres invalide la vérification précédente. |
| Comparaison | Vrais classifieurs sur les mêmes découpes ; anciennes/nouvelles classes séparées, support, top-3, gains/régressions, rejet, latence ; image source et rectangle, filtres par résultat et groupe. |
| Plusieurs utilisateurs | Dataset soumis partagé selon les rôles ; brouillons/historique IndexedDB séparés par ID de compte. Import legacy réservé à l'administrateur local et attribué une seule fois à un compte. Opérateur ML indépendant de l'accès à la file de triage. |

## Parcours utilisateur

1. Installer et démarrer selon [INSTALL.md](../INSTALL.md). Garder le mode local standard : `ADMIN_TRAINING_SNAPSHOT_DIR` non défini.
2. Importer une image existante, analyser, sélectionner une région et l'annoter. Vérifier le nom et la zone, marquer comme prêt, envoyer pour revue.
3. Dans Trier, valider les exemples corrects. Une soumission est immutable : pour modifier une soumission déjà reçue, utiliser la correction admin ; ne pas la renvoyer avec un autre contenu sous le même ID.
4. Dans Classes, confirmer les nouveaux noms après vérification de l'orthographe. Les variantes de casse/Unicode proches ne sont jamais fusionnées silencieusement.
5. Dans Entraîner, vérifier la préparation puis créer le candidat. Il utilise le prior livré, les annotations actuellement approuvées et un DINOv2 local épinglé ; pas de téléchargement implicite pendant la run.
6. Comparer le candidat au modèle actif, sur ses exemples capturés ou sur une analyse soumise. Pour une page sans étiquette validée, afficher le désaccord, pas un score de justesse inventé.
7. Si un exemple est douteux : Dataset → Ouvrir dans le triage → Remettre à vérifier. Refaire un candidat après correction ; les anciens candidats ne sont pas réécrits.

Le protocole réserve des pages avant l'apprentissage lorsqu'il le peut, et lie les duplicatas exacts pour éviter leur séparation entre train/test. Les pré-affectations train/val/test de la revue ne sont pas le partage définitif du candidat : le snapshot et le rapport de la run font foi.

## Migration des anciennes validations

Arrêter les écritures de l'application et sauvegarder le dossier complet `backend/annotations`. Depuis la racine du repo, prévisualiser :

```powershell
.\backend\.venv\Scripts\python.exe scripts/migrate_annotation_reviews.py --annotations-dir backend/annotations
```

Puis, après examen du résultat :

```powershell
.\backend\.venv\Scripts\python.exe scripts/migrate_annotation_reviews.py --annotations-dir backend/annotations --apply
```

Linux : remplacer l'exécutable par `backend/.venv/bin/python`. Le script valide le JSON, crée/vérifie une sauvegarde, publie SQLite sans écraser une base existante et reste idempotent. Conserver le JSON de sauvegarde ; SQLite devient ensuite l'autorité. Les snapshots avancés antérieurs doivent être reconstruits avec le nouveau digest canonique. Ne pas restaurer seulement un ancien JSON après migration.

Les anciennes bbox décimales sont arrondies comme lors de la sauvegarde d'une annotation. La prévisualisation affiche `normalized_bboxes` et `quarantined_invalid_bboxes` ; seules les boîtes réellement inutilisables sont omises de SQLite et leurs éléments redeviennent « à vérifier », hors entraînement. Le JSON original reste dans la sauvegarde vérifiée : revoir ces éléments manuellement après l'import.

### Archive `annotations (2).zip` testée le 24 septembre

Sur une installation Windows, extraire le ZIP dans un dossier neuf, puis exécuter ces commandes depuis la racine du dépôt avec son Python backend (remplacer `<extraction>\annotations` par le vrai chemin) :

```powershell
.\backend\.venv\Scripts\python.exe scripts/repair_missing_annotation_crops.py --annotations-dir "<extraction>\annotations"
.\backend\.venv\Scripts\python.exe scripts/repair_missing_annotation_crops.py --annotations-dir "<extraction>\annotations" --apply
.\backend\.venv\Scripts\python.exe scripts/migrate_annotation_reviews.py --annotations-dir "<extraction>\annotations"
.\backend\.venv\Scripts\python.exe scripts/migrate_annotation_reviews.py --annotations-dir "<extraction>\annotations" --apply
```

Pour ajouter cette archive à une installation possédant déjà `backend/annotations`, utiliser `scripts/merge_annotation_archive.py --source "<extraction>\annotations" --target backend/annotations` puis la même commande avec `--apply`. La fusion refuse les ID en collision et sauvegarde SQLite ; ne pas recopier `review-state.sqlite3` à la main. Avant de déplacer une installation ayant des validations antérieures à cette version, prévisualiser puis appliquer `scripts/upgrade_annotation_review_portability.py --annotations-dir backend/annotations` (ajouter `--apply` pour l'appliquer). Les nouveaux contrôles utilisent le contenu des PNG et des chemins relatifs ; le dossier complet peut ensuite être déplacé avec ses décisions et confirmations de classes.

Sur l'archive testée : 491 décisions importées (475 validées, 16 rejetées), 6 boîtes décimales normalisées et 5 découpes dérivées recréées depuis les images/boîtes originales. Les 21 images et les `metadata.json` d'origine n'ont pas été modifiés. Neuf classes hors modèle de base ont été confirmées localement à partir de ces validations. L'apprentissage réel a créé un candidat **sans activation** : 509 validations locales au total, 486 images uniques retenues, 21 doublons et 2 annotations contradictoires écartées de la run. Apprentissage : actif 206/434, candidat 226/434 ; test réservé : 24/52 pour les deux. Ce test réservé ne prouve donc aucun gain de généralisation.

Une base illisible bloque l'API (503) et n'est jamais supprimée/recréée automatiquement. Le dossier complet contient aussi les sources et découpes nécessaires à une restauration cohérente.

## Vérification effectuée

- Backend Linux : **666 tests réussis, 67 ignorés**. Les tests ignorés restent explicitement optionnels/dépendants de l'environnement ; ce n'est pas une preuve de leur exécution.
- Scripts : **40 tests réussis, 4 ignorés**.
- Frontend (audit du 23 septembre) : **303 tests réussis**, build et lint réussis après les derniers ajustements.
- Suite navigateur Playwright livrée : **12 tests réussis, 1 ignoré** (parcours live opt-in). Les parcours réels ci-dessous ont été exécutés séparément, sans réponses API simulées.
- Windows natif : **124 tests réussis, 2 ignorés**, dont les quatre tests PowerShell 5.1/7 (verrous partagés et chemins avec espaces). Code Python testé dans une copie NTFS, pas exécuté depuis une UNC.
- Réentraînement réel CPU/API sous Linux et Windows natif : cinq images existantes, trois en apprentissage et deux réservées, dry-run sans candidat puis candidat et comparaison avec médias HTTP 200.
- Dernière répétition native Windows : candidat `20260917T211837Z-nogit-4b2522de`, dry-run/full/comparaison réussis depuis un chemin NTFS contenant des espaces ; dimensions source et type MIME des médias vérifiés.
- Résultat mesuré : actif **2/2**, candidat **2/2** sur les exemples réservés, **0 gain, 0 régression**. Ce résultat n'établit aucune amélioration.
- Nouvelle classe, Windows réel : `teocomitl` retirée uniquement du modèle de base d'une fixture isolée puis réintroduite depuis ses vraies images existantes. Confirmation, ID 302, dry/full/comparaison réussis ; nouvelle classe **1/1** sur un exemple réservé. Test technique contrôlé, pas validation scientifique.
- Navigateur Chromium réel sur serveurs isolés : connexion, Dataset et chargement d'images, filtre Validés, remise à vérifier, historique/restauration, dry-run, entraînement CPU, comparaison avec pages/rectangles, retour à l'application.
- Audit de robustesse Chromium : premier administrateur créé sur base vierge puis déconnexion/reconnexion ; cinq onglets parcourus à 390 px ; menus et accès directs vérifiés pour administrateur, reviewer, opérateur ML et contributor. Réponses API 503 ou 200 incomplètes, historique invalide et découpes 404 affichent un état d'erreur ou un repli visible, sans écran blanc. Le bouton Réessayer recharge la file après retour de l'API.
- La file conserve les anciens éléments à zone absente pour correction au lieu de les masquer. Une restauration réussie reste annoncée comme telle si seul le rafraîchissement échoue.
- Contrôle final des médias de comparaison : cinq découpes effectivement chargées dans le navigateur, cinq pages sources authentifiées avec un type MIME image et cinq rectangles affichés.
- Compte opérateur ML réel : accès à Entraîner et Comparer sans permission de lecture de la file de triage.
- Page entière `387_769v.jpg` réellement segmentée : **47 régions**, passage de la région **45** vers l'éditeur, index et focus conservés.
- Aucun contenu utilisateur ni modèle actif utilisé pour les mutations de tests. Les fixtures proviennent uniquement des images présentes dans le dépôt.

Les copies propres de test sont des exports du contenu de travail, pas des clones distants des modifications non publiées. L'installation complète depuis Internet n'a pas été refaite dans cette passe ; les assets locaux vérifiés ont été réutilisés. Le test opt-in de clone neuf reste disponible avec `CLINIC_LOCAL_RETRAIN_E2E=1`.

## Limites conservées volontairement

- Candidat seulement : aucune activation, même avec un bon score.
- Ni nouvelles images, ni apprentissage de MobileSAM/DINOv2 ; adaptation des prototypes à projection gelée.
- L'indépendance vis-à-vis des données historiques du prior est inconnue. La détection de doublons exacts ne détecte pas toutes les variantes redimensionnées. Une seule page/exemple ne prouve pas la généralisation.
- Application locale à rôles, pas service multi-tenant Internet. Ne pas ouvrir les ports pour en faire un déploiement partagé.
- Les historiques locaux n'ont pas de synchronisation entre appareils. L'ancien historique sans propriétaire exige une décision de récupération administrateur.
- Un crash avant commit peut laisser une découpe dérivée orpheline, jamais une décision partiellement enregistrée. Nettoyage différé si le volume disque le justifie.
- Pas de benchmark de charge à grande échelle ni certification de toutes les versions Windows/GPU. Les preuves portent sur les environnements et parcours listés ci-dessus.

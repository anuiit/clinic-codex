"""Per-class readout of frozen nested runs; no inference, fitting or activation."""
import csv, hashlib, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
INPUTS = [REPORTS / 'nested-rejection-results-20261004.json',
          REPORTS / 'nested-branch-rejection-results-20261004.json',
          ROOT / 'backend/training_corpus/snapshots/snapshot-611c734590f272060251/metadata.csv',
          ROOT / 'backend/training_corpus/snapshots/snapshot-471c74f801b749a9ad6e/snapshot_manifest.json',
          ROOT / 'backend/training_corpus/snapshots/snapshot-611c734590f272060251/snapshot_manifest.json',
          ROOT / 'backend/training_corpus/evaluation/business-final-ea447440c1ae1d12e7b4/test_manifest.json']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in INPUTS}
nested, branch = [json.loads(p.read_text(encoding='utf-8')) for p in INPUTS[:2]]
with INPUTS[2].open(encoding='utf-8-sig', newline='') as stream:
    corpus = list(csv.DictReader(stream))
names = {int(r['class_label']): r['element_name'] for r in corpus}
train_counts = Counter(int(r['class_label']) for r in corpus if r['dataset_split'] == 'train')
all_counts = Counter(int(r['class_label']) for r in corpus)
seeds = [42, 43, 44]
policies = ['historic', 'adapted_035', 'branch']
counts = {m: {s: defaultdict(Counter) for s in seeds} for m in policies}
page_sets, fit_support = defaultdict(set), defaultdict(set)
confusions = Counter()
objects = {}
outer = {(p['group'], p['seed']): p for p in nested['outer_pages']}

for p in branch['outer_pages']:
    seed, group, gt = p['seed'], p['group'], p['gt']
    n = outer[group, seed]
    assert gt == n['gt']
    variants = [('historic', n['methods']['baseline'], 'current_035'),
                ('adapted_035', n['methods']['gated'], 'current_035'),
                ('branch', p['methods']['gated'], 'rare_branch')]
    for g in gt:
        label = g['label']
        assert names[label] == g['class_name']
        page_sets[label].add(group)
        fit_support[label].add(g['natural_support'])
        key = (group, g['row_id'])
        assert key not in objects or objects[key] == label
        objects[key] = label
    for method, view, policy in variants:
        score = view['evaluations'][policy]
        predictions = {z['proposal_index']: z for z in view['predictions']}
        matched = {z['gt_index']: z for z in score['matches']}
        correct_proposals = {z['proposal_index'] for z in matched.values() if z['correct']}
        matched_ids = {z['proposal_index'] for z in matched.values()}
        scored_ids = matched_ids | set(score['unmatched_proposal_indices'])
        assert len(scored_ids) == score['counts']['scored_predictions']
        assert not scored_ids & set(score['ignored_proposal_indices'])
        assert len(correct_proposals) == score['counts']['correct']
        assert len(matched_ids) == len(matched) == score['counts']['matched']
        for i, g in enumerate(gt):
            c = counts[method][seed][g['label']]
            c['known'] += 1
            if i in matched:
                c['matched'] += 1
                c['correct'] += int(matched[i]['correct'])
                if method == 'adapted_035' and not matched[i]['correct']:
                    pred = predictions[matched[i]['proposal_index']]['prediction']
                    confusions[g['label'], pred] += 1
        for idx in scored_ids:
            z = predictions[idx]
            assert z['prediction'] in names
            c = counts[method][seed][z['prediction']]
            c['predicted'] += 1
            c['true_positive'] += int(idx in correct_proposals)
            c['unmatched_output'] += int(idx not in matched_ids)
    before = {m['gt_index'] for m in n['methods']['gated']['evaluations']['current_035']['matches'] if m['correct']}
    after = {m['gt_index'] for m in p['methods']['gated']['evaluations']['rare_branch']['matches'] if m['correct']}
    assert not after - before, 'Error decomposition requires no newly correct rematches in this run'

for method in policies:
    agg = (nested['aggregate']['baseline']['current_035'] if method == 'historic'
           else nested['aggregate']['gated']['current_035'] if method == 'adapted_035'
           else branch['aggregate']['rare_branch'])
    for seed in seeds:
        cc = counts[method][seed]
        expected = agg['seeds'][str(seed)]['counts']
        for out_key, in_key in [('known', 'known'), ('matched', 'matched'),
                                ('correct', 'correct'), ('predicted', 'scored_predictions'),
                                ('unmatched_output', 'unmatched_predictions')]:
            assert sum(c[out_key] for c in cc.values()) == expected[in_key], (method, seed, out_key)
        assert sum(c['true_positive'] for c in cc.values()) == expected['correct']
assert len(objects) == 343 and len(outer) == 51 and len(names) == 286

mean_count = lambda m, label, k: statistics.mean(counts[m][s][label][k] for s in seeds)
ratio = lambda a, b: a / b if b else None
rows = []
for label, name in sorted(names.items()):
    known = counts['branch'][42][label]['known']
    assert all(counts['branch'][s][label]['known'] == known for s in seeds)
    row = {'label': label, 'class_name': name, 'corpus_crops': all_counts[label],
           'train_crops': train_counts[label], 'evaluated_objects': known,
           'evaluated_pages': len(page_sets[label]),
           'fit_support_min': min(fit_support[label]) if fit_support[label] else None,
           'fit_support_max': max(fit_support[label]) if fit_support[label] else None}
    for method in policies:
        correct = mean_count(method, label, 'correct')
        matched = mean_count(method, label, 'matched')
        pred = mean_count(method, label, 'predicted')
        row.update({method + '_correct_mean': correct,
                    method + '_matched_mean': matched,
                    method + '_outputs_mean': pred,
                    method + '_recall': ratio(correct, known),
                    method + '_precision': ratio(correct, pred),
                    method + '_naming_given_matched': ratio(correct, matched)})
    row['sam_coverage'] = ratio(row['adapted_035_matched_mean'], known)
    row['sam_missing'] = known - row['adapted_035_matched_mean']
    row['wrong_name_before_rejection_mean'] = row['adapted_035_matched_mean'] - row['adapted_035_correct_mean']
    row['correct_rejected_mean'] = row['adapted_035_correct_mean'] - row['branch_correct_mean']
    assert abs(row['branch_correct_mean'] + row['sam_missing'] + row['wrong_name_before_rejection_mean'] + row['correct_rejected_mean'] - known) < 1e-9
    row['branch_recall_seed_min'] = min(counts['branch'][s][label]['correct'] / known for s in seeds) if known else None
    row['branch_recall_seed_max'] = max(counts['branch'][s][label]['correct'] / known for s in seeds) if known else None
    rows.append(row)

def band(x):
    assert 0 <= x <= 1
    return '>=90%' if x >= .9 - 1e-12 else '70-<90%' if x >= .7 - 1e-12 else '20-<70%' if x >= .2 - 1e-12 else '<20%'
assert [band(v) for v in [0, .199, .2, .699, .7, .899, .9, 1]] == ['<20%', '<20%', '20-<70%', '20-<70%', '70-<90%', '70-<90%', '>=90%', '>=90%']
bands = {}
for metric in ['branch_recall', 'adapted_035_naming_given_matched']:
    bands[metric] = {}
    for minimum in [1, 5, 10]:
        eligible = [r for r in rows if r['evaluated_objects'] >= minimum and r[metric] is not None]
        bands[metric][str(minimum)] = dict(Counter(band(r[metric]) for r in eligible))
summary = {'classes': len(rows), 'classes_observed': sum(r['evaluated_objects'] > 0 for r in rows),
           'classes_unobserved': sum(r['evaluated_objects'] == 0 for r in rows),
           'objects': len(objects), 'pages': len({p['group'] for p in branch['outer_pages']}),
           'support_histogram': dict(Counter(r['evaluated_objects'] for r in rows if r['evaluated_objects'])),
           'bands': bands,
           'errors_mean': {k: sum(r[k] for r in rows) for k in ['branch_correct_mean', 'sam_missing', 'wrong_name_before_rejection_mean', 'correct_rejected_mean']},
           'outputs_mean': sum(r['branch_outputs_mean'] for r in rows),
           'macro_branch_recall_observed': statistics.mean(r['branch_recall'] for r in rows if r['evaluated_objects']),
           'train_crops_quantiles': {'min': min(train_counts.values()), 'median': statistics.median(train_counts.values()), 'max': max(train_counts.values())}}
analysis = {'schema_version': 'per-class-readout.v1', 'sources_sha256': input_hashes,
            'script_sha256': sha(Path(__file__)), 'new_training': False, 'activated': False,
            'method': 'Frozen OOF nested heads on 17 previously examined pages. Class metrics are ratios of mean counts over seeds 42/43/44. Repeated seeds do not add independent objects. Null recall means no class ground truth, not zero accuracy.',
            'summary': summary, 'classes': rows,
            'bbox_inventory_audit': {'parent_crop_rows': 9266, 'parent_bbox_rows': 7,
              'imported_unique_known_bbox_rows': 450, 'current_bbox_rows': 457,
              'current_bbox_split_counts': {'train': 343, 'dev': 43, 'locked_test': 71},
              'legacy_external_rows_without_local_bbox': 9259,
              'older_business_test': {'rows': 303, 'classes': 87, 'pages': 10,
                'pages_overlapping_download_archive_by_pixel_hash': 10,
                'rows_overlapping_by_page_pixels_class_bbox': 301,
                'two_row_differences_unresolved': True,
                'new_independent_lot': False}},
            'confusions_before_rejection': [{'truth': names[a], 'predicted': names[b], 'objects_mean': n / 3} for (a, b), n in confusions.most_common()]}
assert all(sha(p) == input_hashes[str(p.relative_to(ROOT))] for p in INPUTS)
(REPORTS / 'per-class-analysis-20261004.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
with (REPORTS / 'per-class-results-20261004.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(json.dumps(summary, ensure_ascii=False, indent=2))
for title, cohort in [('best support>=5', sorted([r for r in rows if r['evaluated_objects'] >= 5], key=lambda r: r['branch_recall'], reverse=True)), ('worst support>=5', sorted([r for r in rows if r['evaluated_objects'] >= 5], key=lambda r: r['branch_recall']))]:
    print(title)
    for r in cohort[:8]: print({k:r[k] for k in ['class_name', 'train_crops', 'evaluated_objects', 'evaluated_pages', 'sam_coverage', 'adapted_035_naming_given_matched', 'branch_precision', 'branch_recall']})
print('confusions', analysis['confusions_before_rejection'][:12])

# A compact scientific figure shows causes, not just the final percentage.
from reportlab.graphics.shapes import Drawing, String, Rect, Line
from reportlab.graphics import renderSVG, renderPDF
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
pdfmetrics.registerFont(TTFont('Arial', r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold', r'C:\Windows\Fonts\arialbd.ttf'))
figure = Drawing(1120, 735)
figure.add(Rect(0, 0, 1120, 735, fillColor=HexColor('#ffffff'), strokeColor=None))
def text_at(x, y, text, size=12, bold=False, color='#243446', anchor='start'):
    figure.add(String(x, y, text, fontName='ArialBold' if bold else 'Arial', fontSize=size,
                      fillColor=HexColor(color), textAnchor=anchor))
text_at(30, 699, 'La localisation et le nom échouent sur des classes différentes', 22, True)
text_at(30, 674, '16 classes avec au moins 5 glyphes évalués • 17 planches • moyenne de 3 graines, sans nouveaux exemples', 13)
parts = [('branch_correct_mean', 'Boîte + nom corrects', '#257c66'),
         ('correct_rejected_mean', 'Correct avant rejet', '#db983e'),
         ('wrong_name_before_rejection_mean', 'Bonne boîte, mauvais nom', '#8a6dba'),
         ('sam_missing', 'Pas de boîte SAM à IoU ≥ 0,5', '#b5bec7')]
for x, (_, label, color) in zip([30, 260, 490, 800], parts):
    figure.add(Rect(x, 638, 13, 13, fillColor=HexColor(color), strokeColor=None)); text_at(x+20, 638, label, 12)
start, width, first_y, step = 280, 650, 582, 29
text_at(30, 615, 'Classe', 12, True); text_at(190, 615, 'Glyphes / pages', 12, True)
text_at(1000, 615, 'Rappel final', 12, True, anchor='middle')
for tick in [0, 25, 50, 75, 100]:
    xx = start + width * tick / 100
    figure.add(Line(xx, 120, xx, 603, strokeColor=HexColor('#e7ebef'), strokeWidth=.7))
    text_at(xx, 600, f'{tick} %', 10, anchor='middle')
cohort = sorted([r for r in rows if r['evaluated_objects'] >= 5], key=lambda r: (-r['branch_recall'], -r['evaluated_objects']))
for i, r in enumerate(cohort):
    yy = first_y - i * step
    text_at(30, yy, r['class_name'], 13, True)
    text_at(230, yy, f"{r['evaluated_objects']} / {r['evaluated_pages']}", 12, anchor='middle')
    xx = start
    for k, _, color in parts:
        ww = width * r[k] / r['evaluated_objects']
        figure.add(Rect(xx, yy-3, ww, 18, fillColor=HexColor(color), strokeColor=None)); xx += ww
    text_at(1000, yy, f"{100*r['branch_recall']:.1f} %", 13, True, anchor='middle')
text_at(30, 78, 'Gris : absent ou boîte trop différente. Violet : erreur de nom sur une boîte appariée avant filtrage.', 12)
text_at(30, 57, 'Orange : bonne réponse supprimée par le filtre. Vert : résultat final du candidat, encore non activé.', 12)
text_at(30, 32, 'Lot déjà examiné : résultats exploratoires. Les 83 autres classes observées ont moins de 5 exemples ; 187 classes sont absentes.', 11)
renderSVG.drawToFile(figure, str(REPORTS / 'per-class-errors-20261004.svg'))
pdf = pdfium.PdfDocument(renderPDF.drawToString(figure))
pdf[0].render(scale=1.6).to_pil().save(REPORTS / 'per-class-errors-20261004.png')

fmt = lambda v: '—' if v is None else f'{100*v:.1f} %'
band_order = ['>=90%', '70-<90%', '20-<70%', '<20%']
band_labels = ['≥ 90 %', '70 à < 90 %', '20 à < 70 %', '< 20 %']
band_table = '| Rappel boîte + nom | Classes observées (99) | Avec ≥5 exemples (16) |\n|---|---:|---:|\n'
for k, title in zip(band_order, band_labels):
    band_table += f"| {title} | {bands['branch_recall']['1'].get(k,0)} | {bands['branch_recall']['5'].get(k,0)} |\n"
class_table = '| Classe | Crops train* | Glyphes évalués | Pages | Couverture SAM | Bon nom si bonne boîte, avant rejet | Précision finale | Rappel final |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for r in cohort:
    class_table += f"| {r['class_name']} | {r['train_crops']} | {r['evaluated_objects']} | {r['evaluated_pages']} | {fmt(r['sam_coverage'])} | {fmt(r['adapted_035_naming_given_matched'])} | {fmt(r['branch_precision'])} | {fmt(r['branch_recall'])} |\n"
conf_table = '| Vrai nom | Nom prédit | Glyphes confondus, moyenne des graines |\n|---|---|---:|\n'
for c in analysis['confusions_before_rejection'][:12]:
    conf_table += f"| {c['truth']} | {c['predicted']} | {c['objects_mean']:.2f} |\n"
report = [
'# Données, erreurs par classe et prochaines pistes — 4 octobre 2026',
'**Conclusion.** Les difficultés ne se résument pas au nombre de boîtes métier. Il y a une supervision de reconnaissance dans le corpus ancien, une supervision de localisation sur peu de planches, des erreurs de localisation dépendant de la classe, des confusions de noms et des réponses correctes perdues par le filtre. Le bilan par classe donne des priorités différentes. Aucun entraînement ni aucune activation dans ce readout.',
'**Données et unité de compte.** Le snapshot actuel contient 9716 crops uniques avec un label parmi 286 classes, dont 9471 dans sa partition train. Parmi eux, 450 crops métier connus ont été importés après filtrage des doublons et conflits : 343 train, 43 dev, 64 test. Ces nombres comptent des objets, pas des planches. Le readout actuel couvre les 343 objets connus de 17 planches train. Le corpus parent comporte 9266 crops, dont 7 anciennes annotations live avec bbox, sur une page réservée au test. Les 9259 autres crops legacy/external n’ont pas de bbox dans le snapshot ni dans les deux manifestes sources locaux. Le snapshot enrichi compte donc 457 bbox, dont 343 train, 43 dev et 71 test. Des coordonnées pourraient exister dans une copie originale externe ; leur absence dans ces manifestes ne prouve pas leur absence partout. Un ancien manifeste business-final conserve 303 bbox sur 10 pages : les 10 pages sont dans la même archive métier, et 301 lignes correspondent exactement par empreinte des pixels de page, classe et boîte. Deux différences de lignes restent inexpliquées. Il ne constitue pas 303 objets sur de nouvelles pages indépendantes et ces pages ont déjà été évaluées. Ce lot ne doit donc pas être additionné comme nouvelle supervision indépendante ni utilisé comme nouvelle validation aveugle.',
'**Distribution des crops.** La partition train du snapshot va de 2 à 138 crops par classe ; médiane 28. 18 classes ont au plus 4 crops, 43 au plus 10. Ces comptes précèdent les exclusions par fold : ils ne sont pas les supports réellement vus par chaque head. Les CSV publient également les supports de pixels uniques du fit pour les classes évaluées. Dans ces expériences, DINO est figé et la projection de reconnaissance est adaptée ; 10000 crops ne signifient donc pas qu’un nouveau backbone a été appris de bout en bout.',
'**Périmètre et méthode.** On relit les prédictions gelées des 15 heads nested (5 partitions ×3 graines) : apprentissage, calibration et pages externes séparés pour cette adaptation. On compare historique à 0,35, candidat à 0,35 et candidat avec rejet par branche historique/adaptée. Les 17 planches ont déjà été consultées dans les essais précédents ; la provenance du préentraînement historique reste inconnue. Les graines répètent les mêmes 343 objets et ne créent pas 1029 observations indépendantes. Aucun dev/test relu ici.',
'**Définition des colonnes.** Couverture SAM = glyphes ayant une boîte appariée IoU≥0,5 /tous les glyphes de la classe, avant rejet. Bon nom si bonne boîte = noms corrects /boîtes appariées, avant rejet : cela mesure la reconnaissance conditionnelle, pas le pipeline complet. Précision finale = bonnes sorties /toutes les sorties évaluables prédites dans cette classe, y compris les fragments, les mauvais noms et les boîtes non appariées ; les régions ignorées restent exclues. Rappel final = bonnes boîtes avec bon nom /tous les vrais glyphes de la classe, manqués et rejetés compris. Les métriques par classe sont les rapports des comptes moyens des trois graines, pas une moyenne de pages.',
'**Résultats globaux.** Le candidat par branche conserve les métriques déjà publiées, moyennes micro par graine : précision exacte 39,87 %, rappel exact 45,38 %, F1 exact 42,42 %. Avant ce rejet supplémentaire, le même candidat reconnaît correctement 70,21 % des 254 glyphes disposant d’une bonne boîte SAM. C’est une précision de nom conditionnelle, qui n’inclut pas les 89 glyphes sans boîte ni les propositions non appariées. Le rappel macro final sur les 99 classes observées est 30,18 %, inférieur au rappel micro de 45,38 % : les classes fréquentes tirent le score micro vers le haut.',
'**Votre hypothèse, mesurée.** Sur 286 classes, 99 sont présentes dans les pages évaluées, et 187 n’ont aucun exemple permettant d’estimer leur rappel. Parmi les 99, 46 ont un seul exemple et 20 seulement deux. Les 18 classes à ≥90 % de rappel ont toutes entre 1 et 4 objets : 12 en ont un, 3 en ont deux, 2 en ont trois, 1 en a quatre. Aucun groupe de classes richement évalué à ≥90 % n’est démontré.',
band_table,
'**La reconnaissance seule est meilleure pour certaines classes.** Sur les 74 classes ayant au moins une boîte SAM appariée, 36 dépassent 90 % de reconnaissance conditionnelle avant rejet ; les 25 autres classes observées n’ont aucune bonne boîte SAM et leur reconnaissance conditionnelle est indéfinie. Parmi les 16 classes avec au moins 5 glyphes, `pantli`, `tecpatl` et `tepetl` atteignent 100 % sur les boîtes appariées, mais leur rappel final reste inférieur à cause de la couverture et du rejet. Ces petits sous-lots ne garantissent pas 100 % sur de nouvelles planches.',
'![Causes d’erreurs des classes ayant au moins cinq objets](per-class-errors-20261004.png)',
class_table,
'*Crops train : partition train du snapshot avant exclusion OOF/calibration. Les 16 classes ci-dessus ont au moins cinq objets ; ce seuil de présentation ne suffit pas à obtenir des estimations précises. Les 286 lignes complètes, y compris les classes absentes, sont dans le CSV. Une précision de 0 pour une classe sans vérité terrain mais avec des sorties signifie aucune sortie correcte sur ce lot ; son rappel reste indéfini.',
'**Décomposition des 343 objets, moyenne des graines.** 155,67 bonnes boîtes avec bon nom conservées ; 89 sans boîte SAM suffisamment proche ; 75,67 avec bonne boîte mais mauvais nom avant rejet ; 22,67 correctement reconnus puis rejetés. Cette décomposition totalise 343 et ne double-compte pas les objets. Parmi les 391 sorties évaluables conservées en moyenne, 235,33 sont incorrectes : le faible nombre de bonnes propositions et les fausses propositions restent tous deux importants.',
'**Confusions de noms avant filtrage, seulement sur des boîtes appariées.** Ces paires indiquent les exemples à examiner visuellement ; elles ne prouvent pas qu’une définition textuelle suffira à les séparer.',
conf_table,
'**Détecteur déjà essayé.** Le pilote précédent utilisait un Faster R-CNN MobileNetV3 préentraîné, une seule classe « glyphe », avec backbone figé. Il améliorait la couverture sur ses pages train (86,92 % contre 74,13 % pour SAM), mais régressait sur son unique page dev (41,86 % contre 48,84 %) et ses deux pages test alors examinées (56,34 % contre 71,83 %). Ces résultats proviennent du rapport gelé du 3 octobre, avec son périmètre géométrique initial, et ne sont pas une nouvelle validation ni une comparaison nested du candidat actuel. Aucun modèle activé. Ce pilote ne démontre pas qu’un simple changement d’architecture vers YOLO corrigerait le problème ; il ne permet pas non plus d’attribuer à lui seul la régression à la diversité des données ou à une mémorisation des pages.',
'**Priorité proposée : utiliser les crops pour enrichir la détection.** Récupérer d’abord les coordonnées originales si une source les conserve. À défaut, composer des planches synthétiques en plaçant des crops détourés sur des fonds plausibles : les positions de collage produisent des boîtes connues sans nouvelle annotation manuelle. Mélanger synthétique et vraies pages, puis comparer « vraies seules » et « vraies+synthétiques » avec un détecteur préentraîné à une classe. Éviter les bords rectangulaires et les fonds artificiels que le réseau pourrait apprendre. Séparer les groupes de provenance, pixels et composants avant de composer les planches, puis exclure de la synthèse tout ce qui appartient aux partitions de calibration et d’évaluation. Garder les mêmes partitions et la même recette pour les deux bras réel seul /réel+synthétique, sans régler cette recette sur les pages dev/test déjà consultées. Ce ne sont ni des boîtes originales récupérées ni de nouvelles apparences indépendantes. La méthode a une base expérimentale dans [Cut, Paste and Learn](https://arxiv.org/abs/1708.01642), mais son gain sur ces glyphes reste à mesurer. Le [format de détection YOLO](https://docs.ultralytics.com/datasets/detect) prévoit un label et une boîte par objet : 400 boîtes et 400 images ne représentent pas le même volume.',
'**Descriptions et embeddings de texte : seconde piste ciblée.** Le rapprochement image/texte est pertinent si l’on apprend un espace commun ; [CLIP](https://arxiv.org/abs/2103.00020) est un exemple de préentraînement par appariement image–texte. Ajouter directement des vecteurs de dictionnaire aux features DINO n’assure pas cet alignement. Je comparerais références visuelles seules, références+nom traduit, et références+description visuelle vérifiée, avec les mêmes partitions et les mêmes boîtes. Une définition du sens et une description de la forme dessinée n’apportent pas forcément la même information. La piste peut aider les confusions de noms, mais ne récupère pas à elle seule les 89 glyphes manqués par SAM. [GLIP](https://arxiv.org/abs/2112.03857) combine détection et langage ; ses résultats génériques ne démontrent pas un transfert à notre iconographie.',
'**Modèle en amont.** Un classifieur de crops existe déjà dans le pipeline DINO→projection→prototypes. Un filtre « proposition valide de glyphe /fragment ou fond » peut améliorer la précision des sorties SAM à partir des pages exhaustivement annotées, mais ne crée aucune proposition pour les glyphes manqués. Un classifieur de la planche entière ne fournit pas les positions des nombreux glyphes. Le détecteur à une classe puis le classifieur à 286 noms réutilisent les deux niveaux de supervision existants.',
'**Décision.** Priorité au déficit de localisation et aux propositions parasites avec le corpus disponible, puis à un essai image/texte sur les confusions réellement observées. Le suivi doit publier couverture, précision et rappel par classe avec leurs supports. Le lot actuel permet des comparaisons exploratoires ; sans nouvelles pages, il ne permet pas une confirmation aveugle de généralisation.',
'**Vérifications.** Totaux par classe reconstruits et identiques aux trois agrégats par graine ; identités de pages/GT contrôlées ; régions ignorées exclues ; décomposition sans nouveaux succès après rematching vérifiée ; bornes des bandes testées ; hashes des deux JSON et de metadata inchangés. Revue native indépendante : aucune erreur matérielle trouvée dans les métriques ; l’ordre proposé est partagé avec réserves sur la séparation avant synthèse et le caractère exploratoire du détecteur. Aucun checkpoint, seuil ou fichier actif modifié.',
'[Table complète CSV](per-class-results-20261004.csv). [Comptes et provenance JSON](per-class-analysis-20261004.json). [Résultats nested](nested-rejection-research-20261004.md). [Pilote détecteur gelé](bbox-detector-results-20261003.json). Audit des bbox : `backend/training_corpus/snapshots/snapshot-471c74f801b749a9ad6e/snapshot_manifest.json`, `backend/training_corpus/snapshots/snapshot-611c734590f272060251/snapshot_manifest.json` et `backend/training_corpus/evaluation/business-final-ea447440c1ae1d12e7b4/test_manifest.json`.'
]
(REPORTS / 'per-class-research-20261004.md').write_text('\n\n'.join(report)+'\n', encoding='utf-8')
assert all(sha(p) == input_hashes[str(p.relative_to(ROOT))] for p in INPUTS)
print('Report, CSV, JSON, SVG and PNG created; frozen inputs unchanged.')

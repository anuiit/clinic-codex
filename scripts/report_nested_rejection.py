"""Posthoc nested-rejection report and scientific readout figure."""
import hashlib,json,statistics
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String,Rect
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'reports/nested-rejection-results-20261004.json'
d=json.loads(PATH.read_text(encoding='utf-8'));agg=d['aggregate'];seeds=[42,43,44]
METHODS={'baseline':'Historique','paired':'Paires + conservation','gated':'Paires + recours rare'}
POLICIES={'current_035':'Seuil 0,35','calibrated':'F1 sur calibration','rare_guarded':'F1 avec garde rares','rare_branch':'Branche historique à 0,35'}
BRANCH_PATH=ROOT/'reports/nested-branch-rejection-results-20261004.json'
branch=json.loads(BRANCH_PATH.read_text(encoding='utf-8'))
assert branch['optimizer_updates']==0 and not branch['activated'] and all(branch['verification'].values())
agg['gated']['rare_branch']=branch['aggregate']['rare_branch']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path,digest in (d['protocol']['input_sha256']|branch['protocol']['input_sha256']).items():
 p=ROOT/path.removeprefix('/home/sina/omxpro/clinic-codex/') if path.startswith('/home/sina/omxpro/clinic-codex/') else Path('//wsl.localhost/Ubuntu-24.04'+path)
 assert sha(p)==digest,path
assert not d['activated'] and not d['locked_test_scored'] and d['optimizer_updates']==15000 and all(d['verification'].values())
cs=lambda m,p:[agg[m][p]['seeds'][str(s)]['counts'] for s in seeds]
seq=lambda m,p,k:' / '.join(str(z[k]) for z in cs(m,p))
refseq=lambda m,p:' / '.join(str(agg[m][p]['seeds'][str(s)]['reference_rare']['correct']) for s in seeds)
mean=lambda m,p:agg[m][p]['mean_metrics']
thresholds=lambda m,p:([e['selection']['threshold'] for e in branch['folds']] if p=='rare_branch' else [e['selections'][m][p]['threshold'] for e in d['folds']])
b=mean('gated','current_035');a=mean('gated','calibrated');g=mean('gated','rare_guarded')
comparison={}
for m in METHODS:
 comparison[m]={}
 for p in [z for z in agg[m] if z!='current_035']:
  comparison[m][p]=dict(lost_correct=[agg[m][p]['seeds'][str(s)]['lost_correct'] for s in seeds],gained_correct=[agg[m][p]['seeds'][str(s)]['gained_correct'] for s in seeds],incorrect_outputs_removed=[(x['scored_predictions']-x['correct'])-(y['scored_predictions']-y['correct']) for x,y in zip(cs(m,'current_035'),cs(m,p))])
metrics_table='| Variante | Rejet | Correctes /343, graines 42 /43 /44 | Précision exacte | Rappel exact | F1 exact | Rares du fit : correctes /total | Cohorte précédente /10 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for m,name in METHODS.items():
 for p,label in POLICIES.items():
  if p not in agg[m]:continue
  z=mean(m,p)
  metrics_table+=f"| {name} | {label} | {seq(m,p,'correct')} | {100*z['exact_precision']:.2f} % | {100*z['exact_recall']:.2f} % | {100*z['exact_f1']:.2f} % | {seq(m,p,'rare_correct')} / {cs(m,p)[0]['rare_known']} | {refseq(m,p)} |\n"
selections_table='| Partition | Graine | Pages fit /cal /externes | Rares cal | Seuil F1 candidat | Seuil gardé | Seuil branche adaptée | Motif garde |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
for e in d['folds']:
 split=next(z for z in d['protocol']['splits'] if z['fold']==e['fold']);v=e['selections']['gated'];bs=next(z for z in branch['folds'] if z['fold']==e['fold'] and z['seed']==e['seed'])
 selections_table+=f"| {e['fold']} | {e['seed']} | {len(split['fit_pages'])} / {len(split['calibration'])} / {len(split['held'])} | {v['calibrated']['calibration_rare_objects']} | {v['calibrated']['threshold']:.2f} | {v['rare_guarded']['threshold']:.2f} | {bs['selection']['threshold']:.2f} | {v['rare_guarded']['reason']} |\n"
counts_table='| Rejet du candidat | Acceptées /654 | Ignorées | Évaluées | Bonnes sorties perdues | Bonnes sorties gagnées | Sorties incorrectes supprimées, bilan net |\n|---|---:|---:|---:|---:|---:|---:|\n'
for p,label in POLICIES.items():
 changes=comparison['gated'].get(p,{k:[0,0,0] for k in ['lost_correct','gained_correct','incorrect_outputs_removed']})
 counts_table+=f"| {label} | {seq('gated',p,'accepted')} | {seq('gated',p,'ignored_predictions')} | {seq('gated',p,'scored_predictions')} | "+' | '.join(' / '.join(str(x) for x in changes[k]) for k in ['lost_correct','gained_correct','incorrect_outputs_removed'])+' |\n'
pages_table='| Rejet du candidat | Précision moyenne par page | Rappel moyen par page | F1 moyen par page | Macro rare du fit | Macro cohorte précédente |\n|---|---:|---:|---:|---:|---:|\n'
for p,label in POLICIES.items():
 pages_table+=f"| {label} | "+' | '.join(f"{100*agg['gated'][p]['mean_page'][k]:.2f} %" for k in ['exact_precision','exact_recall','exact_f1'])+f" | {100*mean('gated',p)['rare_class_macro']:.2f} % | {100*mean('gated',p)['reference_rare_class_macro']:.2f} % |\n"
guards=d['guards']|{'rare_branch':branch['guard']}
rare_preserved=lambda p:all(guards[p][k] for k in ['rare_correct_no_regression','rare_class_macro_no_regression','reference_rare_no_regression'])
v=mean('gated','rare_branch')
opinion='La branche historique gardée à 0,35 fournit le compromis le plus intéressant de cette série : meilleure précision et meilleur F1 que le même head sans rejet supplémentaire, avec maintien des huit rares observés. Mais le rappel baisse et plus de la moitié des sorties évaluées restent incorrectes. Je recommande de conserver cette variante comme candidat de recherche, sans activation. Le routage dépend d’une classe prédite par l’historique, pas de la rareté réelle : les huit rares conservés sont un résultat mesuré sur ce lot, pas une garantie.'
lines=[
 '# Apprentissage, calibration et évaluation sur pages distinctes — 4 octobre 2026',
 f"**Résultat principal.** Pour le candidat nested, la calibration F1 fait passer la précision exacte de {100*b['exact_precision']:.2f} % à {100*a['exact_precision']:.2f} %, le rappel de {100*b['exact_recall']:.2f} % à {100*a['exact_recall']:.2f} % et le F1 de {100*b['exact_f1']:.2f} % à {100*a['exact_f1']:.2f} %. Correctes : {seq('gated','current_035','correct')} → {seq('gated','calibrated','correct')} /343. Cohorte rare précédente : {refseq('gated','current_035')} → {refseq('gated','calibrated')} /10. Aucun modèle activé.",
 '**Ce qui a changé.** Chaque partition externe contient trois ou quatre pages réservées à l’évaluation. Dans les 13 ou 14 pages restantes, trois pages sont choisies pour la calibration par un tirage déterministe fondé seulement sur les identifiants de page. La fermeture des composants est recalculée sur union des pages externes et de calibration ; tous leurs groupes et pixels sont exclus de l’apprentissage raw et SAM. Il reste dix ou onze pages métier pour apprendre les poids, avec le corpus parent. Les mêmes trois pages servent pour les trois graines d’une partition. Tous les 286 labels restent représentés dans chaque fit.',
 '```mermaid\nflowchart LR\n  A[Train de la partition externe] --> B[10 ou 11 pages métier + corpus parent]\n  A --> C[3 pages de calibration]\n  B --> D[Projection apprise : 1000 updates]\n  C --> E[Seuil fixé sur F1 ou garde rares]\n  D --> E\n  E --> F[3 ou 4 pages externes : une évaluation]\n```',
 '**Poids et coût.** 15 nouveaux heads : cinq partitions × trois graines × 1000 updates fixes. Initialisation historique, prototypes historiques figés, DINO et SAM figés, mêmes pertes équilibrées CE + KL de conservation + paire raw/SAM et même recette AdamW que le candidat précédent. Le helper de fit existant est réutilisé sans modifier sa source ; seuls son dossier de sortie et ses milestones sont redirigés en mémoire, vers le nouveau run et la seule étape 1000. Son argument held est le lot de calibration, pas les pages externes. Le choix de checkpoint n’est pas optimisé.',
 '**Seuils fixés avant lecture externe.** Grille 0,35–0,95 par pas de 0,05 ; seuil maximisant le F1 exact moyen par page de calibration, égalités à 1e-12 vers le plus bas. Résultat principal : choix F1 sans contrainte. Résultat secondaire préenregistré : seulement les seuils ne diminuant ni le nombre de rares corrects ni la macro des classes rares en calibration, puis même maximisation F1. Sans objet rare dans les trois pages de calibration, cette garde se replie explicitement sur 0,35. Toutes les sélections du head sont enregistrées avant son premier calcul sur les pages externes.',
 '**Référence.** L’effet du rejet est comparé à 0,35 avec les mêmes nouveaux poids, qui n’ont vu ni les pages de calibration ni les pages externes pendant cette adaptation. Les anciens heads OOF avaient vu les pages désormais réservées à la calibration : leurs métriques antérieures sont un contexte, pas le contrôle de ce test. L’historique est également relu comme contrôle sans nouvelle optimisation ; ses prédictions externes sont reproduites.',
 '**Résultats externes.** Moyennes des métriques micro des trois graines ; chaque graine couvre une fois les mêmes 17 pages et 343 objets connus. Bonnes sorties = boîte IoU ≥ 0,5 et bon nom ; précision = bonnes sorties /sorties acceptées évaluables ; rappel = bonnes sorties /343. Après chaque rejet, les boîtes sont rematchées en géométrie avant contrôle des noms. Les inconnus/conflits sont ignorés comme précédemment, et leur nombre accepté peut diminuer.',
 metrics_table,
 '![Effet de la calibration sur pages distinctes](nested-rejection-20261004.png)',
 f"**Extension adaptative, sans nouveaux fits.** Après constat du manque de rares en calibration, un readout complémentaire est préenregistré dans un nouveau run. Les sorties que le gating existant dirige vers l’historique gardent le seuil 0,35 ; le seuil des autres sorties est choisi sur les trois mêmes pages de calibration indépendantes des poids. La décision fallback repose sur le top-1 historique prédit et ses supports raw de fit, jamais sur la vérité terrain externe. Le matching et les dénominateurs sont recalculés. Cette extension est rétrospective, distincte du résultat primaire, et ne change aucune source, aucun checkpoint ni aucun résultat précédent. Elle atteint {100*v['exact_precision']:.2f} % de précision, {100*v['exact_recall']:.2f} % de rappel, {100*v['exact_f1']:.2f} % de F1 et {refseq('gated','rare_branch')} rares corrects /10.",
 '**Comptes du candidat.** Les erreurs supprimées sont le bilan net des propositions non appariées et des mauvais noms ; les régions ignorées restent hors dénominateur. Le rejet ne récupère pas les objets sans proposition SAM correspondante.',
 counts_table,
 '**Moyennes par page et rares.** Chaque page a un poids égal dans l’objectif de calibration ; les métriques micro pondèrent les volumes. Le seuil primaire et le secondaire sont présentés séparément, sans sélectionner une politique après examen de ses pages externes.',
 pages_table,
 '**Deux cohortes rares.** Gating et garde utilisent de un à quatre crops raw de pixels uniques dans le fit, après exclusion externe et calibration. Exclure trois pages supplémentaires peut rendre d’autres classes rares : le dénominateur peut donc changer. Pour comparer les recherches, on publie séparément les dix objets de la cohorte définie avec le train externe avant retrait de la calibration. Dans ce run, les deux cohortes contiennent les mêmes dix objets. Ce second compteur ne choisit aucun seuil et ne définit pas le gating.',
 '**Choix et couverture en calibration.** La garde ne peut préserver que les exemples rares effectivement présents et correctement reconnus à 0,35 dans les trois pages de calibration. Un succès sur ce sous-lot ne garantit pas le résultat externe. Les comptes et le motif de repli rendent cette limite visible.',
 selections_table,
 '**Ce que la garde scalaire démontre.** Les partitions 0–3 n’ont aucun rare en calibration : 12/15 heads restent à 0,35. La partition 4 contient six rares en calibration, mais son lot externe contient un seul rare, déjà incorrect à 0,35 et encore incorrect après rejet. Les neuf autres rares externes sont dans la partition 0, où la garde ne filtre pas. Le passage des contrôles rares de cette garde scalaire s’explique donc par son repli ; il ne valide pas un seuil filtrant qui généralise aux rares. Le readout par branche, lui, applique un seuil à la branche adaptée dans les cinq partitions et sa conservation des rares est mesurée séparément.',
 '**Mon avis.** '+opinion+' Une garde passant ces contrôles ne vaut pas autorisation de déploiement. Avec seulement 17 pages, la couverture des classes rares en calibration reste limitée. L’étude demeure adaptative : ces pages ont déjà été examinées dans les essais précédents, et la provenance des données utilisées pour les poids/prototypes historiques reste inconnue. Il ne s’agit pas d’une validation aveugle sur de nouvelles pages. Le runtime actif reste inchangé.',
 f"**Vérifications.** Revue native indépendante du protocole et du runner avant lancement, puis des résultats. Contrôles intégrés : tirage stable malgré l’ordre d’entrée, garde rejetant un meilleur F1 qui perd des rares, repli à 0,35 sans rares, seuil inclusif, tie vers le plus bas et rematching après rejet. Assertions d’exclusion des groupes/pixels, de représentation des 286 classes, de reproduction de chaque plan de batches, de 15 checkpoints et des réponses historiques. Sources, caches, images et fichiers actifs vérifiés par hash avant/après. Temps cumulé des fits : {d['fit_seconds']:.1f} s ; fit + calibration + readout : {d['elapsed_seconds']:.1f} s. 15000 updates, aucune activation ni évaluation dev/test.",
 '[Résultats, courbes, prédictions, partitions et historiques de fit](nested-rejection-results-20261004.json). [Readout complémentaire par branche](nested-branch-rejection-results-20261004.json). [Calibration précédente sur pages ayant servi aux poids](rejection-calibration-research-20261004.md). Runner : `scripts/research_nested_rejection.py` ; run : `backend/training_runs/20261004-nested-rejection-v1`.',
]
(ROOT/'reports/nested-rejection-research-20261004.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
analysis=dict(scope='Posthoc report only, no head/threshold selection',results_sha256=sha(PATH),report_source_sha256=sha(Path(__file__)),branch_results_sha256=sha(BRANCH_PATH),comparison=comparison,guards=guards,inputs_unchanged=True,activated=False)
(ROOT/'reports/nested-rejection-analysis-20261004.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'));pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
canvas=Drawing(1080,760);canvas.add(Rect(0,0,1080,760,fillColor=HexColor('#ffffff'),strokeColor=None))
def text(x,y,s,size=11,bold=False,color='#263b46'):canvas.add(String(x,y,s,fontName='ArialBold' if bold else 'Arial',fontSize=size,fillColor=HexColor(color)))
text(30,720,'Apprendre les poids et régler le rejet sur des pages distinctes',22,True)
text(30,692,'15 nouveaux fits • calibration sur 3 pages exclues des poids • évaluation externe sur 17 pages déjà examinées',11)
for i,(x,label) in enumerate([(30,'Recours rare : rejet'),(380,'Précision exacte'),(560,'Rappel exact'),(745,'F1 exact'),(895,'Rares précédents /10')]):text(x,653,label,11,True)
for i,(p,label) in enumerate(POLICIES.items()):
 y=612-i*62;z=mean('gated',p);color='#626d74' if p=='current_035' else '#117653'
 text(30,y,label,13,True,color)
 for x,k in [(380,'exact_precision'),(560,'exact_recall'),(745,'exact_f1')]:text(x,y,f"{100*z[k]:.1f} %",16,True,color)
 text(895,y,refseq('gated',p),13,True,color)
 text(30,y-22,f"Correctes : {seq('gated',p,'correct')} /343 ; rares du fit : {seq('gated',p,'rare_correct')} /{cs('gated',p)[0]['rare_known']}",10,color='#626d74')
text(30,377,'Répartition des 654 propositions du candidat · moyenne des graines',13,True)
colors=['#117653','#b45151','#d99536','#bdc9d1','#e3e9ed']
for x,label,color in zip([30,210,425,650,835],['Correctes','Mauvais noms','Non appariées','Ignorées','Rejetées'],colors):canvas.add(Rect(x,346,12,12,fillColor=HexColor(color),strokeColor=None));text(x+18,347,label,10)
for i,(p,label) in enumerate(POLICIES.items()):
 y=312-i*51;amounts=[statistics.mean(z[k] for z in cs('gated',p)) for k in ['correct','wrong_name','unmatched_predictions','ignored_predictions','rejected']]
 assert abs(sum(amounts)-654)<1e-8
 text(30,y,label,10,True);x=230
 for amount,color in zip(amounts,colors):
  width=790*amount/654;canvas.add(Rect(x,y-6,width,21,fillColor=HexColor(color),strokeColor=None));x+=width
 text(230,y-22,f"{amounts[0]:.1f} correctes · {amounts[1]:.1f} mauvais noms · {amounts[2]:.1f} non appariées · {amounts[4]:.1f} rejetées",9)
text(30,111,'Rares mesurés : F1 libre '+('passe' if rare_preserved('calibrated') else 'échoue')+' ; par branche '+('passe' if rare_preserved('rare_branch') else 'échoue')+'. Aucun modèle activé.',12,True)
text(30,80,'Par branche : 0,35 pour les sorties routées vers l’historique ; seuil appris sur calibration pour les autres.',11)
text(30,50,'Comparaison principale : mêmes nouveaux poids à 0,35. Trois graines partagent les mêmes pages.',10,color='#626d74')
text(30,25,'Étude rétrospective ; provenance de l’entraînement historique inconnue. Précision ≠ rappel.',10,color='#626d74')
renderSVG.drawToFile(canvas,str(ROOT/'reports/nested-rejection-20261004.svg'))
with pdfium.PdfDocument(renderPDF.drawToString(canvas)) as pdf:pdf[0].render(scale=2).to_pil().save(ROOT/'reports/nested-rejection-20261004.png')
print(json.dumps(dict(comparison=comparison,guards=guards),indent=2))

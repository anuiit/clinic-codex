"""Measured calibration report and figure from frozen readout results."""
import hashlib,json,statistics
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String,Rect,Line
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'reports/rejection-calibration-results-20261004.json'
d=json.loads(PATH.read_text());agg=d['aggregate'];seeds=[42,43,44]
NAMES={'baseline':'Historique','paired':'Paires + conservation','gated':'Paires + recours rare'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path,digest in d['protocol']['input_sha256'].items():
 p=ROOT/path.removeprefix('/home/sina/omxpro/clinic-codex/') if path.startswith('/home/sina/omxpro/clinic-codex/') else Path('//wsl.localhost/Ubuntu-24.04'+path)
 assert sha(p)==digest,path
assert not d['activated'] and d['optimizer_updates']==0 and not d['locked_test_scored'] and all(d['verification'].values())
cs=lambda m,p:[agg[m][p]['seeds'][str(s)]['counts'] for s in seeds]
seq=lambda m,p,k:' / '.join(str(z[k]) for z in cs(m,p))
mean=lambda m,p:agg[m][p]['mean_metrics']
thresholds=lambda m:[e['selections'][m]['threshold'] for e in d['folds']]
summary={}
for m in NAMES:
 before=cs(m,'current_035');after=cs(m,'calibrated')
 summary[m]=dict(selected_thresholds=thresholds(m),lost_correct=[agg[m]['calibrated']['seeds'][str(s)]['lost_correct'] for s in seeds],gained_correct=[agg[m]['calibrated']['seeds'][str(s)]['gained_correct'] for s in seeds],incorrect_outputs_removed=[(a['scored_predictions']-a['correct'])-(b['scored_predictions']-b['correct']) for a,b in zip(before,after)],nonmatched_removed=[a['unmatched_predictions']-b['unmatched_predictions'] for a,b in zip(before,after)])
metrics_table='| Variante et rejet | Correctes, graines 42 /43 /44 | Précision exacte | Rappel exact | F1 exact | Rares /10 | Macro des classes rares |\n|---|---:|---:|---:|---:|---:|---:|\n'
for m,name in NAMES.items():
 for policy,label in [('current_035','0,35'),('calibrated','calibré sur train')]:
  a=mean(m,policy)
  metrics_table+=f"| {name}, {label} | {seq(m,policy,'correct')} | {100*a['exact_precision']:.2f} % | {100*a['exact_recall']:.2f} % | {100*a['exact_f1']:.2f} % | {seq(m,policy,'rare_correct')} | {100*a['rare_class_macro']:.2f} % |\n"
counts_table='| Variante calibrée | Acceptées | Ignorées | Évaluées | Rejetées /654 | Mauvais noms | Non appariées | Objets manqués /343 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for m,name in NAMES.items():counts_table+=f"| {name} | "+' | '.join(seq(m,'calibrated',k) for k in ['accepted','ignored_predictions','scored_predictions','rejected','wrong_name','unmatched_predictions','missed'])+' |\n'
selection_table='| Partition | Graine | Pages de calibration | Historique | Paires | Recours rare | F1 moyen train, recours rare |\n|---|---:|---:|---:|---:|---:|---:|\n'
for e in d['folds']:
 selection_table+=f"| {e['fold']} | {e['seed']} | {len(e['train_pages'])} | "+' | '.join(f"{e['selections'][m]['threshold']:.2f}" for m in NAMES)+f" | {100*e['selections']['gated']['objective']:.2f} % |\n"
gain_table='| Variante | Correctes perdues | Correctes gagnées après rematching | Sorties incorrectes supprimées, bilan net | Non appariées supprimées, bilan net |\n|---|---:|---:|---:|---:|\n'
for m,name in NAMES.items():gain_table+=f"| {name} | "+' | '.join(' / '.join(str(v) for v in summary[m][k]) for k in ['lost_correct','gained_correct','incorrect_outputs_removed','nonmatched_removed'])+' |\n'
pages_table='| Variante | Précision moyenne par page : 0,35 → calibré | Rappel moyen par page | F1 moyen par page |\n|---|---:|---:|---:|\n'
for m,name in NAMES.items():pages_table+=f"| {name} | "+' | '.join(f"{100*agg[m]['current_035']['mean_page'][k]:.2f} → {100*agg[m]['calibrated']['mean_page'][k]:.2f} %" for k in ['exact_precision','exact_recall','exact_f1'])+' |\n'
b=mean('gated','current_035');a=mean('gated','calibrated');s=summary['gated']
rare_ok=d['guard']['rare_correct_no_regression'] and d['guard']['rare_class_macro_no_regression']
opinion=('La calibration améliore le compromis global, avec conservation des rares observés. Elle fournit une piste simple à poursuivre, mais les seuils par partition ne sont pas un seuil global validé pour production.' if rare_ok else 'Le filtre trie utilement les sorties, mais le seuil global choisi pour le F1 sacrifie des glyphes rares. Je ne recommande pas de l’activer tel quel. Le prochain essai devrait d’abord réserver, au sein du train de chaque partition externe, des pages de calibration exclues de l’apprentissage des poids. On règle le seuil sur ce sous-lot, puis on évalue les pages externes. Cela nécessite de nouveaux fits : les checkpoints OOF des autres partitions ont pu voir les pages externes et ne sont pas réutilisables pour cette calibration imbriquée. La conservation des rares doit être contrôlée sur ce sous-lot ; si les rares y sont trop peu nombreux, deux seuils distincts seraient mal étayés.')
lines=[
 '# Calibration du rejet cosinus — 4 octobre 2026',
 f"**Résultat.** Pour le candidat avec recours rare, la précision exacte passe de {100*b['exact_precision']:.2f} % à {100*a['exact_precision']:.2f} %, le rappel exact de {100*b['exact_recall']:.2f} % à {100*a['exact_recall']:.2f} %, et le F1 exact de {100*b['exact_f1']:.2f} % à {100*a['exact_f1']:.2f} %. Les bonnes reconnaissances passent de {seq('gated','current_035','correct')} à {seq('gated','calibrated','correct')} sur 343 objets. Les rares passent de {seq('gated','current_035','rare_correct')} à {seq('gated','calibrated','rare_correct')} sur dix. Aucun entraînement supplémentaire ni activation.",
 '**Protocole fixé avant ce lancement.** Les checkpoints paired step1000, prototypes historiques, caractéristiques DINO et propositions SAM sont réutilisés tels quels. Cinq partitions et trois graines ; même historique, même candidat et même règle de recours rare. Pour chaque méthode/partition/graine, on évalue les seuils 0,35 à 0,95 par pas de 0,05 uniquement sur ses pages d’entraînement autorisées. Le seuil maximise la moyenne arithmétique du F1 exact par page ; une égalité à 1e-12 choisit le plus bas. Le F1 impose à la fois IoU ≥ 0,5 et bon nom. Les pages exclues ne servent à aucun choix de seuil. Les rares sont contrôlés après sélection, sans adapter le seuil à leur résultat held.',
 '**Séparation des données.** La calibration utilise les scores du head propre à chaque partition, et toutes les pages restantes après fermeture des composants communs. Les supports naturels sont les pixels de crops raw uniques par classe dans ce même train. On n’utilise pas les prédictions OOF d’autres partitions pour calibrer. Les prédictions held figées du readout précédent entrent uniquement après la sélection. Le seuil courant 0,35 et ses métriques sont reproduits exactement. Aucun score dev/test ni nouvelle image dev/test n’est calculé.',
 '**Limites.** Ces 17 planches et leurs 343 objets connus ont déjà servi aux recherches précédentes. Le seuil est calibré sur des pages qui ont entraîné la projection : cette calibration train peut être optimiste. Les résultats held évaluent son transfert aux pages exclues de cette adaptation, mais restent rétrospectifs et adaptatifs. La provenance d’entraînement de l’historique reste inconnue. Les trois graines partagent les mêmes pages et ne sont pas trois jeux de test indépendants. Aucun lot inédit n’existe. On ne déduit ni généralisation à l’aveugle ni autorisation de mise en production.',
 '**Résultats held.** Moyennes des métriques micro sur les trois graines ; tous les comptes sont fournis par graine. Chaque graine couvre les 17 pages une fois.',
 metrics_table,
 '![Effet du rejet calibré](rejection-calibration-20261004.png)',
 '**Dénominateurs et détection.** La précision exacte est bonnes boîtes et bons noms /propositions acceptées évaluables ; le rappel exact utilise toujours les 343 objets connus. Les propositions appariées en géométrie sont correctes seulement si leur nom convient. Les sept régions de classe inconnue ou conflictuelle sont ignorées pour les propositions non appariées qui les recouvrent. Le nombre ignoré change avec le rejet. Les non appariées peuvent être du fond, des fragments, des boîtes trop larges ou des concurrentes ; ce terme ne signifie pas uniquement pixels de fond.',
 counts_table,
 '**Ce qui disparaît.** Le rejet ne crée aucune nouvelle proposition SAM et ne corrige pas directement un nom. Il supprime des propositions et refait l’appariement géométrique ; une bonne proposition auparavant concurrencée peut alors être retenue. Le bilan des sorties incorrectes inclut mauvais noms et propositions non appariées. Les compteurs ci-dessous sont calculés sur les objets corrects avant/après, pas simplement sur la différence de leurs totaux.',
 gain_table,
 '**Moyennes par page.** L’objectif de sélection traite chaque page avec le même poids. Les scores micro ci-dessus pondèrent implicitement les volumes. Publier les deux permet de voir leur différence.',
 pages_table,
 '**Seuils sélectionnés, uniquement sur train.** Ces 15 seuils par méthode décrivent une procédure de validation croisée. Leur moyenne ou médiane n’est pas un seuil de production validé.',
 selection_table,
 f"**Transfert de la calibration.** Le F1 moyen par page du candidat sur ses pages train, au seuil choisi, va de {100*min(e['selections']['gated']['objective'] for e in d['folds']):.2f} % à {100*max(e['selections']['gated']['objective'] for e in d['folds']):.2f} %. Le F1 moyen par page held agrégé est {100*agg['gated']['calibrated']['mean_page']['exact_f1']:.2f} %. Cet écart est compatible avec une calibration optimiste sur des exemples déjà appris et des variations entre pages ; il ne permet pas d’attribuer toute la perte à une seule cause. Séparer les pages d’apprentissage des poids et de calibration serait plus informatif avant d’ajouter de la flexibilité au filtre.",
 '**Glyphes rares.** Un objet est rare si sa classe possède de un à quatre crops raw de pixels uniques dans le train de sa partition. Il y a dix objets rares held, tous géométriquement appariés avant rejet. La macro donne le même poids à chaque classe rare, alors que le rappel compte chaque objet. La règle de recours choisit l’historique selon sa classe prédite avant rejet ; elle ne garantit donc pas qu’un seuil cosinus commun conservera cette sortie. Les contrôles de non-régression rares ont '+('réussi sur ce lot.' if rare_ok else 'échoué sur ce lot.')+' Aucun seuil alternatif n’a été choisi après ce constat.',
 '**Mon avis.** '+opinion+' Le rejet seul ne récupère pas les 89 glyphes sans proposition correspondante avant filtrage. Le gain de précision doit être lu avec les bonnes reconnaissances perdues et le rappel restant. Le runtime actif conserve ses poids et sa configuration.',
 '**Vérifications.** Check intégré : seuil inclusif, tie vers le seuil inférieur, erreur sur calibration vide, correspondances recalculées quand une proposition concurrente est rejetée. Les groupes et lignes de calibration sont autorisés par le même scope que les poids. Les supports et métriques de référence sont reproduits. Sources, caches, checkpoints, images et fichiers actifs sont vérifiés par hash avant/après. Une revue indépendante du protocole et du runner puis des résultats complète ces contrôles.',
 f"Temps du readout/calibration : {d['elapsed_seconds']:.1f} s ; zéro mise à jour de gradient. [Courbes train, seuils, prédictions held et métriques](rejection-calibration-results-20261004.json). [Inférence complète précédente](full-proposal-research-20261004.md). Runner : `scripts/research_rejection_calibration.py` ; run : `backend/training_runs/20261004-rejection-calibration-v1`.",
]
(ROOT/'reports/rejection-calibration-research-20261004.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
verification=dict(scope='Posthoc reporting only; no model/threshold selection',report_source_sha256=sha(Path(__file__)),results_sha256=sha(PATH),summary=summary,inputs_unchanged=True,activated=False)
(ROOT/'reports/rejection-calibration-analysis-20261004.json').write_text(json.dumps(verification,indent=2)+'\n')
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'));pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
canvas=Drawing(1080,680);canvas.add(Rect(0,0,1080,680,fillColor=HexColor('#ffffff'),strokeColor=None))
def text(x,y,s,size=11,bold=False,color='#263b46'):canvas.add(String(x,y,s,fontName='ArialBold' if bold else 'Arial',fontSize=size,fillColor=HexColor(color)))
text(30,640,'Calibrer le rejet : ce qu’on gagne et ce qu’on perd',23,True)
text(30,611,'Seuils choisis sur train • résultats sur les pages exclues • 17 pages déjà examinées • aucune activation',11)
text(30,575,'Variante',11,True);text(280,575,'Rejet',11,True)
for x,title in [(435,'Précision exacte'),(620,'Rappel exact'),(795,'F1 exact'),(940,'Rares /10')]:text(x,575,title,11,True)
for i,m in enumerate(NAMES):
 y=545-i*65;text(30,y,NAMES[m],12,True)
 for offset,p,label in [(0,'current_035','Seuil 0,35'),(-23,'calibrated','Calibré sur train')]:
  z=mean(m,p);color='#117653' if p=='calibrated' else '#626d74';text(280,y+offset,label,11,color=color)
  for x,k in [(435,'exact_precision'),(620,'exact_recall'),(795,'exact_f1')]:text(x,y+offset,f"{100*z[k]:.1f} %",14,p=='calibrated',color)
  text(940,y+offset,seq(m,p,'rare_correct'),12,p=='calibrated',color)
text(30,315,'Toutes les propositions du candidat avec recours rare',14,True)
colors=['#117653','#b45151','#d99536','#bdc9d1','#e3e9ed']
for i,(label,color) in enumerate(zip(['Correctes','Mauvais noms','Non appariées','Ignorées','Rejetées'],colors)):
 x=[30,205,410,640,810][i];canvas.add(Rect(x,284,12,12,fillColor=HexColor(color),strokeColor=None));text(x+18,285,label,10)
for j,p in enumerate(['current_035','calibrated']):
 y=249-j*63;text(30,y,'Seuil 0,35' if j==0 else 'Calibré sur train',11,True)
 amounts=[statistics.mean(z[k] for z in cs('gated',p)) for k in ['correct','wrong_name','unmatched_predictions','ignored_predictions','rejected']]
 assert abs(sum(amounts)-654)<1e-8
 x=185
 for amount,color in zip(amounts,colors):
  width=835*amount/654;canvas.add(Rect(x,y-7,width,23,fillColor=HexColor(color),strokeColor=None));x+=width
 text(185,y-25,f"{amounts[0]:.1f} correctes · {amounts[1]:.1f} mauvais noms · {amounts[2]:.1f} non appariées · {amounts[4]:.1f} rejetées",10)
text(30,124,f"Bonnes reconnaissances : {seq('gated','current_035','correct')} → {seq('gated','calibrated','correct')} /343 objets connus.",12,True)
text(30,95,f"Seuils du candidat : {min(thresholds('gated')):.2f} à {max(thresholds('gated')):.2f}. Rares : {seq('gated','calibrated','rare_correct')} /10 après rejet.",11)
text(30,65,'Non-régression des rares : '+('satisfaite sur ce lot.' if rare_ok else 'échouée. Ne pas activer ce seuil commun tel quel.'),12,True,color='#117653' if rare_ok else '#b45151')
text(30,35,'Précision = correctes /sorties évaluées ; rappel = correctes /343. Calibration in-sample, étude rétrospective.',10,color='#626d74')
renderSVG.drawToFile(canvas,str(ROOT/'reports/rejection-calibration-20261004.svg'))
with pdfium.PdfDocument(renderPDF.drawToString(canvas)) as pdf:pdf[0].render(scale=2).to_pil().save(ROOT/'reports/rejection-calibration-20261004.png')
print(json.dumps(dict(summary=summary,guard=d['guard']),indent=2))

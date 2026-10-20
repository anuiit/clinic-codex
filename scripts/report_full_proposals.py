"""Full-proposal report and measured figure; no threshold selection."""
import bisect,collections,hashlib,json,statistics
from pathlib import Path
import numpy as np
from reportlab.graphics.shapes import Drawing,String,Rect,Line
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'reports/full-proposal-results-20261004.json'
d=json.loads(PATH.read_text());agg=d['aggregate'];seeds=[42,43,44]
NAMES={'baseline':'Historique','paired':'Conservation + paires','gated':'Paires + recours historique rare'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# WSL preregistration paths map to this checkout on the desktop host.
for path,digest in d['protocol']['input_sha256'].items():
 p=ROOT/path.removeprefix('/home/sina/omxpro/clinic-codex/') if path.startswith('/home/sina/omxpro/clinic-codex/') else Path('//wsl.localhost/Ubuntu-24.04'+path)
 assert sha(p)==digest,path
assert not d['activated'] and d['optimizer_updates']==0 and not d['locked_test_scored']
assert all(d['verification'].values())

def auc(rows):
 negative=sorted(score for score,label in rows if not label);positive=[score for score,label in rows if label]
 assert positive and negative
 return sum((bisect.bisect_left(negative,x)+bisect.bisect_right(negative,x))/2 for x in positive)/(len(positive)*len(negative))

assert auc([(1,True),(0,False)])==1 and auc([(0,True),(1,False)])==0 and auc([(1,True),(1,False)])==.5
confidence={}
for method in NAMES:
 by_seed={}
 for seed in seeds:
  records=[]
  for e in d['fold_pages']:
   if e['seed']!=seed:continue
   ev=e['methods'][method]['evaluations']['no_rejection'];correct={z['proposal_index'] for z in ev['matches'] if z['correct']};matched={z['proposal_index'] for z in ev['matches']};ignored=set(ev['ignored_proposal_indices'])
   for r in e['methods'][method]['predictions']:
    if r['proposal_index'] in ignored:continue
    records.append(dict(cosine=r['similarity'],margin=r['top3_similarities'][0]-r['top3_similarities'][1],sam=r['sam_confidence'],exact=r['proposal_index'] in correct,geometry=r['proposal_index'] in matched))
  assert len(records)==647
  by_seed[str(seed)]=dict(minimum_cosine=min(r['cosine'] for r in records),auc={score:{target:auc([(r[score],r[target]) for r in records]) for target in ['geometry','exact']} for score in ['cosine','margin','sam']},cosine_quantiles={key:dict(zip(['min','p10','median','p90','max'],np.quantile([r['cosine'] for r in records if r['exact']==truth],[0,.1,.5,.9,1]).tolist())) for key,truth in [('correct',True),('incorrect',False)]})
 confidence[method]=dict(seeds=by_seed,mean_auc={score:{target:statistics.mean(z['auc'][score][target] for z in by_seed.values()) for target in ['geometry','exact']} for score in ['cosine','margin','sam']})
analysis=dict(scope='Posthoc score discrimination on the same examined pages; no threshold sweep, fitting or selection',source_sha256=sha(Path(__file__)),readout_sha256=sha(PATH),confidence=confidence,inputs_and_active_files_unchanged=True)
(ROOT/'reports/full-proposal-analysis-20261004.json').write_text(json.dumps(analysis,indent=2)+'\n')
counts=lambda m:[agg[m]['current_035']['seeds'][str(s)]['counts'] for s in seeds]
summary=lambda m:agg[m]['current_035']['mean_metrics']
sequence=lambda m,key:' / '.join(str(z[key]) for z in counts(m))
c=counts('gated')[0]
assert all(z['rejected']==0 and z['unmatched_predictions']==393 and z['ignored_predictions']==7 for m in NAMES for z in counts(m))
for m in NAMES:assert agg[m]['current_035']==agg[m]['no_rejection']
table='| Variante | Bonnes boîtes + bons noms, graines 42 /43 /44 | Précision exacte | Rappel exact | F1 exact | Rares /10 |\n|---|---:|---:|---:|---:|---:|\n'
table+='\n'.join(f"| {name} | {sequence(m,'correct')} | {100*summary(m)['exact_precision']:.2f} % | {100*summary(m)['exact_recall']:.2f} % | {100*summary(m)['exact_f1']:.2f} % | {sequence(m,'rare_correct')} |" for m,name in NAMES.items())
auc_table='| Variante | Cosinus : boîte appariée | Marge top1−top2 : boîte appariée | Score SAM : boîte appariée | Cosinus : boîte + nom corrects |\n|---|---:|---:|---:|---:|\n'
auc_table+='\n'.join(f"| {name} | {confidence[m]['mean_auc']['cosine']['geometry']:.3f} | {confidence[m]['mean_auc']['margin']['geometry']:.3f} | {confidence[m]['mean_auc']['sam']['geometry']:.3f} | {confidence[m]['mean_auc']['cosine']['exact']:.3f} |" for m,name in NAMES.items())
lines=[
 '# Inférence complète sur les propositions SAM — 4 octobre 2026',
 '**Résultat principal.** Le gain de reconnaissance est confirmé, mais il ne réduit pas le bruit des propositions SAM. Sur 654 propositions, sept sont ignorées et 647 sont évaluées : 254 correspondent à une boîte connue et 393 restent non appariées. Le seuil actuel de rejet, 0,35, ne rejette aucune sortie dans aucune variante ou graine. Le modèle actif est inchangé.',
 '**Périmètre.** Le métier confirme que toutes les annotations de chaque planche sont couvertes et qu’aucun nouveau lot n’est disponible. Cette évaluation utilise donc les mêmes 17 pages déjà examinées, avec les cinq partitions et trois graines de l’adaptation précédente. Chaque page est exclue de son entraînement de projection, après fermeture des composants. La règle du recours rare reste celle déjà testée ; aucune optimisation ni sélection de seuil supplémentaire n’est faite. L’historique a une provenance d’entraînement inconnue : les scores ne constituent pas une validation indépendante à l’aveugle.',
 '**Vérité terrain.** 343 objets de classes connues, plus sept régions approuvées de classe inconnue ou conflictuelle à ignorer. Toutes les géométries distinctes approuvées ont été vérifiées, y compris les crops dédupliqués du train : ce contrôle n’ajoute ici aucun objet connu. Les régions ambiguës ne sont pas traitées comme du fond. Les 16 annotations rejetées de l’archive ne servent pas de vérité terrain.',
 '**Matching.** IoU ≥ 0,5, appariement un-à-un fondé uniquement sur la géométrie, puis vérification du nom. Le helper trie par IoU décroissante puis indices GT et proposition décroissants en cas d’égalité. Après appariement des objets connus, les propositions restantes qui chevauchent une région ignorée à IoU ≥ 0,5 sont exclues du dénominateur. Les autres sont des erreurs de détection : cela peut inclure des fragments, des boîtes trop larges et des doublons, ainsi que du fond. Ce terme ne signifie pas que toutes ces régions sont des pixels de fond.',
 '**Résultats au seuil courant 0,35.** Moyennes des métriques sur les trois graines ; les comptes restent publiés par graine.',
 table,
 f"La précision exacte divise les boîtes correctement placées et nommées par les 647 propositions évaluées. Le rappel exact les divise par les 343 objets connus. Le F1 combine ces deux quantités. Le candidat avec recours rare atteint en moyenne {100*summary('gated')['exact_precision']:.2f} % de précision exacte, contre {100*summary('baseline')['exact_precision']:.2f} % pour l’historique : il corrige des noms sans supprimer les mauvaises propositions. Sur les seuls 254 crops appariés, sa top-1 est {100*summary('gated')['conditional_top1']:.2f} % ; ce chiffre décrit une autre question et ne mesure pas la fiabilité de toutes les sorties.",
 '**Décomposition de la détection.** Dans toutes les variantes : 654 propositions, sept ignorées, 254 appariées, 393 non appariées, 89 objets connus manqués. Précision géométrique : 39,26 % ; rappel géométrique : 74,05 % ; F1 géométrique : 51,31 %. Ces nombres restent inchangés puisque SAM est figé et qu’aucune proposition n’est rejetée.',
 f"Parmi les 393 non appariées, {c['duplicate_predictions']} chevauchent néanmoins un objet connu à IoU ≥ 0,5 déjà utilisé par l’appariement : ce sont des propositions concurrentes/doublons selon ce critère. Les {c['unmatched_other']} autres restent sous le seuil de correspondance avec toute boîte connue. La nouvelle recette laisse {sequence('gated','wrong_name')} mauvais noms sur les boîtes appariées, contre 94 pour l’historique.",
 '![Décomposition de toutes les sorties](full-proposal-20261004.png)',
 '**Effet du rejet.** Les résultats avec seuil 0,35 sont strictement identiques à ceux sans rejet. Le seuil ne filtre donc rien sur ces 17 pages. La similarité cosinus n’est pas une probabilité de présence de glyphe ; la bonne valeur de rejet historique n’est pas automatiquement une calibration valide pour une nouvelle projection.',
 '**Diagnostic des scores, après évaluation.** Sans choisir de seuil, on mesure si les scores classent les bonnes sorties au-dessus des mauvaises. L’AUC compare toutes les paires positif/négatif, une égalité comptant pour moitié : 0,5 décrit un ordre sans discrimination. La cible « boîte appariée » ignore le nom ; la cible « boîte + nom » exige les deux. Les sept propositions ignorées sont exclues. La marge est la différence entre les similarités des deux premiers noms, et le score SAM reste le score de qualité du masque.',
 auc_table,
 'Les quantiles par graine des cosinus corrects et incorrects et les AUC détaillées figurent dans le JSON d’analyse. Ce diagnostic rétrospectif peut orienter un prochain essai, mais il ne valide aucun nouveau seuil. Un éventuel filtrage devra être réglé uniquement sur les pages train de chaque partition, puis évalué sur les pages exclues, avec un contrôle des glyphes rares.',
 '**Mon avis.** Le recours historique reste la meilleure variante de reconnaissance de cette série, mais la chaîne complète produit encore beaucoup de sorties incorrectes. Le prochain essai le plus simple est de calibrer le rejet cosinus existant uniquement sur les pages train de chaque partition, puis de mesurer les pages exclues et les glyphes rares. L’AUC cosinus du candidat, 0,720 pour les boîtes et 0,813 pour les boîtes et noms corrects, indique un signal de tri utile mais imparfait. Si ce réglage ne fournit pas un compromis acceptable entre précision, rappel et rares, on pourra tester un filtre appris de validité des propositions. Changer encore la seule projection de noms ne résout pas les 393 propositions non appariées ni les 89 glyphes manqués. Sans nouvelles pages, on peut mesurer ces adaptations par partitions, tout en gardant explicitement la limite rétrospective. Je conserve le runtime actif tant qu’un gain sur la précision complète, le rappel et les rares n’est pas démontré.',
 '**Vérifications.** La génération SAM à grille 16×16, quatre points par batch et marge cinq pixels reproduit les nombres de propositions, les pixels et les features des 254 anciennes découpes. Leurs prédictions sont reproduites avec les mêmes checkpoints. Les sources, caches, images, checkpoints et fichiers actifs sont vérifiés par hash. Le premier lancement a terminé l’extraction, puis échoué sur une clé technique manquante avant tout score. Sa source et le motif sont archivés ; v2 reprend le cache vérifié sans nouvelle extraction ni changement de recette. Les readouts v2 ont été relus indépendamment et aucun défaut matériel n’a été trouvé.',
 '[Prédictions et métriques](full-proposal-results-20261004.json). [Distributions et discrimination des scores](full-proposal-analysis-20261004.json). [Rapport des entraînements précédents](crop-compatible-research-20261004.md). Run : `backend/training_runs/20261004-full-proposal-readout-v2` ; script : `scripts/research_full_proposals.py`.',
]
(ROOT/'reports/full-proposal-research-20261004.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'));pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
canvas=Drawing(1080,650);canvas.add(Rect(0,0,1080,650,fillColor=HexColor('#ffffff'),strokeColor=None))
def text(x,y,s,size=11,bold=False,color='#263b46'):canvas.add(String(x,y,s,fontName='ArialBold' if bold else 'Arial',fontSize=size,fillColor=HexColor(color)))
colors=['#117653','#b45151','#d99536','#bdc9d1','#eef2f5']
text(30,610,'Ce que produit l’inférence complète',23,True)
text(30,581,'17 planches déjà examinées • 343 glyphes connus • 654 propositions SAM • aucune activation',12)
labels=['Boîte + nom corrects','Boîte correcte, mauvais nom','Sans correspondance IoU ≥ 0,5','Ignorées','Rejetées']
for i,(label,color) in enumerate(zip(labels,colors)):
 x=[30,245,510,825,935][i];canvas.add(Rect(x,545,12,12,fillColor=HexColor(color),strokeColor=None));text(x+17,546,label,10)
text(280,509,'Répartition des 654 propositions · moyenne des trois graines',11,True)
text(965,509,'Précision exacte',10,True)
for i,m in enumerate(NAMES):
 y=465-i*103;cs=counts(m)
 text(30,y,NAMES[m] if m!='gated' else 'Paires + recours rare',12,True)
 amounts=[statistics.mean(z[k] for z in cs) for k in ['correct','wrong_name','unmatched_predictions','ignored_predictions','rejected']]
 assert abs(sum(amounts)-654)<1e-8
 x=280
 for amount,color in zip(amounts,colors):
  width=650*amount/654;canvas.add(Rect(x,y-9,width,28,fillColor=HexColor(color),strokeColor=None));x+=width
 text(965,y,f"{100*summary(m)['exact_precision']:.1f} %",14,True)
 text(280,y-33,f"{amounts[0]:.1f} correctes · {amounts[1]:.1f} mauvais noms · 393 non appariées · 7 ignorées",10)
 text(280,y-53,'Bonnes reconnaissances par graine : '+sequence(m,'correct'),10,color='#626d74')
for val in [0,160,254,400,654]:
 x=280+650*val/654;canvas.add(Line(x,167,x,177,strokeColor=HexColor('#626d74')));text(x-8,151,str(val),9)
text(30,111,'Détection : 254/343 objets connus appariés ; 89 manqués. 393 propositions non appariées dans toutes les variantes.',11)
text(30,84,f"Doublons/concurrents selon l’IoU : {c['duplicate_predictions']}. Seuil de rejet 0,35 : aucune sortie filtrée.",11)
text(30,56,'Précision exacte : bonnes boîtes et bons noms /647 sorties évaluées. Les 7 régions ambiguës sont exclues.',10,color='#626d74')
text(30,30,'Comparaison exploratoire : trois graines sur les mêmes pages, sans nouvel entraînement ni calibration du seuil.',10,color='#626d74')
renderSVG.drawToFile(canvas,str(ROOT/'reports/full-proposal-20261004.svg'))
with pdfium.PdfDocument(renderPDF.drawToString(canvas)) as pdf:pdf[0].render(scale=2).to_pil().save(ROOT/'reports/full-proposal-20261004.png')
print(json.dumps({'diagnostic_auc':{m:z['mean_auc'] for m,z in confidence.items()},'duplicate_predictions':c['duplicate_predictions'],'minimum_cosine':{m:[z['minimum_cosine'] for z in v['seeds'].values()] for m,v in confidence.items()}},indent=2))

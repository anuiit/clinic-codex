"""Render completed DINO-tail results; reporting only."""
import base64, collections, csv, hashlib, html, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'backend/training_runs/cen-semantic-report-deps'), str(ROOT)]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN = ROOT/'backend/training_runs/20261004-dino-tail-v2'
OUT = ROOT/'reports/dino-tail-20261004-v2/report'
OUT.mkdir(parents=True,exist_ok=True)
r = json.loads((RUN/'results.json').read_text())
verification = json.loads((RUN/'verification.json').read_text())
assert verification['inputs_active_weights_and_images_unchanged']
views = json.loads((ROOT/'backend/training_runs/20261004-dino-views-v1/results.json').read_text())
assert all(views['verification'].values())
vp = json.loads((ROOT/'backend/training_runs/20261004-dino-views-v1/preregistration.json').read_text())
for name,digest in vp['input_sha256'].items():
    with Path(name).open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest()==digest
methods = ['best_prior','head_only','last_two_blocks']
titles = ['Meilleur candidat précédent','Tête seule · 200 mises à jour','DINO : deux derniers blocs + tête']
families = r['families']
correct = [families[k]['mask']['gated']['correct'] for k in methods]
assert all(families[k]['mask']['gated']['count']==254 and families[k]['raw']['gated']['count']==343 for k in methods)
fig, ax = plt.subplots(figsize=(11.6,3.5),dpi=160)
fig.patch.set_facecolor('#f7f9fc'); ax.set_facecolor('#f7f9fc')
for i,n in enumerate(correct):
    ax.barh(i,n,color='#2463b5',height=.54)
    ax.barh(i,254-n,left=n,color='#db8e3b',height=.54)
    ax.barh(i,89,left=254,color='#bbc4d0',height=.54)
    ax.text(n/2,i,f'{n} corrects',ha='center',va='center',color='white',fontweight='bold')
    ax.text(n+(254-n)/2,i,f'{254-n} mauvais noms',ha='center',va='center',fontsize=9)
    ax.text(298.5,i,'89 manqués',ha='center',va='center',fontsize=9)
ax.set_yticks(range(3),titles); ax.invert_yaxis(); ax.set_xlim(0,343)
ax.set_xlabel('343 éléments annotés connus · mêmes découpes SAM · recours historique pour les classes rares')
ax.set_title('Adapter DINOv2 : effet sur le résultat complet',loc='left',fontweight='bold',pad=16)
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(axis='y',length=0,pad=12)
fig.tight_layout(); fig.savefig(OUT/'comparison.png'); plt.close(fig)
rows = []
for k,title in zip(methods,titles):
    f = families[k]; raw=f['raw']['gated']; mask=f['mask']['gated']; rare=f['mask']['rare_gated']
    rows.append(f'<tr><td>{title}</td><td>{raw["correct"]}/343 · {100*raw["micro"]:.1f} %</td><td>{mask["correct"]}/254 · {100*mask["micro"]:.1f} %</td><td>{mask["correct"]}/343 · {100*f["joint_known"]["rate"]:.1f} %</td><td>{rare["correct"]}/{rare["count"]}</td></tr>')
old = [v for e in r['folds'] if e['method']=='head_only' for v in e['baseline']['mask']]
new = [v for e in r['folds'] if e['method']=='last_two_blocks' for v in e['evaluation']['mask']]
by_id={v['row_id']:v for v in old}; assert len(by_id)==254
stats=collections.defaultdict(lambda:[0,0,0])
for v in new:
    before=by_id[v['row_id']]; assert before['label']==v['label'] and before['group']==v['group']
    a=stats[v['label']]; a[0]+=1; a[1]+=int(before['gated_prediction']==v['label']); a[2]+=int(v['gated_prediction']==v['label'])
names={int(v['class_label']):v['element_name'] for v in csv.DictReader((ROOT/'backend/training_corpus/snapshots/snapshot-611c734590f272060251/metadata.csv').open())}
changed=sorted([(names[k],*v,v[2]-v[1]) for k,v in stats.items() if v[2]!=v[1]],key=lambda v:(-v[-1],v[0]))
class_rows=''.join(f'<tr><td>{html.escape(name)}</td><td>{n}</td><td>{a}</td><td>{b}</td><td>{delta:+d}</td></tr>' for name,n,a,b,delta in changed)
delta=r['comparisons']['best_prior']['delta_correct']; interval=r['comparisons']['best_prior']['exploratory_page_bootstrap_95_pp']
verdict = f'Le réglage testé apporte {delta:+d} bonnes réponses par rapport au meilleur candidat précédent ; le gain reste trop incertain pour justifier une activation.'
ci=f'Gain sur le résultat complet : {r["comparisons"]["best_prior"]["delta_joint_pp"]:+.2f} points ; intervalle descriptif par planche [{interval[0]:+.2f} ; {interval[1]:+.2f}] points.'
view_rows=[]
for k,title in zip(methods,titles):
    matched=views['raw_by_sam_availability'][k]['matched']
    rect=views['families'][k]['rectangle']['gated']
    masked=views['families'][k]['mask']['gated']
    view_rows.append(f'<tr><td>{title}</td><td>{matched["correct"]}/254 · {100*matched["micro"]:.1f} %</td><td>{rect["correct"]}/254 · {100*rect["micro"]:.1f} %</td><td>{masked["correct"]}/254 · {100*masked["micro"]:.1f} %</td></tr>')
view_table=''.join(view_rows)

image=base64.b64encode((OUT/'comparison.png').read_bytes()).decode()
page=f'''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test : adaptation de DINOv2</title>
<style>body{{background:#f7f9fc;color:#213248;font:16px/1.55 system-ui;margin:0}}main{{max-width:1100px;margin:40px auto;padding:0 24px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:22px;margin-top:32px}}.note{{padding:16px 20px;background:#eaf0f8;border-left:4px solid #2463b5}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;background:white}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dfe5ed}}th{{background:#eaf0f8}}.scroll{{overflow:auto}}.small{{font-size:14px;color:#516278}}code{{font-size:13px}}</style>
<main><p class="small">Expérience locale · 4 octobre 2026 · modèle actif inchangé</p><h1>Adapter le réseau visuel améliore-t-il la reconnaissance ?</h1>
<p class="note"><strong>{verdict}</strong><br>{ci}</p><img src="data:image/png;base64,{image}" alt="Comparaison des bonnes réponses, mauvais noms et éléments manqués sur343 annotations">
<h2>Résultats au checkpoint fixé à l'avance</h2><div class="scroll"><table><thead><tr><th>Configuration</th><th>Nom sur boîte métier fournie</th><th>Nom sur découpe SAM trouvée</th><th>Localisé et bien nommé</th><th>Cas rares SAM</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="small">Scores top1 avant rejet par seuil de confiance. Le troisième score est le rappel sur les annotations connues. Il ne mesure pas la précision des propositions supplémentaires : les planches comportent des annotations manquantes.</p>
<h2>Diagnostic de découpe : les mêmes 254 éléments</h2><div class="scroll"><table><thead><tr><th>Configuration</th><th>Boîte métier rectangulaire</th><th>Boîte SAM rectangulaire</th><th>Boîte SAM avec masque</th></tr></thead><tbody>{view_table}</tbody></table></div>
<p>Le masque SAM reste utile. Garder le fond dans les mêmes boîtes SAM donne 166 bonnes réponses avec DINO adapté, contre 182 avec masque. Avec les boîtes métier fournies, le gain de reconnaissance sur les objets trouvés passe de 155 à 168 bonnes réponses ; sur les 89 objets manqués par SAM, il passe de 54 à 56.</p>
<p class="small">Ce diagnostic a été décidé après le résultat principal : comparaison descriptive, sans nouvel entraînement. Boîtes métier et boîtes SAM ont des géométries différentes. Entre rectangle SAM et masque SAM, la boîte et le padding restent identiques ; la règle de recours historique est appliquée à chaque vue. Tous les résultats masqués reproduisent ceux du test principal.</p>
<h2>Ce qui a été entraîné</h2><p>Les deux branches repartent des mêmes cinq têtes de reconnaissance du candidat précédent. Nouvel AdamW, taux 0,00001, 200 mises à jour, mêmes exemples et pertes. La seconde branche adapte aussi les blocs 10 et 11 de DINOv2-S/14 et sa normalisation finale. Les dix premiers blocs, les prototypes et SAM restent fixes.</p>
<p>Le contrôle passe par les mêmes activations intermédiaires et les mêmes derniers blocs gelés. Les calculs utilisent float32 et le mode déterministe strict. Le recours au modèle historique pour les classes ayant 1 à 4 exemples distincts conserve la même règle.</p>
<h2>Classes dont le résultat change</h2><div class="scroll"><table><thead><tr><th>Classe</th><th>Éléments évalués</th><th>Corrects avant</th><th>Corrects après</th><th>Différence</th></tr></thead><tbody>{class_rows}</tbody></table></div>
<p class="small">Liste descriptive des découpes SAM : un changement sur un ou deux objets ne suffit pas à établir la performance générale d'une classe.</p>
<h2>Limites et preuves</h2><p>17 planches déjà examinées, 343 annotations et 254 correspondances SAM ; cinq plis avec exclusion des planches, composants de source et pixels concernés des pertes et gradients. Les images tenues à l'écart sont prétraitées par un préfixe fixe. Une seule graine d'entraînement. Aucune extraction de caractéristiques ni évaluation d'images dev/test pendant ce test ; aucun modèle activé.</p>
<p>Les poids sont repris depuis les checkpoints précédents ; l'état de leur ancien optimiseur n'est pas repris. Le contrôle rend les deux nouvelles branches comparables sous ce régime. Le checkpoint final à 200 pas est le résultat principal ; le checkpoint à 100 pas reste un diagnostic.</p>
<p>Le premier essai a été interrompu à cause d'un avertissement de calcul non déterministe. Ses scores sont exclus ; son cache float32, sans apprentissage, est réutilisé et vérifié.</p>
<p class="small">Sources : backend/training_runs/20261004-dino-tail-v2/preregistration.json, results.json, verification.json ; reports/dino-tail-results-20261004.json. Reproduction : <code>python -m scripts.research_dino_tail check</code>, puis <code>run</code> et <code>verify</code>.</p></main></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
summary={k:r[k] for k in ['schema_version','run','families','comparisons','activated','locked_test_scored','seed','total_optimizer_updates','scope']}
summary['view_diagnostic']={k:views[k] for k in ['families','raw_by_sam_availability','verification','scope']}
summary['class_changes']=[dict(name=name,count=n,before=a,after=b,delta=delta) for name,n,a,b,delta in changed]
summary['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'index.html',OUT/'comparison.png']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(report=str(OUT),delta=delta,counts=correct)))

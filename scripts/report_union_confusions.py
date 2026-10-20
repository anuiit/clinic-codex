"""Portable union/confusion report from completed, pinned research results."""
import base64,collections,csv,hashlib,html,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"backend/training_runs/cen-semantic-report-deps"),str(ROOT)]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import torch
from scripts.import_annotation_archive import _pixel_sha
RUN=ROOT/"backend/training_runs/20261005-union-confusions-v2"
OUT=ROOT/"reports/union-confusions-20261005-v2/report"
OUT.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def esc(value):return html.escape(str(value))
u=read(RUN/"union-results.json");r=read(RUN/"confusions-results.json")
for result in [u,r]:
    assert not result["activated"] and not result["locked_test_scored"]
    for path,digest in result["protocol"]["input_sha256"].items():assert sha(Path(path))==digest,path
for e in r["folds"]:assert sha(RUN/f'{e["method"]}_fold{e["fold"]}.pt')==e["checkpoint_sha256"]
cen=read(ROOT/"reports/cen-semantic-20261004-v2/class-text.json")
entries={e["class_label"]:e for e in cen["entries"]}
names=[entries[i]["name"] for i in range(286)]
assert names==read(ROOT/"backend/codex_model/config.json")["class_names"]
base={v:[q for e in r["folds"] if e["method"]=="control" for q in e["baseline"][v]] for v in ["raw","mask"]}
candidate={v:[q for e in r["folds"] if e["method"]=="hard_margin" for q in e["evaluation"][v]] for v in ["raw","mask"]}
assert {q["row_id"] for q in base["raw"]}=={q["row_id"] for q in candidate["raw"]}
classes=[]
for label in sorted({q["label"] for q in base["raw"]}):
    known=[q for q in base["raw"] if q["label"]==label]
    masked=[q for q in base["mask"] if q["label"]==label]
    later=[q for q in candidate["mask"] if q["label"]==label]
    confusion=collections.Counter(q["prediction"] for q in masked if q["prediction"]!=label)
    classes.append(dict(label=label,name=names[label],known=len(known),located=len(masked),
        correct=sum(q["prediction"]==label for q in masked),top3=sum(label in q["top3"] for q in masked),
        raw_correct=sum(q["prediction"]==label for q in known),
        hard_correct=sum(q["prediction"]==label for q in later),
        conditional=sum(q["prediction"]==label for q in masked)/len(masked) if masked else None,
        joint=sum(q["prediction"]==label for q in masked)/len(known),
        confusions=[dict(name=names[c],label=c,count=n) for c,n in confusion.most_common(3)]))
assert sum(q["known"] for q in classes)==343 and sum(q["located"] for q in classes)==254 and sum(q["correct"] for q in classes)==180
transitions={}
for view in ["raw","mask"]:
    gains=[];losses=[];sourcecounts=collections.Counter()
    for page in u["pages"]:
        s=page["naming"][view]["sam"];v=page["naming"][view]["union"]
        sc={q["gt_index"] for q in s["matches"] if q["correct"]};vc={q["gt_index"] for q in v["matches"] if q["correct"]}
        for kind,ids,target in [("gain",vc-sc,gains),("loss",sc-vc,losses)]:
            for j in sorted(ids):
                gt=page["gt"][j];target.append(dict(group=page["group"],gt_index=j,label=gt["label"],name=names[gt["label"]]))
        rows={q["proposal_index"]:q for q in page["proposals"][view]}
        for m in v["matches"]:
            if m["correct"]:sourcecounts[rows[m["proposal_index"]]["source"]]+=1
    transitions[view]=dict(gained_correct=len(gains),lost_correct=len(losses),gains=gains,losses=losses,correct_sources=dict(sourcecounts))
    assert len(gains)-len(losses)==u["naming"][view]["union"]["counts"]["correct"]-177
mine=[]
for e in r["folds"]:
    if e["method"]!="hard_margin":continue
    mine.append(dict(fold=e["fold"],pairs=[dict(names=[names[i] for i in q["labels"]],**q) for q in e["mining"]["pairs"]],
        mined_rows=len(e["mining"]["rows"]),eligible_terms=sum(h["eligible_terms"] for h in e["history"]),
        positive_terms=sum(h["positive_terms"] for h in e["history"]),steps=len(e["history"])))
summary=dict(schema_version="union-confusions-report.v1",union_geometry=u["geometry"],union_naming=u["naming"],
    coverage=u["coverage"],matched_set_union=u["matched_set_union"],naming_transitions=transitions,
    recognition=r["families"],classes=classes,mining=mine,activated=False,locked_test_scored=False,
    policies=dict(partial_export=True,precision_F1_interpretable=False,union_views_both_published=True,
        class_metrics="Same fixed SAM best_prior associations, not union",CEN="External catalogue descriptions only, not loss or predictions",
        top3="Gated score vector; top1 exactly the recorded prediction",scope="Same17 previously inspected train pages, one seed"),
    inputs_sha256={str(RUN/"union-results.json"):sha(RUN/"union-results.json"),str(RUN/"confusions-results.json"):sha(RUN/"confusions-results.json"),str(Path(__file__)):sha(Path(__file__))})
write(OUT/"summary.json",summary)

fig,axes=plt.subplots(1,2,figsize=(11.8,3.6),dpi=160)
labels=["SAM","YOLO partiel","SAM + YOLO"]
for ax,values,title in [(axes[0],[u["geometry"][k]["counts"]["matched"] for k in ["sam","yolo","union"]],"Boîtes associées aux annotations connues"),
                       (axes[1],[177,139,210],"Bonne boîte + bon nom · vue rectangle YOLO")]:
    for i,n in enumerate(values):
        ax.barh(i,n,color="#2766ac",height=.55);ax.barh(i,343-n,left=n,color="#e3e9f0",height=.55)
        ax.text(n+3,i,str(n),va="center",fontsize=10,fontweight="bold")
    ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(0,365);ax.set_xticks([0,100,200,300])
    ax.set_xlabel("Enregistrements connus · total 343");ax.set_title(title,fontsize=11,pad=14)
    ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0);ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(OUT/"comparison.png");plt.close(fig)
def picture(im):
    im=im.copy().convert("RGB");im.thumbnail((210,145))
    b=io.BytesIO();im.save(b,format="JPEG",quality=82)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()

for page in u["pages"]:
    with Image.open(page["image"]) as image:
        assert "image-pixels:"+_pixel_sha(image.convert("RGB"))==page["group"],page["image"]
union_examples=[]
for page in u["pages"]:
    matches={q["gt_index"]:q for q in page["naming"]["raw"]["union"]["matches"]}
    byprop={q["proposal_index"]:q for q in page["proposals"]["raw"]}
    for j in page["coverage"]["yolo_only"]:
        m=matches.get(j)
        if m is None:continue
        prop=byprop[m["proposal_index"]];gt=page["gt"][j]
        if prop["source"]!="yolo":continue
        union_examples.append((names[gt["label"]],page,j,prop,m))
union_examples.sort(key=lambda q:(q[0],q[1]["group"],q[2]))
seen=set();tiles=[]
for name,page,j,prop,match in union_examples:
    if name in seen:continue
    seen.add(name)
    image=Image.open(page["image"]).convert("RGB")
    x,y,w,h=page["gt"][j]["bbox"];a,b,c,d=max(0,int(x-20)),max(0,int(y-20)),min(image.width,int(x+w+20)),min(image.height,int(y+h+20))
    crop=image.crop((a,b,c,d));draw=ImageDraw.Draw(crop)
    draw.rectangle((x-a,y-b,x+w-a,y+h-b),outline="#18804c",width=2)
    xx,yy,ww,hh=prop["bbox"];draw.rectangle((xx-a,yy-b,xx+ww-a,yy+hh-b),outline="#c87616",width=2)
    tiles.append(f'<figure><img src="{picture(crop)}" alt="Annotation {esc(name)} et boîte YOLO"/><figcaption><b>{esc(name)}</b> · YOLO propose {esc(names[prop["prediction"]])}<br>IoU {match["iou"]:.2f} · {"bon nom" if match["correct"] else "nom incorrect"}</figcaption></figure>')
    if len(tiles)==12:break

snapshot=ROOT/"backend/training_corpus/snapshots/snapshot-611c734590f272060251"
meta={q["row_id"]:q for q in csv.DictReader((snapshot/"metadata.csv").open())}
fixture_path=ROOT/"backend/training_runs/20261003-recognition-mechanisms-v2/crop-fixture.pt"
assert sha(fixture_path)==read(fixture_path.parent/"crop-manifest.json")["fixture_sha256"]
fixture=torch.load(fixture_path,weights_only=True,map_location="cpu")
fi={q["row_id"]:i for i,q in enumerate(fixture["rows"]) if q["split"]=="train"}
rawmap={q["row_id"]:q for q in base["raw"]}
sections=[];example_count=0
for cl in sorted(classes,key=lambda q:(-q["known"],q["name"])):
    label=cl["label"];e=entries[label]
    meanings=" / ".join(e["meanings"]["fr"][:3] or e["meanings"]["es"][:3]) or "Sens absent du catalogue"
    source=e.get("notice_source") or {}
    refs=f'<a href="{esc(e["gdn_url"])}">GDN : {esc(e["name"])}</a>'
    if source.get("url"):refs+=f' · <a href="{esc(source["url"])}">Notice Tlachia · thème {esc(source.get("theme",""))}</a>'
    wrong=[q for q in base["mask"] if q["label"]==label and q["prediction"]!=label]
    wrong.sort(key=lambda q:q["row_id"])
    samples=[]
    if cl["known"]>=5:
        for q in wrong[:2]:
            rr=meta[q["row_id"]];assert rr["dataset_split"]=="train"
            assert sha(snapshot/rr["image_path"])==rr["output_sha256"],q["row_id"]
            raw=Image.open(snapshot/rr["image_path"]).convert("RGB")
            i=fi[q["row_id"]];rgb=fixture["rgb"][i].numpy().copy();mask=fixture["masks"][i].numpy();rgb[~mask]=255
            top=" → ".join(esc(names[k]) for k in q["top3"])
            samples.append(f'<div class="sample"><div><img src="{picture(raw)}" alt="Boîte métier de {esc(cl["name"])}"/><span>Boîte métier fournie</span></div><div><img src="{picture(Image.fromarray(rgb))}" alt="Masque SAM de {esc(cl["name"])}"/><span>Masque SAM</span></div><p>Top 3 : {top}<br>Bon nom {"présent" if label in q["top3"] else "absent"} · repli historique {"oui" if q["fallback"] else "non"}</p></div>')
            example_count+=1
    conf=" ; ".join(f'{esc(v["name"])} ({v["count"]})' for v in cl["confusions"]) or "Aucune confusion sur les boîtes associées"
    conditional="—" if cl["conditional"] is None else f'{100*cl["conditional"]:.1f} %'
    sections.append(f'<details><summary><b>{esc(cl["name"])}</b> · {cl["correct"]}/{cl["known"]} retrouvés et bien nommés · {cl["located"]} localisés</summary><p>Sens CEN : {esc(meanings)}<br>{refs}</p><p>Noms corrects : {cl["correct"]}/{cl["located"]} ({conditional}) · top 3 : {cl["top3"]}/{cl["located"]} · marge ciblée : {cl["hard_correct"]}/{cl["located"]}</p><p>Confusions observées : {conf}</p>{"".join(samples)}<p class="note">{"Effectif inférieur à 5 : taux très instable." if cl["known"]<5 else "Exemples descriptifs du lot exploratoire."}</p></details>')

def table(headers,rows):
    return "<table><thead><tr>"+"".join("<th>"+esc(h)+"</th>" for h in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+str(x)+"</td>" for x in row)+"</tr>" for row in rows)+"</tbody></table>"
geom_table=table(["Propositions","Boîtes connues","Sorties","Doublons associés aux annotations","Sans correspondance dans l’export"],
    [[labels[i],str(u["geometry"][k]["counts"]["matched"])+"/343",u["geometry"][k]["counts"]["accepted"],u["geometry"][k]["counts"]["duplicate_predictions"],u["geometry"][k]["counts"]["unmatched_predictions"]] for i,k in enumerate(["sam","yolo","union"])])
name_table=table(["Lecture","SAM","YOLO partiel","Union"],[
    ["YOLO rectangle, SAM masque"]+[f'{u["naming"]["raw"][k]["counts"]["correct"]}/343' for k in ["sam","yolo","union"]],
    ["Masques pour les deux sources"]+[f'{u["naming"]["mask"][k]["counts"]["correct"]}/343' for k in ["sam","yolo","union"]]])
rec_table=table(["Reconnaissance","Boîte métier · top 1","SAM masque · top 1","SAM masque · top 3"],[
    [title,f'{r["families"][k]["raw"]["correct"]}/343',f'{r["families"][k]["mask"]["correct"]}/254',f'{r["families"][k]["mask"]["top3_correct"]}/254']
    for k,title in [("best_prior","Meilleur précédent"),("control","Contrôle · 200 pas"),("hard_margin","Marge ciblée · 200 pas")]])
pairs_table=table(["Partition","Paires minées dans son entraînement","Exemples minés","Termes positifs pendant le fit"],[
    [q["fold"],"<br>".join(" ↔ ".join(map(esc,z["names"])) for z in q["pairs"]) or "Aucune",q["mined_rows"],q["positive_terms"]] for q in mine])
pairconf=collections.Counter((q["label"],q["prediction"]) for q in base["mask"] if q["label"]!=q["prediction"])
conf_table=table(["Nom attendu","Nom proposé","Erreurs"],[[esc(names[a]),esc(names[b]),n] for (a,b),n in pairconf.most_common(12)])
frequent=sorted([q for q in classes if q["known"]>=5],key=lambda q:(q["joint"],-q["known"],q["name"]))
class_table=table(["Classe · au moins 5 annotations","Connues","SAM localisées","Bon nom /localisées","Bon nom /connues","Bon nom dans top 3"],[
    [esc(q["name"]),q["known"],q["located"],f'{q["correct"]}/{q["located"]}',f'{q["correct"]}/{q["known"]}',f'{q["top3"]}/{q["located"]}'] for q in frequent])
plot="data:image/png;base64,"+base64.b64encode((OUT/"comparison.png").read_bytes()).decode()
document=f'''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM + YOLO et confusions — 5 octobre 2026</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#1a2634;background:#f7f9fc;margin:0}}main{{max-width:1120px;margin:auto;padding:30px 22px}}h1{{line-height:1.2}}h2{{margin-top:36px}}a{{color:#175fac}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:16px 0}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid #d9e1eb;vertical-align:top}}th{{background:#eaf0f7}}.scroll{{overflow:auto}}.note{{color:#536273;font-size:14px}}.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px}}figure{{background:white;margin:0;padding:14px}}figure img{{height:145px;max-width:100%;object-fit:contain}}figcaption{{font-size:14px}}details{{background:white;padding:13px 16px;margin:9px 0;border-radius:7px}}summary{{cursor:pointer}}.sample{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0;padding:12px;background:#eef3f8}}.sample img{{height:120px;width:100%;object-fit:contain}}.sample span{{display:block;text-align:center;font-size:13px}}.sample p{{grid-column:1/-1;margin:0}}.plot{{width:100%;height:auto}}@media(max-width:600px){{main{{padding:20px 12px}}th,td{{padding:8px}}.sample{{grid-template-columns:1fr 1fr}}}}</style><main>
<h1>SAM + YOLO : une meilleure couverture, davantage de propositions à vérifier</h1>
<p>Deux essais terminés : concaténation des propositions sauvegardées et adaptation ciblée de la reconnaissance. Aucun modèle actif modifié. Lot de 17 planches déjà étudiées, annotations incomplètes, une graine.</p>
<img class="plot" src="{plot}" alt="Comparaison des boîtes et des bons noms SAM, YOLO et union"/>
<h2>1 · Complémentarité des propositions</h2><div class="scroll">{geom_table}</div>
<p>SAM et YOLO associent tous deux 169 annotations. SAM en associe seul 85 ; YOLO seul 50 ; 39 ne sont associés par aucun des deux lorsqu’ils sont évalués séparément. Leur union d’ensembles couvre 304 enregistrements. Le matching un-à-un refait sur toutes les propositions en associe 305 : certaines affectations changent avec les propositions concurrentes.</p>
<p>Les 1 604 propositions sont conservées avec leur source, sans NMS. Les doublons comptés correspondent aux sorties supplémentaires recouvrant des annotations connues ; ils ne mesurent pas tous les doublons possibles. Les sorties sans correspondance peuvent être du fond, des doublons ou des glyphes non annotés. <b>Aucune précision globale ni F1 ne peut être déduite de cet export.</b></p>
<h2>Bonne boîte et bon nom, avec le même classifieur</h2><div class="scroll">{name_table}</div>
<p>Filtre sémantique historique 0,35 et seuils YOLO déjà gelés. Les deux vues prévues sont publiées. Sur la vue rectangle, l’union gagne {transitions["raw"]["gained_correct"]} résultats corrects et en perd {transitions["raw"]["lost_correct"]}, soit +33 nets ; sur les masques, +23 nets. Les affectations sont géométriques, indépendantes du nom.</p>
<p class="note">Ce comparateur utilise les modèles de reconnaissance du protocole YOLO, entraînés avec leurs pages de calibration exclues. Son SAM de référence obtient 177/343. Il diffère du meilleur précédent 180/343 et du DINO adapté 182/343 : ne pas attribuer leurs écarts à la fusion seule.</p>
<h2>Exemples retrouvés par YOLO et manqués par SAM</h2><p class="note">Vert : annotation connue. Orange : boîte YOLO retenue dans l’union. Exemples choisis par nom puis identifiant, pas par confiance. Aucun ajout d’annotation.</p><div class="tiles">{"".join(tiles)}</div>
<h2>3 · Essai ciblé sur les confusions</h2><div class="scroll">{rec_table}</div>
<p>Départ du meilleur précédent, DINO figé et prototypes fixes. Cinq partitions, deux bras de 200 pas, mêmes batches et expositions. Le bras ciblé ajoute une marge cosinus de 0,05 (poids 1) entre classes difficiles minées dans son entraînement uniquement. CE équilibrée, préservation historique KL et cohérence boîte métier/masque restent identiques. Toutes les paires et tous les calculs sont aussi présents dans le contrôle, avec poids de marge nul.</p>
<p><b>La marge ciblée ne récupère aucun nom supplémentaire sur les masques : 180/254 dans les deux bras.</b> Le top 3 retrouve le bon nom dans 213/254 cas, dont 33 erreurs du top 1 ; il reste 41 cas où les trois propositions sont incorrectes. Ce top 3 ne retrouve pas les 89 enregistrements manqués par SAM.</p>
<div class="scroll">{pairs_table}</div>
<h2>Confusions et performances par classe</h2><p>Cette partie utilise le meilleur précédent sur les mêmes 254 masques SAM, sans fusion. Chaque effectif est affiché. Les classes absentes du lot n’ont pas de mesure de performance.</p><div class="scroll">{conf_table}{class_table}</div>
<h2>Inspecter les classes et les références CEN</h2><p>Ouvrir une classe pour voir ses résultats, les confusions et des exemples. Les images « boîte métier fournie » servent au diagnostic ; elles ne sont pas fournies à l’inférence. Les sens CEN et liens Tlachia/GDN documentent les noms. Ils ne sont utilisés ni dans la mine, ni dans la perte, ni dans les prédictions. Plusieurs sens ou variantes peuvent coexister.</p>
{"".join(sections)}
<h2>Interprétation et limites</h2><p>La complémentarité géométrique est réelle sur les annotations connues de ce lot. La concaténation brute multiplie cependant les propositions à vérifier et peut modifier des affectations déjà correctes. Elle constitue un diagnostic, pas une configuration à déployer telle quelle. L’essai de marge ciblée n’améliore pas le résultat joint ; le top 3 apporte surtout une piste d’assistance à la correction.</p>
<p>343 enregistrements d’annotation, pas une garantie de 343 objets physiques distincts : un quasi doublon a été identifié. Même lot réutilisé, historique de préentraînement partiellement connu, annotations incomplètes, une graine et 200 pas pour l’adaptation. Aucun nouveau lot indépendant, aucun score dev/test, aucune activation. Les hashes des sources, données, checkpoints et fichiers actifs sont vérifiés.</p>
<p><a href="summary.json">Données détaillées</a></p></main></html>'''
(OUT/"index.html").write_text(document,encoding="utf-8")
summary["report_sha256"]={"index.html":sha(OUT/"index.html"),"comparison.png":sha(OUT/"comparison.png")}
summary["example_count"]=example_count
write(OUT/"summary.json",summary)
print(json.dumps(dict(report=str(OUT),bytes=len(document.encode()),examples=example_count,transitions={k:{x:v[x] for x in ["gained_correct","lost_correct","correct_sources"]} for k,v in transitions.items()},confusions=[dict(expected=names[a],predicted=names[b],count=n) for (a,b),n in pairconf.most_common(8)],frequent_classes=frequent),ensure_ascii=False))

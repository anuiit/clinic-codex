"""Render the frozen cross-source deduplication experiment and its changed cases."""
import base64, collections, hashlib, html, io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"backend/training_runs/cen-semantic-report-deps"), str(ROOT)]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scripts import research_union_dedup as exp
from scripts.import_annotation_archive import _pixel_sha

OUT = ROOT/"reports/union-dedup-20261005-v1/report"
OUT.mkdir(parents=True, exist_ok=True)
result = exp.u.read(exp.RUN/"results.json")
saved = exp.u.read(exp.INPUT)
exp.u.verify(result["protocol"])
exp.u.verify(saved["protocol"])
names = exp.u.read(ROOT/"backend/codex_model/config.json")["class_names"]
pages = {p["group"]:p for p in saved["pages"]}
def esc(x): return html.escape(str(x))
def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def table(headers, rows):
    return "<div class=scroll><table><tr>"+"".join("<th>"+esc(x)+"</th>" for x in headers)+"</tr>"+"".join(
        "<tr>"+"".join("<td>"+esc(x)+"</td>" for x in row)+"</tr>" for row in rows)+"</table></div>"
titles = {"union":"Union brute", "consensus":"Même nom · IoU ≥ 0,70", "spatial":"Géométrie · IoU ≥ 0,85"}
def metric_table(view):
    rows=[]
    for method, r in result["summary"][view].items():
        c,d=r["counts"],r["delta_union"]
        rows.append([titles[method],c["generated"],c["matched"],
            f'{c["correct"]}/343 ({100*c["correct"]/343:.1f} %)',
            c["duplicate_predictions"],f'+{d["gained_correct"]} / −{d["lost_correct"]}'])
    return table(["Règle","Propositions","Boîtes connues retrouvées","Bonne boîte + bon nom",
                  "Doublons sur les annotations connues","Gains / pertes de résultats corrects"],rows)

fig,axes=plt.subplots(1,3,figsize=(14,3.7),dpi=160)
for ax,key,title in zip(axes,["generated","matched","correct"],
                        ["Propositions conservées","Boîtes connues retrouvées","Bonne boîte + bon nom"]):
    values=[result["summary"]["raw"][m]["counts"][key] for m in titles]
    ax.barh(range(3),values,color=["#7b8797","#e69332","#2376ad"],height=.55)
    ax.set_yticks(range(3),["Union","Même nom","Géométrie"])
    ax.invert_yaxis();ax.set_xlim(0,max(values)*1.2)
    for i,n in enumerate(values):ax.text(n+max(values)*.025,i,str(n),va="center",fontweight="bold")
    ax.set_title(title,fontsize=11,pad=16)
    ax.spines[["top","right","left"]].set_visible(False);ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    ax.set_xlabel("Nombre · vue rectangle YOLO",fontsize=9)
fig.tight_layout();fig.savefig(OUT/"comparison.png");plt.close(fig)
def picture(im):
    b=io.BytesIO();im.save(b,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

# Publish every transition; image tiles are deduplicated by page and GT only.
changes=[]
for p in result["pages"]:
    old=pages[p["group"]]
    for view,methods in p["views"].items():
        rows={r["proposal_index"]:r for r in old["proposals"][view]}
        before={m["gt_index"]:m for m in methods["union"]["score"]["matches"]}
        for method in exp.POLICIES:
            current=methods[method]
            after={m["gt_index"]:m for m in current["score"]["matches"]}
            for kind in ["lost_correct","gained_correct"]:
                for i in current["delta_union"][kind]:
                    b=before.get(i);a=after.get(i)
                    removed=next((r for r in current["suppressed"] if b and r["proposal_index"]==b["proposal_index"]),None)
                    keeper=rows[removed["keeper_index"]] if removed else None
                    assigned_keeper=next((m for m in after.values() if keeper and m["proposal_index"]==keeper["proposal_index"]),None)
                    other_gt=assigned_keeper["gt_index"] if assigned_keeper else None
                    changes.append(dict(group=p["group"],view=view,method=method,kind=kind,
                        gt_index=i,expected_name=names[old["gt"][i]["label"]],
                        before=rows[b["proposal_index"]] if b else None,
                        after=rows[a["proposal_index"]] if a else None,
                        suppression=removed,keeper=keeper,keeper_assigned_gt=other_gt,
                        keeper_assigned_name=names[old["gt"][other_gt]["label"]] if other_gt is not None else None,
                        truth_pair_iou=exp.g.box_iou(exp.g.box(old["gt"][i]["bbox"])["box"],
                            exp.g.box(old["gt"][other_gt]["bbox"])["box"]) if other_gt is not None else None))
tiles=[];seen=set()
for e in changes:
    identity=(e["group"],e["gt_index"])
    if identity in seen:continue
    seen.add(identity);p=pages[e["group"]]
    with Image.open(p["image"]) as im:
        im=im.convert("RGB")
        assert "image-pixels:"+_pixel_sha(im)==p["group"]
        boxes=[p["gt"][e["gt_index"]]["bbox"]]+[r["bbox"] for r in [e["before"],e["after"],e["keeper"]] if r]
        a=max(0,int(min(b[0] for b in boxes))-18);b=max(0,int(min(b[1] for b in boxes))-18)
        c=min(im.width,int(max(z[0]+z[2] for z in boxes))+18);d=min(im.height,int(max(z[1]+z[3] for z in boxes))+18)
        scale=min(3.,260/max(c-a,d-b))
        base=im.crop((a,b,c,d)).resize((max(1,int((c-a)*scale)),max(1,int((d-b)*scale))))
    def panel(layers):
        z=base.copy();draw=ImageDraw.Draw(z)
        for bbox,color in layers:
            x,y,w,h=bbox;draw.rectangle(((x-a)*scale,(y-b)*scale,(x+w-a)*scale,(y+h-b)*scale),outline=color,width=2)
        return picture(z)
    gt=p["gt"][e["gt_index"]]
    truthlayers=[(gt["bbox"],"#168449")]
    if e["keeper_assigned_gt"] is not None and e["keeper_assigned_gt"]!=e["gt_index"]:
        truthlayers.append((p["gt"][e["keeper_assigned_gt"]]["bbox"],"#923dba"))
    pre=[(r["bbox"],color) for r,color in [(e["before"],"#e58b16"),(e["keeper"],"#1479c7")] if r]
    post=[(r["bbox"],color) for r,color in [(e["keeper"],"#1479c7"),(e["after"],"#cc3030")] if r]
    explanation=""
    if e["keeper_assigned_gt"] is not None and e["keeper_assigned_gt"]!=e["gt_index"]:
        j=e["keeper_assigned_gt"]
        explanation=f'La boîte SAM portant le bon nom subsiste. Elle est attribuée à l’annotation {j}, de nom {esc(e["keeper_assigned_name"])} ; le recouvrement entre les deux annotations est {e["truth_pair_iou"]:.3f}. Le matching ne peut attribuer une sortie à deux annotations. Ces annotations proches doivent être arbitrées par le métier ; aucune n’est supprimée ici.'
    else:
        explanation="La suppression change la boîte associée et son nom. Ce changement est mesuré après une nouvelle association géométrique, sans modifier le classifieur."
    caption="Perte comptée" if e["kind"]=="lost_correct" else "Gain compté"
    related=[x for x in changes if (x["group"],x["gt_index"])==identity]
    rows=[[x["view"],titles[x["method"]],"perte" if x["kind"]=="lost_correct" else "gain",
           names[x["before"]["prediction"]] if x["before"] else "aucun",
           names[x["after"]["prediction"]] if x["after"] else "aucune boîte associée"] for x in related]
    tiles.append(f'<article><h3>{caption} · {esc(e["expected_name"])}</h3><div class=panels>'
        f'<figure><img src="{panel(truthlayers)}" alt="Annotations proches de {esc(e["expected_name"])}"><figcaption>Annotations : cible verte, autre annotation violette</figcaption></figure>'
        f'<figure><img src="{panel(pre)}" alt="Boîtes avant suppression pour {esc(e["expected_name"])}"><figcaption>Avant : YOLO orange, SAM bleu</figcaption></figure>'
        f'<figure><img src="{panel(post)}" alt="Boîtes après suppression pour {esc(e["expected_name"])}"><figcaption>Après : SAM bleu, boîte associée à la cible rouge</figcaption></figure></div><p>{explanation}</p>'
        +table(["Vue","Règle","Transition","Nom avant","Nom après"],rows)+"</article>")

pair_flags=[]
for e in changes:
    if e["kind"]=="lost_correct" and e["keeper_assigned_gt"] is not None and e["keeper_assigned_gt"]!=e["gt_index"]:
        pair=dict(group=e["group"],indices=sorted([e["gt_index"],e["keeper_assigned_gt"]]),
            name=e["expected_name"],iou=e["truth_pair_iou"],status="Needs human adjudication; not deleted")
        if pair not in pair_flags:pair_flags.append(pair)
assert len(pair_flags)==2 and len(seen)==3
page_rows=[]
for i,p in enumerate(result["pages"],1):
    vals=[]
    for method in titles:
        c=p["views"]["raw"][method]["score"]["counts"]
        vals.append(f'{c["correct"]}/{c["known"]} · {c["generated"]} propositions')
    page_rows.append([f'Planche {i} · {p["group"].split(":")[1][:12]}',*vals])
plot=picture(Image.open(OUT/"comparison.png"))
document=f'''<!doctype html><html lang=fr><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Fusion SAM + YOLO : test de dédoublonnage</title><style>
body{{font:16px/1.55 system-ui,sans-serif;background:#f6f8fb;color:#223044;margin:0}}main{{max-width:1150px;margin:auto;padding:28px 22px}}h1{{line-height:1.2}}h2{{margin-top:32px}}a{{color:#1463a0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:16px 0}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d7e0ec;vertical-align:top}}th{{background:#e8eef7}}.plot{{width:100%;height:auto}}article{{background:white;padding:20px;margin:18px 0;border-radius:8px}}.panels{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}figure img{{height:230px;width:100%;object-fit:contain}}figcaption,.note{{font-size:14px;color:#546378}}.callout{{padding:16px;background:#e9f1fa;border-left:4px solid #2376ad}}summary{{cursor:pointer}}@media(max-width:650px){{main{{padding:20px 12px}}.panels{{grid-template-columns:1fr}}figure img{{height:210px}}}}</style>
<main><h1>Fusion SAM + YOLO : supprimer les doublons apporte peu de gain</h1>
<p>Deux règles fixées avant leur évaluation, mêmes modèles et mêmes 17 planches. Aucun entraînement, aucune modification du modèle actif. Résultats exploratoires sur 343 annotations connues d’un export incomplet.</p>
<div class=callout>La règle géométrique conserve 210/343 résultats corrects et retire 68 propositions sur 1 604, soit 4,2 %. Un résultat est gagné, un autre perdu. La règle « même nom » retire 101 propositions mais tombe à 208/343. Les deux pertes comptées concernent des annotations proches de même nom : une boîte correcte subsiste, attribuée à l’autre annotation.</div>
<img class=plot src="{plot}" alt="Comparaison des propositions, boîtes connues et bons noms pour les trois règles">
<h2>Résultats · SAM masqué et YOLO rectangle</h2>{metric_table("raw")}
<p>Le taux conjoint mesure uniquement les annotations connues. La référence SAM seule de ce comparateur est 177/343 ; l’union brute atteint 210/343. Ce comparateur emploie les têtes du protocole imbriqué YOLO, différentes des essais précédents SAM à 180 ou 182.</p>
<h2>Contrôle · masques pour les deux sources</h2>{metric_table("mask")}
<p>La vue entièrement masquée donne 200/343 avec l’union et 199/343 avec chacune des règles. Le dédoublonnage dépend des noms prédits pour « même nom » ; il supprime donc un nombre différent de propositions selon la vue. La règle géométrique supprime les mêmes 68 boîtes dans les deux vues.</p>
<h2>Comment l’inférence est simulée</h2>
<ol><li>Reprendre toutes les propositions SAM et les propositions YOLO passant leurs seuils de détection déjà gelés.</li>
<li>Conserver les noms et similarités des mêmes têtes de reconnaissance. Le filtre sémantique existant reste à 0,35.</li>
<li>Conserver toutes les boîtes SAM. Supprimer uniquement une proposition YOLO si elle recouvre une SAM : au moins 0,70 et même nom pour « même nom » ; au moins 0,85 pour « géométrie ». Les deux boîtes doivent passer 0,35. Aucun choix guidé par les annotations.</li>
<li>Refaire l’association géométrique un-à-un aux annotations connues, seuil IoU 0,50, puis compter les boîtes et noms corrects. Publier les deux règles et les deux vues, sans sélectionner un seuil sur leurs résultats.</li></ol>
<p>Pas de NMS entre deux SAM ou deux YOLO, pas de moyenne des coordonnées ou des noms. Les petits objets fortement imbriqués échappent au seuil de recouvrement, mais cette règle ne garantit pas la préservation de tous les objets proches.</p>
<h2>Tous les cas qui changent le résultat correct</h2>
<p>Les six transitions des deux vues correspondent à trois emplacements distincts, tous affichés ci-dessous. Les couleurs portent sur les coordonnées des boîtes sauvegardées, pas sur des masques reconstruits.</p>{"".join(tiles)}
<h2>Ce que cet essai permet de conclure</h2>
<p><b>Le dédoublonnage simple ne résout pas le volume des sorties.</b> Après la règle géométrique, il reste 1 219 propositions sans correspondance dans l’export, dont 138 recouvrent des annotations connues. Il reste 1 081 autres sorties non appariées : glyphes omis, fragments, doublons ou fond, sans adjudication possible à partir de cet export.</p>
<p>Les paires proches révèlent une ambiguïté d’association entre enregistrements. Le score compte les annotations de l’export. Le métier doit décider si chacune de ces paires correspond à deux glyphes distincts ou à un doublon avant d’interpréter ces pertes. Ce lot contient des annotations manquantes confirmées et des chevauchements à arbitrer. Aucun nettoyage automatique ni nouveau score sur des annotations corrigées n’a été effectué.</p>
<p>Mon avis : conserver l’union comme source de candidats, garder les alternatives lors d’un regroupement pour revue, et faire valider ces chevauchements avant de supprimer des boîtes automatiquement. Il faut une adjudication métier des sorties pour mesurer la précision et calibrer un rejet de fond. Ce test n’apporte aucune nouvelle preuve d’amélioration du modèle ni de généralisation.</p>
<details><summary>Résultats par planche · vue rectangle</summary>{table(["Planche","Union","Même nom","Géométrie"],page_rows)}</details>
<p class=note>Vérifications : auto-tests du filtre, reproduction exacte des résultats union pour les deux vues, hashes des entrées et poids actifs inchangés, provenance des pixels affichés vérifiée. Aucun passage sur les pages dev/test, aucune activation. Même lot déjà inspecté et une seule graine ; précision globale et F1 non interprétables.</p>
</main></html>'''
(OUT/"index.html").write_text(document,encoding="utf-8")
summary=dict(schema_version="union-dedup-report.v1",summary=result["summary"],changes=changes,
    annotation_pairs_to_review=pair_flags,displayed_locations=len(seen),
    inputs_sha256={str(exp.RUN/"results.json"):sha(exp.RUN/"results.json"),str(exp.INPUT):sha(exp.INPUT),
                  str(Path(__file__)):sha(Path(__file__))},
    artifacts_sha256={name:sha(OUT/name) for name in ["index.html","comparison.png"]},
    verification=dict(experiment_hashes_verified=True,displayed_page_pixels_verified=True,
                      all_changed_correct_locations_displayed=True),
    activated=False,locked_test_scored=False)
exp.u.write(OUT/"summary.json",summary)
print(dict(out=str(OUT),displayed_locations=len(seen),transitions=len(changes),annotation_pairs_to_review=pair_flags))

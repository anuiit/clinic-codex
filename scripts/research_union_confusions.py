"""Saved-proposal union and train-only hard-confusion ablation; no active writes."""
import argparse, collections, csv, json, os, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts import research_crop_compatible as p
from scripts import research_full_proposals as geom
ROOT=p.core.ROOT
RUN=ROOT/"backend/training_runs/20261005-union-confusions-v2"
PARTIAL=ROOT/"backend/training_runs/20261004-yolo-partial-label-v2"
SAM=ROOT/"reports/nested-rejection-results-20261004.json"
CEN=ROOT/"reports/cen-semantic-20261004-v2/class-text.json"
SEED,STEPS,MARGIN,PAIR_LIMIT=42,200,.05,10


def read(path): return p.m.read_json(path)
def write(path,value): p.m.write_json(path,value)
def pin(paths): return {str(q):p.m.sha256(q) for q in sorted(set(paths))}
def active(): return list((ROOT/"backend/codex_model").glob("weights/*.pt"))+[ROOT/"backend/codex_model/config.json"]
def verify(protocol): p.verify(protocol)


def register(name,paths,**fields):
    RUN.mkdir(parents=True,exist_ok=True)
    dest=RUN/(name+"-preregistration.json")
    if dest.exists():
        result=read(dest);verify(result);return result
    result=dict(schema_version=name+".v1",input_sha256=pin(paths+[Path(__file__)]+active()),
        seed=SEED,activated=False,locked_test_scored=False,
        scope="Exploratory: same17 previously examined train pages, incomplete annotations; no new blind validation or exhaustive precision/F1",**fields)
    write(dest,result);return result


def safe_score(gt,ignore,rows,threshold=None):
    result=geom.score_page(gt,ignore,rows,threshold)
    c=result["counts"]
    result["metrics"]=dict(known_box_recall=c["matched"]/c["known"],
        known_joint_recall=c["correct"]/c["known"],
        conditional_name_accuracy=c["correct"]/c["matched"] if c["matched"] else None)
    return result


def aggregate(records):
    counts={k:sum(r["counts"][k] for r in records) for k in records[0]["counts"]}
    return dict(counts=counts,known_box_recall=counts["matched"]/counts["known"],
        known_joint_recall=counts["correct"]/counts["known"],
        conditional_name_accuracy=counts["correct"]/counts["matched"] if counts["matched"] else None)


def tagged(sam,yolo):
    rows=[]
    for origin,source in [("sam",sam),("yolo",yolo)]:
        for r in source:
            rows.append(dict(r,source=origin,source_proposal_index=r["proposal_index"],proposal_index=len(rows)))
    return rows


def union():
    paths=[SAM,PARTIAL/"results.json",PARTIAL/"detection-results.json",
        ROOT/"reports/yolo-experiments-final-20261004.json",Path(geom.__file__),
        ROOT/"scripts/evaluate_glyph_benchmark.py"]
    protocol=register("union",paths,optimizer_updates=0,
        detector_cuts="Reuse existing per-fold YOLO pilot calibrated confidence cuts; all frozen SAM proposals",
        naming_cut="Existing current_035 comparator in saved artifacts; inclusive similarity>=0.35",
        pairing="Same nested seed42 recognition heads for both sources; not DINO-tail or the differently fitted best_prior",
        fusion="SAM then YOLO concatenation, retain every proposal with provenance; no NMS, label-guided selection or new threshold",
        association="Existing geometric greedy IoU>=0.5, stable indices, known GT then explicit ignore regions",
        reports="Separate union of independently matched GT sets from actual rematching; no exhaustive precision or F1")
    sam={q["group"]:q for q in read(SAM)["outer_pages"] if q["seed"]==SEED}
    yy=read(PARTIAL/"results.json")
    det={q["group"]:(fit,q) for fit in read(PARTIAL/"detection-results.json")["fits"] for q in fit["outer_pages"]}
    pages=[]
    for y in yy["pages"]:
        s=sam[y["group"]];fit,d=det[y["group"]]
        assert s["fold"]==y["fold"]==fit["fold"]
        assert [(r["bbox"],r["label"]) for r in s["gt"]]==[(r["bbox"],r["label"]) for r in y["gt"]]
        assert [r["bbox"] for r in s["ignore"]]==[r["bbox"] for r in y["ignore"]]
        gt,ign=s["gt"],s["ignore"]
        sr=s["methods"]["gated"]["predictions"]
        cutoff=fit["selection"]["threshold"]
        yr=[q for q in d["predictions"] if q["detector_confidence"]>=cutoff]
        assert [q["proposal_index"] for q in yr]==[q["proposal_index"] for q in y["views"]["mask"]["predictions"]]
        yr=[dict(q,prediction=-1,similarity=1.) for q in yr]
        sm=safe_score(gt,ign,tagged(sr,[]))
        ym=safe_score(gt,ign,tagged([],yr))
        both=tagged(sr,yr);um=safe_score(gt,ign,both)
        ss={m["gt_index"] for m in sm["matches"]};ys={m["gt_index"] for m in ym["matches"]}
        coverage=dict(sam_only=sorted(ss-ys),yolo_only=sorted(ys-ss),both=sorted(ss&ys),
            missed_by_both=sorted(set(range(len(gt)))-(ss|ys)),set_union=len(ss|ys))
        reads={};policies={}
        for view in ["raw","mask"]:
            sy=tagged(sr,y["views"][view]["predictions"])
            policies[view]={}
            for method,rows in [("sam",tagged(sr,[])),("yolo",tagged([],y["views"][view]["predictions"])),("union",sy)]:
                policies[view][method]=safe_score(gt,ign,rows,.35)
            assert policies[view]["yolo"]["counts"]==y["views"][view]["current_035"]["counts"]
            reads[view]=sy
        assert sm["counts"]["matched"]==sum(m in ss for m in range(len(gt)))
        pages.append(dict(group=y["group"],fold=y["fold"],gt=gt,ignore=ign,
            image=d["image"],detector_cut=cutoff,coverage=coverage,
            geometry=dict(sam=sm,yolo=ym,union=um),naming=policies,proposals=reads))
    assert len(pages)==17 and len({q["group"] for q in pages})==17
    totals={k:aggregate([q["geometry"][k] for q in pages]) for k in ["sam","yolo","union"]}
    # Geometry-only readout: naming placeholders are deliberately not published as scores.
    for value in totals.values():
        for key in ["correct","wrong_name","rare_correct"]:value["counts"].pop(key)
        for key in ["known_joint_recall","conditional_name_accuracy"]:value.pop(key)
    for page in pages:
        for value in page["geometry"].values():
            for key in ["correct","wrong_name","rare_correct"]:value["counts"].pop(key)
            for key in ["known_joint_recall","conditional_name_accuracy"]:value["metrics"].pop(key)
            for match in value["matches"]:match.pop("correct")
    assert (totals["sam"]["counts"]["known"],totals["sam"]["counts"]["matched"],totals["yolo"]["counts"]["matched"])==(343,254,219)
    naming={v:{k:aggregate([q["naming"][v][k] for q in pages]) for k in ["sam","yolo","union"]} for v in ["raw","mask"]}
    assert naming["mask"]["sam"]["counts"]["correct"]==177
    verify(protocol)
    out=dict(protocol=protocol,geometry=totals,naming=naming,pages=pages,
        coverage={k:sum(len(q["coverage"][k]) for q in pages) for k in ["sam_only","yolo_only","both","missed_by_both"]},
        matched_set_union=sum(q["coverage"]["set_union"] for q in pages),
        optimizer_updates=0,activated=False,locked_test_scored=False,
        verification=dict(inputs_and_active_files_unchanged=True,same_page_gt_order=True,
            yolo_saved_recognition_counts_reproduced=True,sam_reference177_reproduced=True))
    write(RUN/"union-results.json",out)
    print("UNION",json.dumps(dict(geometry={k:q["counts"] for k,q in totals.items()},coverage=out["coverage"],naming={v:{k:q["counts"]["correct"] for k,q in a.items()} for v,a in naming.items()})),flush=True)


def hard_values(scores,y,pairs):
    # ponytail: fixed ten class pairs; expand only after evidence on fresh pages.
    eligible=pairs[y].any(1)
    negative=scores.masked_fill(~pairs[y],-2.).max(1).values
    values=F.relu(MARGIN+negative-scores.gather(1,y[:,None]).squeeze(1))
    return values,eligible


def check():
    pairs=torch.zeros(3,3,dtype=torch.bool);pairs[0,1]=pairs[1,0]=True
    scores=torch.tensor([[.2,.3,.8],[.9,.2,.1],[.1,.2,.4]],requires_grad=True);y=torch.tensor([0,1,2])
    values,eligible=hard_values(scores,y,pairs)
    assert eligible.tolist()==[True,True,False]
    torch.testing.assert_close(values[:2],torch.tensor([.15,.75]))
    values[eligible].mean().backward()
    assert scores.grad[0,0]<0 and scores.grad[0,1]>0 and scores.grad[0,2]==0
    assert scores.grad[2].abs().sum()==0
    rows=tagged([dict(bbox=[0,0,10,10],prediction=0,proposal_index=7,similarity=.7)],
                [dict(bbox=[30,0,10,10],prediction=1,proposal_index=7,similarity=.7)])
    assert [q["proposal_index"] for q in rows]==[0,1] and rows[1]["source"]=="yolo"
    gt=[dict(bbox=[0,0,10,10],label=0,natural_support=5),dict(bbox=[30,0,10,10],label=1,natural_support=5)]
    assert safe_score(gt,[],rows)["counts"]["correct"]==2
    return dict(margin_gradients_passed=True,ineligible_rows_zero_gradient=True,union_provenance_and_matching_passed=True)


def data():
    d=p.base.setup();d["fixture"]=p.m.load_fixture(d);return d


@torch.no_grad()
def mine(d,model,train):
    ids=[i for i in train if i>=d["n"]]
    scores=model(d["features"][ids])@d["old"].T
    labels=d["labels"][ids]
    wrong=scores.clone();wrong.scatter_(1,labels[:,None],-2.)
    wscore,negative=wrong.max(1)
    difficulty=F.relu(MARGIN+wscore-scores.gather(1,labels[:,None]).squeeze(1))
    mass=collections.Counter();support=collections.Counter()
    for y,n,v in zip(labels.tolist(),negative.tolist(),difficulty.tolist()):
        key=tuple(sorted((y,n)));support[key]+=1;mass[key]+=v
    chosen=sorted([k for k,v in mass.items() if v>0],key=lambda k:(-mass[k],k))[:PAIR_LIMIT]
    matrix=torch.zeros(286,286,dtype=torch.bool,device="cuda")
    for a,b in chosen:matrix[a,b]=matrix[b,a]=True
    return matrix,dict(rows=ids,pairs=[dict(labels=list(k),initial_hinge_mass=mass[k],rows_with_nearest_negative=support[k]) for k in chosen],
        positive_hinges=int((difficulty>0).sum()),groups=sorted({d["groups"][i] for i in ids}))


@torch.no_grad()
def evaluate(d,model,held,raw):
    pixels=collections.defaultdict(set)
    for i in raw:pixels[int(d["labels"][i])].add(d["pixels"][i])
    counts=torch.tensor([len(pixels[c]) for c in range(286)],device="cuda")
    rawids=[i for i in range(d["n"]) if d["splits"][i]=="train" and d["groups"][i] in held]
    masks=[d["n"]+r["runtime_index"] for r in d["fixture"]["rows"] if r["split"]=="train" and r["source_group"] in held]
    output={}
    for view,ids in [("raw",rawids),("mask",masks)]:
        scores=model(d["features"][ids])@d["old"].T;old=d["teacher"][ids]@d["old"].T
        hist=old.argmax(1);fallback=(counts[hist]>=1)&(counts[hist]<=4)
        gated=torch.where(fallback[:,None],old,scores)
        val,top=gated.topk(3,dim=1)
        output[view]=[dict(index=i,row_id=d["objects"][i],group=d["groups"][i],label=int(d["labels"][i]),
            prediction=int(top[j,0]),ungated_prediction=int(scores[j].argmax()),historical_prediction=int(hist[j]),
            fallback=bool(fallback[j]),natural_support=int(counts[int(d["labels"][i])]),
            top3=top[j].tolist(),top3_scores=val[j].tolist()) for j,i in enumerate(ids)]
    return output


def fit(d,fold,held,method,protocol):
    dest=RUN/f"{method}_fold{fold}.json"
    if dest.exists():verify(protocol);return read(dest)
    raw,train,lookup,anchor,excluded=p.scope(d,held)
    cp=torch.load(p.RUN/f"paired_fold{fold}_seed{SEED}_step1000.pt",map_location="cpu",weights_only=True)
    assert set(cp["train_ids"])==set(train)
    model=p.core.ProjectionHead().cuda();model.load_state_dict(cp["projection"]);model.eval()
    pairs,mining=mine(d,model,train)
    assert set(mining["rows"])<=set(train) and not set(mining["groups"])&set(excluded)
    baseline=evaluate(d,model,held,raw)
    plan=p.core.batches(d["labels"],train,d["n"],SEED,1000+STEPS)[1000:]
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model.train();history=[];exposure=collections.Counter();extra=collections.Counter();started=time.monotonic()
    for step,ids in enumerate(plan,1):
        y=d["labels"][ids];scores=model(d["features"][ids])@d["old"].T
        ce,_=p.class_mean(F.cross_entropy(scores/.1,y,reduction="none"),y)
        replay=[j for j,i in enumerate(ids) if i in anchor]
        kl=ce*0
        if replay:
            target=(d["teacher"][[ids[j] for j in replay]]@d["old"].T/.2).detach()
            v=F.kl_div(F.log_softmax(scores[replay]/.2,1),F.softmax(target,1),reduction="none").sum(1)
            kl,_=p.class_mean(v,y[replay])
        hv,eligible=hard_values(scores,y,pairs)
        hard=hv.sum()*0
        if eligible.any():hard,_=p.class_mean(hv[eligible],y[eligible])
        pairids=[i for i in ids if i in lookup]
        pair=ce*0
        if pairids:
            model.eval()
            a=model(d["features"][[lookup[i] for i in pairids]]);b=model(d["features"][pairids])
            pair=(1-(a*b).sum(1)).mean();model.train()
            extra.update(pairids);extra.update(lookup[i] for i in pairids)
        loss=ce+4*kl+pair+(1. if method=="hard_margin" else 0.)*hard
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        assert torch.isfinite(grad);opt.step();exposure.update(ids)
        history.append(dict(step=step,ce=float(ce.detach()),kl=float(kl.detach()),pair=float(pair.detach()),
            hard=float(hard.detach()),eligible_terms=int(eligible.sum()),positive_terms=int((hv[eligible]>0).sum())))
    model.eval();evaluation=evaluate(d,model,held,raw)
    checkpoint=RUN/f"{method}_fold{fold}.pt"
    torch.save(dict(projection=model.state_dict(),prototypes=d["old"].cpu(),train_ids=train,fold=fold,seed=SEED,step=STEPS),checkpoint)
    assert set(exposure)<=set(train) and {int(d["labels"][i]) for i in exposure}==set(range(286))
    out=dict(fold=fold,seed=SEED,method=method,held=held,excluded=excluded,baseline=baseline,evaluation=evaluation,
        mining=mining,history=history,exposure=dict(exposure),extra_pair_exposure=dict(extra),
        batch_sha256=p.m.digest(np.array([i for batch in plan for i in batch],dtype=np.int64)),
        checkpoint_sha256=p.m.sha256(checkpoint),elapsed_seconds=time.monotonic()-started,
        initial_state_sha256=p.m.digest(np.concatenate([v.numpy().ravel() for v in initial.values()])),
        max_weight_change=max(float((model.state_dict()[k].detach().cpu()-v).abs().max()) for k,v in initial.items()))
    write(dest,out);verify(protocol)
    print("CONFUSIONS",fold,method,"pairs",len(mining["pairs"]),"raw/mask",[sum(r["prediction"]==r["label"] for r in evaluation[v]) for v in ["raw","mask"]],flush=True)
    return out


def train():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d=data();torch.use_deterministic_algorithms(True)
    paths=d["extra_paths"]+[Path(p.__file__),Path(p.base.__file__),Path(p.m.__file__),Path(p.core.__file__),CEN,
        ROOT/"reports/rare-readout-results-20261004.json"]
    paths+=[p.RUN/f"paired_fold{fold}_seed{SEED}_step1000.pt" for fold in range(5)]
    protocol=register("confusions",paths+[Path(q) for q in d["before"]],steps=STEPS,folds=5,optimizer_updates=2000,
        initialization="Existing best paired1000 fold checkpoint, frozen original DINO, fixed historical prototypes",
        optimizer=dict(name="AdamW",lr=1e-5,weight_decay=1e-4,clip_norm=1),
        control="Same mined pairs, forwards, batches and exposures; only hard loss coefficient0 vs1 differs",
        losses=dict(ce_temperature=.1,historical_correct_KL_temperature=.2,KL_multiplier=4,raw_mask_cosine_multiplier=1,
            hard="class-balanced hinge(margin+best selected wrong cosine-true cosine)",hard_margin=MARGIN,hard_lambda=1),
        mining="Own-fit SAM masked objects after source-component/pixel exclusions; nearest wrong class per object, sum initial hinge by unordered pair; top10 positive masses, class-id ties; no held scores or labels",
        primary="Fixed final step200, one seed; no checkpoint, thresholds or pairs selected from held outcomes",
        CEN="Descriptive catalogue only; not used in mining, loss or inference",
        localization="Same254 existing SAM associations on343 saved business crops; no new detector",
        checks=check())
    entries=[]
    for fold,held in enumerate(p.m.folds()):
        arms=[fit(d,fold,held,method,protocol) for method in ["control","hard_margin"]]
        a,b=arms
        assert a["mining"]==b["mining"] and a["initial_state_sha256"]==b["initial_state_sha256"]
        assert a["batch_sha256"]==b["batch_sha256"] and a["exposure"]==b["exposure"] and a["extra_pair_exposure"]==b["extra_pair_exposure"]
        assert a["baseline"]==b["baseline"] and a["history"][0]==b["history"][0]
        entries+=arms;write(RUN/"confusions-progress.json",dict(folds=entries))
    from scripts.summarize_recognition_mechanisms import classification
    families={}
    for family,method,field in [("best_prior","control","baseline"),("control","control","evaluation"),("hard_margin","hard_margin","evaluation")]:
        rows={v:[r for e in entries if e["method"]==method for r in e[field][v]] for v in ["raw","mask"]}
        assert len(rows["raw"])==343 and len(rows["mask"])==254
        assert len({r["row_id"] for r in rows["raw"]})==343 and len({r["row_id"] for r in rows["mask"]})==254
        families[family]={v:dict(classification(rows[v]),top3_correct=sum(r["label"] in r["top3"] for r in rows[v])) for v in ["raw","mask"]}
        families[family]["known_joint"]=families[family]["mask"]["correct"]/343
    assert families["best_prior"]["raw"]["correct"]==209 and families["best_prior"]["mask"]["correct"]==180
    verify(protocol)
    out=dict(protocol=protocol,families=families,folds=entries,optimizer_updates=2000,activated=False,locked_test_scored=False,
        verification=dict(all_inputs_and_active_weights_unchanged=True,paired_batches_and_exposures_equal=True,
            train_only_mining=True,first_loss_terms_identical=True,all286_classes_supervised=True,baseline209_180_reproduced=True))
    write(RUN/"confusions-results.json",out);print("FAMILIES",json.dumps(families),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["check","union","train"])
    phase=parser.parse_args().phase
    if phase=="check":print(json.dumps(check()))
    elif phase=="union":union()
    else:train()

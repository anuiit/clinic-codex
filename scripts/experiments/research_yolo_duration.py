"""Adaptive duration control: same frozen YOLO comparison, fixed1000 updates."""
import argparse,copy,random
from pathlib import Path
from scripts.experiments import research_synthetic_detector as s
from scripts.experiments import research_yolo_recognition_long as r
SOURCE=s.RUN
RUN=s.ROOT/'backend/training_runs/20261004-yolo-synthetic-1000-v1'
STEPS=1000

def prepare():
    assert not RUN.exists();RUN.mkdir()
    old=s.read(SOURCE/'final-preregistration.json');s.verify(old)
    assert (SOURCE/'detection-results.json').exists() and (SOURCE/'recognition/results.json').exists()
    data=s.read(SOURCE/'dataset.json')
    for sp in data['splits']:
        real_rng=random.Random(20261004+sp['fold']);syn_rng=random.Random(42000+sp['fold'])
        real=[real_rng.sample(range(len(sp['real'])),2) for _ in range(STEPS)]
        synth=[syn_rng.sample(range(len(sp['synth'])),2) for _ in range(STEPS)]
        assert real[:200]==sp['real_plan'] and synth[:200]==sp['synth_plan']
        sp['real_plan']=real;sp['synth_plan']=synth
    s.write(RUN/'dataset.json',data)
    p=copy.deepcopy(old);p.update(steps=STEPS,optimizer_updates=10*STEPS,
        selection='Fixed final update1000, no checkpoint selection. Same per arm/fold calibration3 pages and grid, maximize mean calibration detection F1 at IoU>=.5; ties lower threshold; save before outer readout.',
        duration_control=dict(adaptive=True,source=str(SOURCE),trigger='Real losses still decreasing at200 updates; completed pilot below SAM. Test fixed1000 for both arms, not select winning duration on outer scores. Publish200 and1000, same initialization and exact first200 sample prefix. LR cosine now spans1000, other settings unchanged; this changes duration and schedule horizon together.',seed=42,extra_compute_explicit=True))
    for path in [Path(__file__),RUN/'dataset.json',SOURCE/'detection-results.json',SOURCE/'recognition/results.json']:
        p['input_sha256'][str(path)]=s.sha(path)
    s.write(RUN/'final-preregistration.json',p);s.verify(p)
    s.RUN=RUN;s.STEPS=STEPS;r.RUN=RUN/'recognition';r.prepare()
    print('LONG_PROTOCOL_FROZEN',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','recognize']);phase=parser.parse_args().phase
    if phase=='prepare':prepare()
    elif phase=='run':s.RUN=RUN;s.STEPS=STEPS;s.run()
    else:
        s.RUN=RUN;s.STEPS=STEPS;r.RUN=RUN/'recognition';r.run()


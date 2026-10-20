"""Bounded one-class Faster R-CNN pilot from reviewed manuscript boxes."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import math
import random
import shutil
import sys
from types import MethodType, SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import box_iou, nms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from scripts.train_segmentation import build_dataset, evaluate, read_json, sha256, state_fingerprint, write_json, xyxy

TILE_SIZE, TILE_STRIDE = 768, 512
SCORE_THRESHOLD, NMS_THRESHOLD, MAX_PAGE_BOXES = .25, .5, 100


def tile_windows(width, height):
    def origins(length):
        return sorted(set(range(0, max(1, length - TILE_SIZE + 1), TILE_STRIDE)) | {max(0, length - TILE_SIZE)})
    grid = [(x, y, min(width, x + TILE_SIZE), min(height, y + TILE_SIZE))
            for y in origins(height) for x in origins(width)]
    # The fixed whole-page view also covers objects larger than a tile.
    return list(dict.fromkeys(grid + [(0, 0, width, height)]))


def contains(window, box):
    x1, y1, x2, y2 = xyxy(box)
    return window[0] <= x1 and window[1] <= y1 and window[2] >= x2 and window[3] >= y2


def clipped(box, window):
    x1, y1, x2, y2 = xyxy(box)
    x1, y1 = max(x1, window[0]), max(y1, window[1])
    x2, y2 = min(x2, window[2]), min(y2, window[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 - window[0], y1 - window[1], x2 - window[0], y2 - window[1]]


def make_tiles(page):
    windows = tile_windows(page['width'], page['height'])
    owners = []
    for box in page['boxes']:
        x, y, w, h = box['bbox']
        eligible = [i for i, win in enumerate(windows) if contains(win, box['bbox'])]
        assert eligible, 'Whole-page view must contain every valid box'
        owners.append(min(eligible, key=lambda i: ((windows[i][2]-windows[i][0]) * (windows[i][3]-windows[i][1]),
            ((windows[i][0]+windows[i][2])/2-x-w/2)**2 + ((windows[i][1]+windows[i][3])/2-y-h/2)**2)))
    result = []
    for index, window in enumerate(windows):
        positives, ignores = [], []
        for i, box in enumerate(page['boxes']):
            visible = clipped(box['bbox'], window)
            if visible is None:
                continue
            if owners[i] == index:
                assert contains(window, box['bbox'])
                positives.append(visible)
            else:
                ignores.append(visible)
        ignores += [v for b in page.get('ignore_boxes', []) if (v := clipped(b, window)) is not None]
        if positives:
            result.append({'source_group': page['source_group'], 'image_path': page['image_path'],
                'window': list(window), 'boxes': positives, 'ignore_boxes': ignores})
    assert sum(len(t['boxes']) for t in result) == len(page['boxes'])
    return result


def ignore_background_labels(labels, boxes, ignored):
    if len(ignored) and len(boxes):
        overlap = box_iou(boxes, ignored).amax(1) > 0
        labels = labels.clone()
        labels[(labels == 0) & overlap] = -1
    return labels


def install_ignore_regions(model):
    """Keep unresolved annotations/other tiles' instances out of negative sampling."""
    original_resize = model.transform.resize
    def resize(self, image, target=None):
        previous_size = image.shape[-2:]
        image, target = original_resize(image, target)
        if target is not None:
            target['ignore_boxes'] = resize_boxes(target['ignore_boxes'], previous_size, image.shape[-2:])
        return image, target
    model.transform.resize = MethodType(resize, model.transform)
    original_rpn = model.rpn.assign_targets_to_anchors
    def rpn_assign(self, anchors, targets):
        labels, matched = original_rpn(anchors, targets)
        return [ignore_background_labels(l, a, t['ignore_boxes']) for l, a, t in zip(labels, anchors, targets)], matched
    model.rpn.assign_targets_to_anchors = MethodType(rpn_assign, model.rpn)
    original_select = model.roi_heads.select_training_samples
    original_assign = model.roi_heads.assign_targets_to_proposals
    def select(self, proposals, targets):
        # ponytail: one research forward at a time; pass ignore tensors explicitly if parallel forwards are needed.
        self._pilot_ignore_boxes = [t['ignore_boxes'] for t in targets]
        return original_select(proposals, targets)
    def assign(self, proposals, gt_boxes, gt_labels):
        indices, labels = original_assign(proposals, gt_boxes, gt_labels)
        return indices, [ignore_background_labels(l, p, i) for l, p, i in zip(labels, proposals, self._pilot_ignore_boxes)]
    model.roi_heads.select_training_samples = MethodType(select, model.roi_heads)
    model.roi_heads.assign_targets_to_proposals = MethodType(assign, model.roi_heads)


def make_model(checkpoint, device):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    checkpoint = Path(checkpoint)
    if checkpoint.name != weights.url.rsplit('/', 1)[1] or not sha256(checkpoint).startswith('fb6a3cc7'):
        raise ValueError('Expected hash-verified official COCO MobileNetV3 Faster R-CNN checkpoint')
    hub = checkpoint.parent / 'torch-hub'
    cached = hub / 'checkpoints' / checkpoint.name
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists():
        shutil.copyfile(checkpoint, cached)
    if sha256(cached) != sha256(checkpoint):
        raise ValueError('Cached detector weights differ')
    torch.hub.set_dir(str(hub.resolve()))
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, progress=False,
        trainable_backbone_layers=0, min_size=TILE_SIZE, max_size=1024,
        box_score_thresh=SCORE_THRESHOLD, box_nms_thresh=NMS_THRESHOLD, box_detections_per_img=100)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    assert not any(p.requires_grad for p in model.backbone.body.parameters())
    install_ignore_regions(model)
    return model.to(device)


def as_tensor(image, device):
    pixels = np.array(image.convert('RGB'), copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255


class Detector:
    def __init__(self, model, device):
        self.model, self.device = model, device

    @torch.no_grad()
    def segment_page(self, pixels):
        self.model.eval()
        image = Image.fromarray(pixels)
        boxes, scores = [], []
        # These windows depend only on image dimensions, never on target boxes.
        for x1, y1, x2, y2 in tile_windows(image.width, image.height):
            prediction = self.model([as_tensor(image.crop((x1, y1, x2, y2)), self.device)])[0]
            shifted = prediction['boxes'] + torch.tensor([x1, y1, x1, y1], device=self.device)
            boxes.append(shifted); scores.append(prediction['scores'])
        boxes, scores = torch.cat(boxes), torch.cat(scores)
        keep = nms(boxes, scores, NMS_THRESHOLD)[:MAX_PAGE_BOXES]
        output = []
        for box, score in zip(boxes[keep].cpu().tolist(), scores[keep].cpu().tolist()):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
            x2, y2 = min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2))
            if x2 > x1 and y2 > y1:
                output.append(SimpleNamespace(bbox=(x1, y1, x2-x1, y2-y1), confidence=score))
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['import-report', 'snapshot', 'checkpoint', 'sam-results', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--exhaustive-confirmed', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not args.exhaustive_confirmed:
        parser.error('Require human confirmation that all page glyphs were annotated')
    if args.epochs < 1 or not 0 < args.learning_rate < 1 or args.output_dir.exists():
        parser.error('Invalid recipe or existing output directory')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    dataset = build_dataset(args.import_report, args.snapshot)
    imported = read_json(args.import_report)
    for page in dataset['pages']:
        positive_geometries = {tuple(b['bbox']) for b in page['boxes']}
        ignored = {tuple(r['bbox']) for r in imported['rows']
            if 'image-pixels:'+r['source_image_pixel_sha256'] == page['source_group']
            and tuple(r['bbox']) not in positive_geometries}
        for bbox in ignored:
            x,y,w,h=bbox
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > page['width'] or y+h > page['height']:
                raise ValueError('Invalid ignored geometry')
        page['ignore_boxes'] = [list(b) for b in sorted(ignored)]
    dataset['annotation_exhaustiveness'] = 'Human confirmed all glyphs annotated; unresolved/rejected geometry is ignored, so scored positive GT is a subset'
    dataset['exhaustiveness_confirmation'] = {'source':'user message', 'text':'Tous les glyphes de chaque planche'}
    dataset['split_ignore_counts'] = {s:sum(len(p['ignore_boxes']) for p in dataset['pages'] if p['split']==s)
        for s in ['train','dev','locked_test']}
    tiles = [t for p in dataset['pages'] if p['split']=='train' for t in make_tiles(p)]
    assert sum(len(t['boxes']) for t in tiles) == dataset['split_box_counts']['train']
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'dataset.json', dataset)
    write_json(args.output_dir/'train_tiles.json', tiles)
    sam_results = read_json(args.sam_results)
    assert [p['source_group'] for p in dataset['pages']] == [p['source_group'] for p in sam_results['dataset']['pages']]
    recipe = {'schema_version':'glyph-bbox-detector-pilot.v1', 'model':'fasterrcnn_mobilenet_v3_large_fpn',
        'pretrained_source':FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.url, 'pretrained_sha256':sha256(args.checkpoint),
        'dataset_sha256':sha256(args.output_dir/'dataset.json'), 'script_sha256':sha256(__file__),
        'tile_dataset_sha256':sha256(args.output_dir/'train_tiles.json'), 'seed':42, 'epochs':args.epochs,
        'optimizer':'AdamW', 'learning_rate':args.learning_rate, 'weight_decay':.0001, 'batch_size':2,
        'trainable':'FPN, RPN and ROI heads; image backbone frozen', 'classes':['background','glyph'],
        'losses':['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg'],
        'tile_size':TILE_SIZE,'tile_stride':TILE_STRIDE,'additional_view':'whole page, at training/inference',
        'tile_targets':'one fully containing owner per GT; other/intersecting/uncertain boxes ignored',
        'score_threshold':SCORE_THRESHOLD,'nms_threshold':NMS_THRESHOLD,'max_proposals_per_page':MAX_PAGE_BOXES,
        'selection':'dev only: matches IoU>=.5, then mean matched IoU; proposals<=1.5*SAM dev proposals+5; latest fallback if none eligible',
        'test_status':'Existing page-separated heldout already examined in prior experiments: exploratory comparison, not a new blind test',
        'promotion':'none','exhaustiveness_confirmation':dataset['exhaustiveness_confirmation']}
    write_json(args.output_dir/'preregistration.json', recipe)
    model = make_model(args.checkpoint, args.device)
    torch.save(model.state_dict(), args.output_dir/'initial.pt')
    original_body = state_fingerprint(model, 'backbone.body')
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=.0001)
    detector = Detector(model, args.device)
    dev_pages = [p for p in dataset['pages'] if p['split']=='dev']
    initial_dev = evaluate(detector, dev_pages)
    print('READY',len(tiles),'tiles',dataset['split_box_counts'],'ignored',dataset['split_ignore_counts'],'trainable',trainable_count,flush=True)
    images = {path:Image.open(path).convert('RGB') for path in {t['image_path'] for t in tiles}}
    history, best_score, selected_epoch = [], None, None
    for epoch in range(1, args.epochs+1):
        random.shuffle(tiles); model.train()
        loss_sums, batches = Counter(), 0
        for start in range(0, len(tiles), 2):
            inputs, targets = [], []
            for tile in tiles[start:start+2]:
                image = images[tile['image_path']].crop(tile['window'])
                boxes = torch.tensor(tile['boxes'], dtype=torch.float32)
                ignored = torch.tensor(tile['ignore_boxes'], dtype=torch.float32).reshape(-1,4)
                if random.random() < .5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    for coordinates in [boxes, ignored]:
                        coordinates[:, [0,2]] = image.width - coordinates[:, [2,0]]
                inputs.append(as_tensor(image, args.device))
                targets.append({'boxes':boxes.to(args.device),'labels':torch.ones(len(boxes),dtype=torch.int64,device=args.device),
                    'ignore_boxes':ignored.to(args.device)})
            losses = model(inputs, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite detector loss')
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
            optimizer.step(); batches += 1
            loss_sums.update({k:float(v.detach()) for k,v in losses.items()})
        torch.save(model.state_dict(), args.output_dir/'latest.pt')
        dev = evaluate(detector, dev_pages)
        c = dev['counts']
        budget = c['proposal_count'] <= 1.5*sam_results['baseline']['dev']['counts']['proposal_count']+5
        score = (c['matched_boxes'],c['mean_matched_iou'])
        if budget and (best_score is None or score > best_score):
            best_score, selected_epoch = score, epoch
            torch.save(model.state_dict(), args.output_dir/'best.pt')
        history.append({'epoch':epoch,'losses':{k:v/batches for k,v in loss_sums.items()},'dev':dev,'within_proposal_budget':budget})
        write_json(args.output_dir/'history.json',history)
        print('EPOCH',epoch,'LOSSES',history[-1]['losses'],'DEV',c,'BUDGET',budget,flush=True)
    eligible = selected_epoch is not None
    if not eligible:
        selected_epoch = args.epochs
        shutil.copyfile(args.output_dir/'latest.pt', args.output_dir/'best.pt')
    model.load_state_dict(torch.load(args.output_dir/'best.pt',map_location=args.device,weights_only=True))
    assert state_fingerprint(model,'backbone.body') == original_body
    candidate = {s:evaluate(detector,[p for p in dataset['pages'] if p['split']==s]) for s in ['train','dev','locked_test']}
    result = {'schema_version':'glyph-bbox-detector-results.v1','dataset':dataset,'recipe':recipe,'train_tile_count':len(tiles),
        'trainable_parameters':trainable_count,'initial_dev':initial_dev,'selected_epoch':selected_epoch,'selection_eligible':eligible,
        'history':history,'baseline_sam':sam_results['baseline'],'candidate':candidate,
        'verification':{'backbone_unchanged':True,'checkpoint_sha256':sha256(args.output_dir/'best.pt'),
            'heldout_pages_not_used_for_fit':True,'selected_on_dev_only':True},'activated':False}
    write_json(args.output_dir/'results.json',result)
    print('RESULTS',args.output_dir/'results.json',flush=True)


if __name__ == '__main__':
    main()

"""Checks for page-independent tiling and ignored review regions."""
import numpy as np
import torch
from scripts.train_bbox_detector import contains, ignore_background_labels, make_tiles, tile_windows


def test_each_box_has_one_complete_owner_and_fragments_are_ignored():
    page={'source_group':'train-page','image_path':'unused','width':1500,'height':1400,
          'boxes':[{'bbox':[600,600,200,200]},{'bbox':[10,20,1400,100]}],
          'ignore_boxes':[[500,500,40,40]]}
    tiles=make_tiles(page)
    assert sum(len(t['boxes']) for t in tiles)==2
    for tile in tiles:
        x,y,_,_=tile['window']
        for box in tile['boxes']:
            absolute=[box[0]+x,box[1]+y,box[2]-box[0],box[3]-box[1]]
            assert contains(tile['window'],absolute)
    assert any(t['window']==[0,0,1500,1400] for t in tiles)
    assert any(t['ignore_boxes'] for t in tiles)


def test_ignore_regions_never_override_positive_labels():
    boxes=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.],[1.,1.,9.,9.],[50.,50.,60.,60.]])
    ignored=torch.tensor([[0.,0.,10.,10.]])
    labels=torch.tensor([0,0,1,-1])
    result=ignore_background_labels(labels,boxes,ignored)
    assert result.tolist()==[-1,0,1,-1]
    assert labels.tolist()==[0,0,1,-1]
    assert torch.equal(ignore_background_labels(labels,boxes,torch.empty(0,4)),labels)


def test_inference_windows_cover_page_and_depend_only_on_size():
    for w,h in [(140,129),(750,1046),(1562,2486)]:
        windows=tile_windows(w,h)
        assert len(windows)==len(set(windows))
        assert (0,0,w,h) in windows
        covered=np.zeros((h,w),dtype=bool)
        for x1,y1,x2,y2 in windows:
            assert 0<=x1<x2<=w and 0<=y1<y2<=h
            covered[y1:y2,x1:x2]=True
        assert covered.all()


def test_real_forward_resizes_ignored_regions_and_excludes_rpn_roi_negatives():
    from types import MethodType
    from torch import nn
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign, box_iou
    from scripts.train_bbox_detector import install_ignore_regions
    torch.set_num_threads(4)
    backbone=nn.Conv2d(3,8,3,stride=4,padding=1)
    backbone.out_channels=8
    model=FasterRCNN(backbone,num_classes=2,min_size=64,max_size=64,
        rpn_anchor_generator=AnchorGenerator(((8,16,32),),((.5,1.,2.),)),
        box_roi_pool=MultiScaleRoIAlign(['0'],2,2),
        rpn_pre_nms_top_n_train=100,rpn_post_nms_top_n_train=50)
    install_ignore_regions(model)
    captured={}
    rpn_assign=model.rpn.assign_targets_to_anchors
    def record_rpn(self,anchors,targets):
        labels,matched=rpn_assign(anchors,targets)
        captured['rpn']=(anchors[0],labels[0],targets[0]['ignore_boxes'])
        return labels,matched
    roi_assign=model.roi_heads.assign_targets_to_proposals
    def record_roi(self,proposals,boxes,labels):
        indices,assigned=roi_assign(proposals,boxes,labels)
        captured['roi']=(proposals[0],assigned[0],self._pilot_ignore_boxes[0])
        return indices,assigned
    model.rpn.assign_targets_to_anchors=MethodType(record_rpn,model.rpn)
    model.roi_heads.assign_targets_to_proposals=MethodType(record_roi,model.roi_heads)
    model.train()
    target={'boxes':torch.tensor([[2.,2.,8.,8.]]),'labels':torch.tensor([1]),
            'ignore_boxes':torch.tensor([[20.,20.,30.,30.]])}
    losses=model([torch.rand(3,32,32)],[target])
    assert set(losses)=={'loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg'}
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    for stage in ['rpn','roi']:
        boxes,labels,ignored=captured[stage]
        assert torch.allclose(ignored,torch.tensor([[40.,40.,60.,60.]]))
        overlap=box_iou(boxes,ignored).amax(1)>0
        assert overlap.any() and (labels[overlap]==-1).any()
        assert not (labels[overlap]==0).any()
        assert (labels>0).any()

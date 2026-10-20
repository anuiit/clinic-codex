import torch
from scripts.research_recognition_mechanisms import contrastive,limited_supports,render_views


def test_support_caps_distinct_pixels_pages_and_holdout():
    labels=torch.tensor([0,0,0,0,1,1,1])
    groups=['a','a','b','held','a','b','a'];pixels=['x','x','y','z','m','n','o']
    ids=[0,1,2,4,5,6]
    one=limited_supports(labels,groups,pixels,ids,1,42)
    two=limited_supports(labels,groups,pixels,ids,2,42)
    assert one==limited_supports(labels,groups,pixels,ids,1,42)
    assert set(one)<=set(two)<=set(ids) and 3 not in two
    assert len(one)==2 and len(two)==4
    for label in [0,1]:
        selected=[i for i in two if labels[i]==label]
        assert len({pixels[i] for i in selected})==2 and len({groups[i] for i in selected})==2


def test_contrastive_objects_and_crop_views():
    z=torch.tensor([[1.,0.],[.8,.2],[1.,0.],[0.,1.],[.1,.9]],requires_grad=True)
    y=torch.tensor([0,0,0,1,1]);groups=['a','b','a','a','b'];pixels=['x','y','x','v','w'];objects=['one','two','one','three','four']
    loss,pos=contrastive(z,y,groups,pixels,objects)
    assert pos[0,1] and not pos[0,2] and not pos.diagonal().any()
    assert torch.isfinite(loss) and loss>0
    loss.backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    no,mask=contrastive(z,y,['a']*5,pixels,objects)
    assert not mask.any() and no==0
    import numpy as np
    rgb=np.full((11,11,3),40,dtype=np.uint8);m=np.zeros((11,11),dtype=bool);m[5,5]=True
    views=render_views(rgb,m)
    assert np.array_equal(views['rectangle'],rgb)
    assert (views['mask'][~m]==255).all() and (views['mask'][m]==40).all()
    assert (views['half_background'][~m]==147).all() and (views['half_background'][m]==40).all()
    assert (views['dilated3'][2:9,2:9]==40).all() and (views['dilated3'][0]==255).all()


def test_same_class_unqualified_pair_is_not_a_negative():
    import math
    z=torch.tensor([[1.,0.],[1.,0.],[1.,0.]],requires_grad=True)
    loss,pos=contrastive(z,torch.tensor([0,0,0]),['a','b','a'],['x','y','x'],['one','two','one'])
    assert pos.tolist()==[[False,True,False],[True,False,True],[False,True,False]]
    assert torch.allclose(loss,torch.tensor(math.log(2)/3),atol=1e-6)
    loss.backward();assert torch.isfinite(z.grad).all()

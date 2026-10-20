import torch
import pytest

from scripts.train_contrastive_recognition import cross_page_batches, positive_mask, supervised_contrastive_loss


def test_controlled_batches_never_use_excluded_rows_and_pair_distinct_pages():
    labels = torch.tensor([c for c in range(20) for _ in range(8)])
    groups = [f"page-{c}-{i//4}" for c in range(20) for i in range(8)]
    train = list(range(160))
    one = list(cross_page_batches(labels, groups, train, 42, steps=2))
    two = list(cross_page_batches(labels, groups, train, 42, steps=2))
    assert [b[0] for b in one] == [b[0] for b in two]
    for indices, local, support in one:
        assert set(indices) <= set(train) and support.sum() == 80 and len(indices) == len(set(indices))
        mask = positive_mask(local, [groups[i] for i in indices], [str(i) for i in indices])
        assert not mask.diagonal().any() and mask.any(dim=1).all()
        for c in local.unique():
            a = {groups[indices[i]] for i in range(len(indices)) if local[i] == c and support[i]}
            b = {groups[indices[i]] for i in range(len(indices)) if local[i] == c and not support[i]}
            assert a.isdisjoint(b)


def test_supcon_rewards_same_class_agreement_and_has_finite_gradients():
    labels = torch.tensor([0, 0, 1, 1])
    mask = positive_mask(labels, ['a','b','a','b'], ['1','2','3','4'])
    good = torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]], requires_grad=True)
    bad = torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]], requires_grad=True)
    assert supervised_contrastive_loss(good,mask) < supervised_contrastive_loss(bad,mask)
    loss = supervised_contrastive_loss(bad,mask); loss.backward()
    assert torch.isfinite(bad.grad).all() and bad.grad.abs().sum() > 0
    # Identical pixels from another page are never used as positive evidence.
    duplicate = positive_mask(labels,['a','b','a','b'],['same','same','3','4'])
    with pytest.raises(ValueError,match='invalid positive mask'):
        supervised_contrastive_loss(good,duplicate)


def test_small_groups_reduce_batch_size_instead_of_repeating_examples():
    labels = torch.tensor([c for c in range(20) for _ in range(2)])
    groups = [f"page-{i}" for i in range(40)]
    indices, local, support = next(cross_page_batches(labels, groups, list(range(40)), 42, steps=1))
    assert len(indices) == len(set(indices)) == 40 and int(support.sum()) == 20
    assert positive_mask(local, [groups[i] for i in indices], [str(i) for i in indices]).any(dim=1).all()

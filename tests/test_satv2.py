import numpy as np
import pytest
import torch
import torch.nn as nn

from pointcept.datasets.for_instance import TreeInstanceParser
from pointcept.models.segment_any_tree_v2.loss import (
    SegmentAnyTreeV2Loss,
    discriminative_loss_single,
    sigmoid_focal_loss,
    soft_dice_loss,
    soft_iou_score_target,
)
from pointcept.models.segment_any_tree_v2.nms import (
    greedy_mask_nms,
    non_max_suppression,
    pairwise_mask_iou,
)
from pointcept.models.segment_any_tree_v2.segment_any_tree_v2 import (
    SegmentAnyTreeV2,
    SegmentAnyTreeV2Decoder,
    farthest_point_sample,
)
from pointcept.models.utils.structure import Point
from pointcept.utils.satv2_metrics import (
    instance_metrics,
    semantic_confusion_matrix,
    semantic_iou,
    summarize_detection,
)


def test_fps_selects_all_candidates_before_repeating():
    features = torch.tensor([[0.0], [1.0], [3.0]])
    selected = farthest_point_sample(features, 5)
    assert selected.shape == (5,)
    assert set(selected[:3].tolist()) == {0, 1, 2}
    assert torch.equal(selected[3:], selected[:2])


def test_tree_instance_parser_excludes_nonpositive_ids():
    parsed = TreeInstanceParser()(
        {
            "segment": np.array([0, 1, 2, 1, 2, 1]),
            "instance": np.array([8, -1, 0, 7, 12, 7]),
        }
    )["instance"]
    assert parsed.tolist() == [-1, -1, -1, 0, 1, 0]


def test_decoder_and_all_losses_have_finite_gradients():
    torch.manual_seed(3)
    features = torch.randn(8, 8, requires_grad=True)
    embeddings = torch.randn(8, 5, requires_grad=True)
    segment = torch.tensor([0, 1, 1, 2, 2, 1, 2, 0])
    instance = torch.tensor([-1, 0, 0, 1, 1, 2, 2, -1])
    tree_mask = segment != 0
    decoder = SegmentAnyTreeV2Decoder(
        in_channels=8,
        dim=8,
        num_queries=4,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
    )
    prediction = decoder.forward_sample(
        features, embeddings, tree_mask, instance, use_isa=True
    )
    assert prediction["mask_logits"].shape == (4, 6)
    assert prediction["score_logits"].shape == (4,)
    assert len(prediction["aux_outputs"]) == 1
    assert torch.all(tree_mask[prediction["seed_indices"]])

    criterion = SegmentAnyTreeV2Loss(warmup_epochs=1)
    semantic_logits = torch.randn(8, 3, requires_grad=True)
    result = criterion(
        semantic_logits=semantic_logits,
        embeddings=embeddings,
        predictions=[prediction],
        segment=segment,
        instance=instance,
        counts=torch.tensor([8]),
        current_epoch=1,
        instance_active=True,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(semantic_logits.grad).all()


def test_paper_loss_helpers_and_chunking():
    mask_logits = torch.tensor([[0.0, 1.0], [-1.0, 0.5], [0.2, -0.3]])
    probabilities = mask_logits.sigmoid()
    target_masks = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    valid_queries = torch.tensor([True, False, True])

    numerator = 2 * (probabilities * target_masks).sum(dim=1) + 1
    denominator = probabilities.sum(dim=1) + target_masks.sum(dim=1) + 1
    expected_dice = (1 - numerator / denominator)[valid_queries].sum() / 3
    assert torch.allclose(
        soft_dice_loss(probabilities, target_masks, valid_queries), expected_dice
    )

    focal_chunked = sigmoid_focal_loss(
        mask_logits,
        probabilities,
        target_masks,
        valid_queries,
        alpha=0.25,
        gamma=2.0,
        chunk_size=1,
    )
    focal_full = sigmoid_focal_loss(
        mask_logits,
        probabilities,
        target_masks,
        valid_queries,
        alpha=0.25,
        gamma=2.0,
        chunk_size=64,
    )
    assert torch.allclose(focal_chunked, focal_full)

    gt_masks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    expected_iou = torch.stack(
        [
            probabilities[:, 0] / (probabilities.sum(dim=1) + 1 - probabilities[:, 0]),
            probabilities[:, 1] / (probabilities.sum(dim=1) + 1 - probabilities[:, 1]),
        ],
        dim=1,
    ).max(dim=1).values
    assert torch.allclose(
        soft_iou_score_target(probabilities, gt_masks), expected_iou, atol=1e-6
    )

    embeddings = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [4.0, 2.0], [9.0, 9.0]]
    )
    instances = torch.tensor([0, 0, 1, 1, 2])
    loss_var, loss_dist, loss_reg = discriminative_loss_single(
        embeddings, instances, delta_var=0.5, delta_dist=2.5
    )
    means = torch.tensor([[0.5, 0.0], [4.0, 1.0]])
    expected_var = torch.tensor((0.0 + 0.25) / 2)
    expected_dist = (5.0 - torch.linalg.vector_norm(means[0] - means[1])).square()
    expected_reg = torch.linalg.vector_norm(means, dim=1).mean()
    assert torch.allclose(loss_var, expected_var)
    assert torch.allclose(loss_dist, expected_dist)
    assert torch.allclose(loss_reg, expected_reg)

    amp_embeddings = embeddings.half().requires_grad_()
    amp_losses = discriminative_loss_single(
        amp_embeddings, instances, delta_var=0.5, delta_dist=2.5
    )
    assert all(loss.dtype == torch.float32 for loss in amp_losses)
    sum(amp_losses).backward()
    assert torch.isfinite(amp_embeddings.grad).all()

    for edge_instances in (
        torch.full((5,), -1),
        torch.tensor([0, 0, -1, -1, -1]),
    ):
        edge_embeddings = embeddings.half().requires_grad_()
        edge_losses = discriminative_loss_single(
            edge_embeddings, edge_instances, delta_var=0.5, delta_dist=2.5
        )
        assert all(loss.dtype == torch.float32 for loss in edge_losses)
        sum(edge_losses).backward()
        assert torch.isfinite(edge_embeddings.grad).all()


class _ToyPointBackbone(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()
        self.projection = nn.Linear(3, out_channels)

    def forward(self, data_dict):
        point = Point(data_dict)
        point.feat = self.projection(point.feat)
        return point


class _ToyTensorBackbone(_ToyPointBackbone):
    def forward(self, data_dict):
        # Like Volt, require batch indices and return only point features.
        assert data_dict["batch"].tolist() == [0] * 4 + [1] * 4
        return self.projection(data_dict["feat"])


@pytest.mark.parametrize("backbone_type", [_ToyPointBackbone, _ToyTensorBackbone])
def test_training_output_contains_only_scalars(backbone_type):
    model = SegmentAnyTreeV2(
        backbone=dict(type=backbone_type, out_channels=8),
        backbone_out_channels=8,
        semantic_num_classes=2,
        tree_class_indices=(1,),
        embedding_dim=5,
        decoder=dict(
            dim=8, num_queries=3, num_layers=1, num_heads=2, hidden_dim=16
        ),
        criteria=dict(class_weights=(1.0, 2.0)),
    )
    model.train()
    data_dict = {
        "coord": torch.randn(8, 3),
        "grid_coord": torch.arange(24).reshape(8, 3),
        "feat": torch.randn(8, 3),
        "offset": torch.tensor([4, 8]),
        "segment": torch.tensor([0, 1, 1, 1, 1, 1, 1, 0]),
        "instance": torch.tensor([-1, 0, 0, 1, 1, 2, 2, -1]),
    }
    output = model(data_dict)
    assert output
    assert "seg_logits" not in output
    assert "pred_semantic" not in output
    assert "embeddings" not in output
    assert all(value.ndim == 0 for value in output.values())
    output["loss"].backward()
    assert torch.isfinite(model.backbone.projection.weight.grad).all()
    point = model(data_dict, return_point=True)["point"]
    assert isinstance(point, Point)
    assert point.feat.shape == (8, 8)
    assert torch.equal(point.offset, data_dict["offset"])
    assert torch.equal(point.coord, data_dict["coord"])


def test_empty_tree_decoder_and_nms():
    decoder = SegmentAnyTreeV2Decoder(
        in_channels=8,
        dim=8,
        num_queries=3,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
    )
    prediction = decoder.forward_sample(
        torch.randn(4, 8),
        torch.randn(4, 5),
        torch.zeros(4, dtype=torch.bool),
        torch.full((4,), -1),
        use_isa=True,
    )
    assert prediction["mask_logits"].shape == (3, 0)

    masks = torch.tensor(
        [[True, True, False], [True, True, False], [False, False, True]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    ious = pairwise_mask_iou(masks)
    expected_ious = torch.tensor(
        [[1.0, 1.0, 0.0]] * 2 + [[0.0, 0.0, 1.0]]
    )
    assert torch.equal(ious, expected_ious)
    assert non_max_suppression(ious, scores, 0.1).tolist() == [0, 2]
    keep = greedy_mask_nms(masks, scores, 0.1)
    assert keep.tolist() == [0, 2]


def test_paper_metrics_on_perfect_prediction():
    semantic_target = np.array([0, 1, 1, 2, 2])
    confusion = semantic_confusion_matrix(semantic_target, semantic_target)
    class_iou, mean_iou = semantic_iou(confusion)
    np.testing.assert_allclose(class_iou, 1)
    assert mean_iou == 1

    instances = np.array([0, 1, 1, 2, 2])
    result = instance_metrics(instances, instances, {1: 0.9, 2: 0.8})
    detection = summarize_detection(result["tp"], result["fp"], result["fn"])
    assert detection == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["weighted_coverage"] == 1.0

"""SegmentAnyTreeV2 model with selectable Transformer/Mamba decoder."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point

from .decoder import SegmentAnyTreeV2Decoder, SegmentAnyTreeV2MambaDecoder
from .loss import SegmentAnyTreeV2Loss
from .nms import greedy_mask_nms


@MODELS.register_module("SaTv2-myself-v1m1")
class SegmentAnyTreeV2(PointModel):
    def __init__(
        self,
        backbone,
        backbone_out_channels: int = 128,
        semantic_num_classes: int = 3,
        semantic_ignore_index: int = -1,
        tree_class_indices=(1, 2),
        embedding_dim: int = 5,
        warmup_epochs: int = 30,
        decoder=None,
        criteria=None,
        score_threshold: float = 0.15,
        mask_threshold: float = 0.5,
        nms_iou_threshold: float = 0.1,
        min_points: int = 0,
        decoder_type: str | None = None,
    ):
        super().__init__()
        norm = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.backbone = build_model(backbone)
        self.semantic_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm(backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, semantic_num_classes),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, embedding_dim),
        )
        decoder = {} if decoder is None else dict(decoder)
        configured_type = decoder.pop("type", None)
        decoder_type = decoder_type or configured_type or "transformer"
        decoder_cls = {
            "transformer": SegmentAnyTreeV2Decoder,
            "mamba": SegmentAnyTreeV2MambaDecoder,
        }.get(decoder_type)
        if decoder_cls is None:
            raise ValueError("decoder_type must be transformer or mamba")
        self.decoder = decoder_cls(
            in_channels=backbone_out_channels, **decoder
        )
        criteria = {} if criteria is None else criteria
        self.criteria = SegmentAnyTreeV2Loss(
            ignore_index=semantic_ignore_index,
            warmup_epochs=warmup_epochs,
            **criteria,
        )

        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.tree_class_indices = tuple(tree_class_indices)
        self.warmup_epochs = warmup_epochs
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.min_points = min_points
        self.current_epoch = 0

    def before_epoch(self):
        # Pointcept folds several logical data passes into one trainer epoch
        # when validation is less frequent.  Recover the paper's logical epoch.
        loop = int(getattr(self.trainer.cfg.data.train, "loop", 1))
        self.current_epoch = int(self.trainer.epoch) * loop

    def set_epoch(self, epoch: int):
        """Set the logical epoch for non-Pointcept/custom training loops."""

        self.current_epoch = int(epoch)

    def _tree_mask(
        self, semantic_logits: torch.Tensor, segment: torch.Tensor | None
    ) -> torch.Tensor:
        if self.training and segment is not None:
            labels = segment
        else:
            labels = semantic_logits.argmax(dim=1)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_index in self.tree_class_indices:
            mask |= labels == class_index
        return mask

    def _full_masks(self, prediction: dict) -> torch.Tensor:
        full = prediction["mask_logits"].new_full(
            (self.decoder.num_queries, prediction["num_points"]), -100
        )
        full[:, prediction["tree_indices"]] = prediction["mask_logits"]
        return full

    @torch.no_grad()
    def inference(self, prediction: dict) -> dict[str, torch.Tensor]:
        scores = prediction["score_logits"].sigmoid()
        masks = self._full_masks(prediction).sigmoid() >= self.mask_threshold
        # Match the reference post-processing order: NMS first, followed by
        # score and cluster-size filtering.
        if len(scores):
            nms_keep = greedy_mask_nms(masks, scores, self.nms_iou_threshold)
            scores = scores[nms_keep]
            masks = masks[nms_keep]
            keep = (scores >= self.score_threshold) & (
                masks.sum(dim=1) >= max(self.min_points, 1)
            )
            scores = scores[keep]
            masks = masks[keep]
            order = scores.argsort(descending=True)
            scores = scores[order]
            masks = masks[order]
        return {
            "pred_scores": scores.cpu(),
            "pred_masks": masks.cpu(),
            # Instance masks are class-agnostic trees.  Index 1 is used only
            # for compatibility with Pointcept's generic instance API.
            "pred_classes": torch.ones(len(scores), dtype=torch.long),
        }

    def forward(self, data_dict, return_point: bool = False):
        point = Point(data_dict)
        backbone_output = self.backbone(point)
        if isinstance(backbone_output, Point):
            point = backbone_output
        elif isinstance(backbone_output, torch.Tensor):
            # Volt returns features in input point order; retain batch metadata.
            point.feat = backbone_output
        else:
            raise TypeError("SaTv2 requires a backbone returning Point or Tensor")
        if return_point:
            return {"point": point}

        semantic_logits = self.semantic_head(point.feat)
        embeddings = self.embedding_head(point.feat)
        counts = offset2bincount(point.offset)
        feature_list = point.feat.split(counts.tolist())
        semantic_list = semantic_logits.split(counts.tolist())
        embedding_list = embeddings.split(counts.tolist())
        coordinate_list = point.coord.split(counts.tolist())
        segment_list = (
            data_dict["segment"].split(counts.tolist())
            if "segment" in data_dict
            else [None] * len(counts)
        )
        instance_list = (
            data_dict["instance"].split(counts.tolist())
            if "instance" in data_dict
            else [None] * len(counts)
        )

        use_isa = (not self.training) or self.current_epoch >= self.warmup_epochs
        predictions = []
        for features, semantic, embedding, segment, instance, coordinates in zip(
            feature_list,
            semantic_list,
            embedding_list,
            segment_list,
            instance_list,
            coordinate_list,
        ):
            predictions.append(
                self.decoder.forward_sample(
                    features=features,
                    embeddings=embedding,
                    tree_mask=self._tree_mask(semantic, segment),
                    instance=instance,
                    use_isa=use_isa,
                    coordinates=coordinates,
                )
            )

        output = {}
        if "segment" in data_dict and "instance" in data_dict:
            output.update(
                self.criteria(
                    semantic_logits=semantic_logits,
                    embeddings=embeddings,
                    predictions=predictions,
                    segment=data_dict["segment"],
                    instance=data_dict["instance"],
                    counts=counts,
                    current_epoch=self.current_epoch,
                    instance_active=use_isa,
                )
            )
        # Pointcept's training logger treats every returned item as a scalar.
        # Dense predictions remain available to validation/inference only.
        if self.training:
            return output

        output.update(
            seg_logits=semantic_logits,
            pred_semantic=semantic_logits.argmax(dim=1),
            embeddings=embeddings,
        )
        if len(predictions) != 1:
            raise ValueError("SaTv2 inference currently requires batch size 1")
        output.update(self.inference(predictions[0]))
        return output

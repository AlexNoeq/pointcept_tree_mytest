"""Paper-faithful losses for the Pointcept SegmentAnyTreeV2 model."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses import LovaszLoss


class InstanceTargets(NamedTuple):
    """One-to-many mask targets shared by all decoder layers."""

    valid_queries: torch.Tensor
    query_masks: torch.Tensor
    gt_masks: torch.Tensor


def soft_dice_loss(
    probabilities: torch.Tensor,
    target_masks: torch.Tensor,
    valid_queries: torch.Tensor,
) -> torch.Tensor:
    """Equation 18, normalized by the paper's fixed query count Q."""

    if probabilities.shape != target_masks.shape:
        raise ValueError("probabilities and target_masks must have the same shape")
    if valid_queries.shape != probabilities.shape[:1]:
        raise ValueError("valid_queries must have shape [num_queries]")
    if probabilities.shape[1] == 0 or not valid_queries.any():
        return probabilities.sum() * 0

    numerator = 2 * (probabilities * target_masks).sum(dim=1) + 1
    denominator = probabilities.sum(dim=1) + target_masks.sum(dim=1) + 1
    per_query = 1 - numerator / denominator
    return per_query[valid_queries].sum() / probabilities.shape[0]


def sigmoid_focal_loss(
    mask_logits: torch.Tensor,
    probabilities: torch.Tensor,
    target_masks: torch.Tensor,
    valid_queries: torch.Tensor,
    alpha: float,
    gamma: float,
    chunk_size: int,
) -> torch.Tensor:
    """Equations 19--20 with invalid ISA queries contributing zero loss.

    Query chunking lowers peak temporary memory for large tree-voxel masks
    without changing the Q*V normalization or the resulting gradients.
    """

    num_queries, num_tree_points = mask_logits.shape
    if num_tree_points == 0 or not valid_queries.any():
        return mask_logits.sum() * 0

    valid_indices = valid_queries.nonzero(as_tuple=False).flatten()
    loss = mask_logits.sum() * 0
    for start in range(0, len(valid_indices), chunk_size):
        indices = valid_indices[start : start + chunk_size]
        logits = mask_logits.index_select(0, indices)
        probability = probabilities.index_select(0, indices)
        targets = target_masks.index_select(0, indices)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probability * targets + (1 - probability) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = loss + (alpha_t * (1 - p_t).pow(gamma) * bce).sum()
    return loss / (num_queries * num_tree_points)


def soft_iou_score_target(
    probabilities: torch.Tensor, gt_masks: torch.Tensor
) -> torch.Tensor:
    """Equation 21: maximum soft IoU against every ground-truth tree."""

    if probabilities.shape[1] != gt_masks.shape[1]:
        raise ValueError("probabilities and gt_masks must share the point dimension")
    if probabilities.shape[1] == 0 or gt_masks.shape[0] == 0:
        return probabilities.new_zeros(probabilities.shape[0])

    intersection = probabilities @ gt_masks.transpose(0, 1)
    union = (
        probabilities.sum(dim=1, keepdim=True)
        + gt_masks.sum(dim=1).unsqueeze(0)
        - intersection
    )
    return (intersection / (union + 1e-6)).max(dim=1).values


def discriminative_loss_single(
    embeddings: torch.Tensor,
    instance: torch.Tensor,
    delta_var: float,
    delta_dist: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equations 24--26 for one sample, vectorized over valid instances."""

    valid_points = instance >= 0
    if not valid_points.any():
        zero = embeddings.sum(dtype=torch.float32) * 0
        return zero, zero, zero

    # Keep norm, scatter accumulation, and pairwise center distances in FP32.
    # The cast remains differentiable, so gradients flow back to AMP embeddings.
    point_embeddings = embeddings[valid_points].float()
    point_instances = instance[valid_points]
    _, inverse, counts = torch.unique(
        point_instances, sorted=True, return_inverse=True, return_counts=True
    )
    eligible_instances = counts >= 2
    if not eligible_instances.any():
        zero = embeddings.sum(dtype=torch.float32) * 0
        return zero, zero, zero

    num_instances = len(counts)
    embedding_sums = point_embeddings.new_zeros(
        (num_instances, embeddings.shape[1])
    )
    embedding_sums = embedding_sums.index_add(0, inverse, point_embeddings)
    means = embedding_sums / counts.to(point_embeddings.dtype).unsqueeze(1)

    distances = torch.linalg.vector_norm(
        point_embeddings - means[inverse], ord=2, dim=1
    )
    variance = F.relu(distances - delta_var).square()
    variance_sums = variance.new_zeros(num_instances)
    variance_sums = variance_sums.index_add(0, inverse, variance)
    variance_per_instance = variance_sums / counts.to(variance.dtype)
    loss_var = variance_per_instance[eligible_instances].mean()

    eligible_means = means[eligible_instances]
    if len(eligible_means) > 1:
        center_distances = torch.pdist(eligible_means, p=2)
        loss_dist = F.relu(2 * delta_dist - center_distances).square().mean()
    else:
        loss_dist = eligible_means.sum() * 0
    loss_reg = torch.linalg.vector_norm(eligible_means, ord=2, dim=1).mean()
    return loss_var, loss_dist, loss_reg


class SegmentAnyTreeV2Loss(nn.Module):
    """SegmentAnyTreeV2 training objective from Equations 15--27."""

    def __init__(
        self,
        class_weights=(1.0, 10.0, 2.0),
        ignore_index: int = -1,
        semantic_ce_weight: float = 0.5,
        semantic_lovasz_weight: float = 0.5,
        dice_weight: float = 2.0,
        focal_weight: float = 2.0,
        score_weight: float = 0.5,
        decoder_loss_weight: float = 1.0,
        discriminative_weight: float = 1.0,
        auxiliary_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        delta_var: float = 0.5,
        delta_dist: float = 2.5,
        discriminative_reg_weight: float = 1e-3,
        warmup_epochs: int = 30,
        discriminative_clip: float = 10.0,
        loss_chunk_size: int = 64,
    ):
        super().__init__()
        if loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive")
        self.register_buffer("class_weights", torch.tensor(class_weights))
        self.ignore_index = ignore_index
        self.semantic_ce_weight = semantic_ce_weight
        self.semantic_lovasz_weight = semantic_lovasz_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.score_weight = score_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.discriminative_weight = discriminative_weight
        self.auxiliary_weight = auxiliary_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.discriminative_reg_weight = discriminative_reg_weight
        self.warmup_epochs = warmup_epochs
        self.discriminative_clip = discriminative_clip
        self.loss_chunk_size = loss_chunk_size
        self.lovasz = LovaszLoss(mode="multiclass", ignore_index=ignore_index)

    def semantic_loss(
        self, semantic_logits: torch.Tensor, segment: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Equation 16; CE consumes logits, unlike reference-code NLL."""

        if (segment != self.ignore_index).any():
            loss_ce = F.cross_entropy(
                semantic_logits,
                segment.long(),
                weight=self.class_weights.to(semantic_logits),
                ignore_index=self.ignore_index,
            )
            loss_lovasz = self.lovasz(semantic_logits, segment.long())
        else:
            loss_ce = semantic_logits.sum() * 0
            loss_lovasz = semantic_logits.sum() * 0
        loss = (
            self.semantic_ce_weight * loss_ce
            + self.semantic_lovasz_weight * loss_lovasz
        )
        return loss, loss_ce, loss_lovasz

    def discriminative_loss(
        self,
        embeddings: torch.Tensor,
        instance: torch.Tensor,
        counts: torch.Tensor,
        current_epoch: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding_list = embeddings.split(counts.tolist())
        instance_list = instance.split(counts.tolist())
        terms = [
            discriminative_loss_single(
                embedding,
                instance_,
                delta_var=self.delta_var,
                delta_dist=self.delta_dist,
            )
            for embedding, instance_ in zip(embedding_list, instance_list)
        ]
        loss_var = torch.stack([term[0] for term in terms]).mean()
        loss_dist = torch.stack([term[1] for term in terms]).mean()
        loss_reg = torch.stack([term[2] for term in terms]).mean()
        loss = loss_var + loss_dist + self.discriminative_reg_weight * loss_reg
        if current_epoch >= self.warmup_epochs // 2:
            loss = loss.clamp(max=self.discriminative_clip)
        return loss, {
            "loss_discriminative_var": loss_var,
            "loss_discriminative_dist": loss_dist,
            "loss_discriminative_reg": loss_reg,
        }

    @staticmethod
    def build_instance_targets(
        target: dict, dtype: torch.dtype
    ) -> InstanceTargets:
        """Build the paper's ISA one-to-many assignment once per sample."""

        seed_instance = target["seed_instance"]
        tree_instance = target["tree_instance"]
        valid_queries = seed_instance >= 0
        query_masks = (
            (seed_instance[:, None] == tree_instance[None, :])
            & valid_queries[:, None]
        ).to(dtype)
        gt_ids = torch.unique(tree_instance[tree_instance >= 0])
        gt_masks = (tree_instance[None, :] == gt_ids[:, None]).to(dtype)
        return InstanceTargets(valid_queries, query_masks, gt_masks)

    def instance_layer_loss(
        self, prediction: dict[str, torch.Tensor], targets: InstanceTargets
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask_logits = prediction["mask_logits"]
        score_logits = prediction["score_logits"]
        probabilities = mask_logits.sigmoid()

        loss_dice = soft_dice_loss(
            probabilities, targets.query_masks, targets.valid_queries
        )
        loss_focal = sigmoid_focal_loss(
            mask_logits,
            probabilities,
            targets.query_masks,
            targets.valid_queries,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            chunk_size=self.loss_chunk_size,
        )
        score_target = soft_iou_score_target(probabilities, targets.gt_masks)
        loss_score = F.mse_loss(
            score_logits.sigmoid(), score_target.detach()
        )

        loss = (
            self.dice_weight * loss_dice
            + self.focal_weight * loss_focal
            + self.score_weight * loss_score
        )
        return loss, {
            "loss_mask_dice": loss_dice,
            "loss_mask_focal": loss_focal,
            "loss_mask_score": loss_score,
        }

    def instance_loss(self, predictions: list[dict]) -> tuple[torch.Tensor, dict]:
        final_losses = []
        final_terms = []
        auxiliary_losses = []
        for target in predictions:
            targets = self.build_instance_targets(target, target["mask_logits"].dtype)
            final_loss, terms = self.instance_layer_loss(target, targets)
            final_losses.append(final_loss)
            final_terms.append(terms)

            auxiliary = target.get("aux_outputs", [])
            if auxiliary:
                auxiliary_losses.append(
                    torch.stack(
                        [
                            self.instance_layer_loss(prediction, targets)[0]
                            for prediction in auxiliary
                        ]
                    ).sum()
                )
            else:
                auxiliary_losses.append(final_loss * 0)

        loss_final = torch.stack(final_losses).mean()
        loss_auxiliary = torch.stack(auxiliary_losses).mean()
        loss = loss_final + self.auxiliary_weight * loss_auxiliary
        reduced = {
            key: torch.stack([terms[key] for terms in final_terms]).mean()
            for key in final_terms[0]
        }
        reduced["loss_mask_aux"] = loss_auxiliary
        return loss, reduced

    def forward(
        self,
        semantic_logits: torch.Tensor,
        embeddings: torch.Tensor,
        predictions: list[dict],
        segment: torch.Tensor,
        instance: torch.Tensor,
        counts: torch.Tensor,
        current_epoch: int,
        instance_active: bool,
    ) -> dict[str, torch.Tensor]:
        loss_semantic, loss_ce, loss_lovasz = self.semantic_loss(
            semantic_logits, segment
        )
        loss_discriminative, output = self.discriminative_loss(
            embeddings, instance, counts, current_epoch
        )
        if instance_active:
            decoded_predictions = [
                prediction
                for prediction in predictions
                if not prediction.get("instance_decode_skipped", False)
            ]
            if decoded_predictions:
                loss_instance, instance_terms = self.instance_loss(
                    decoded_predictions
                )
            else:
                loss_instance = semantic_logits.sum() * 0
                instance_terms = {
                    "loss_mask_dice": loss_instance,
                    "loss_mask_focal": loss_instance,
                    "loss_mask_score": loss_instance,
                    "loss_mask_aux": loss_instance,
                }
        else:
            loss_instance = semantic_logits.sum() * 0
            instance_terms = {
                "loss_mask_dice": loss_instance,
                "loss_mask_focal": loss_instance,
                "loss_mask_score": loss_instance,
                "loss_mask_aux": loss_instance,
            }

        total = (
            loss_semantic
            + self.decoder_loss_weight * loss_instance
            + self.discriminative_weight * loss_discriminative
        )
        output.update(instance_terms)
        output.update(
            loss=total,
            loss_semantic=loss_semantic,
            loss_semantic_ce=loss_ce,
            loss_semantic_lovasz=loss_lovasz,
            loss_instance=loss_instance,
            loss_discriminative=loss_discriminative,
        )
        return output

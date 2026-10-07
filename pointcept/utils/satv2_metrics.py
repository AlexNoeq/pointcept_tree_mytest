"""Paper metrics for class-agnostic individual-tree segmentation."""

from __future__ import annotations

import numpy as np


def semantic_confusion_matrix(prediction, target, num_classes=3, ignore_index=-1):
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("semantic prediction and target shapes differ")
    valid = (
        (target != ignore_index)
        & (target >= 0)
        & (target < num_classes)
        & (prediction >= 0)
        & (prediction < num_classes)
    )
    encoded = target[valid].astype(np.int64) * num_classes + prediction[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def semantic_iou(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    intersection = np.diag(confusion)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan),
        where=union > 0,
    )
    return iou, float(np.nanmean(iou))


def masks_to_instance_labels(masks, scores, num_points=None):
    """Resolve overlapping masks by confidence (local ``max_conf`` merge)."""

    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if masks.ndim != 2 or masks.shape[0] != len(scores):
        raise ValueError("masks must be [K, N] and align with scores")
    if num_points is None:
        num_points = masks.shape[1]
    labels = np.zeros(num_points, dtype=np.int64)
    score_by_id = {}
    for instance_id, index in enumerate(np.argsort(-scores), start=1):
        support = masks[index] & (labels == 0)
        if support.any():
            labels[support] = instance_id
            score_by_id[instance_id] = float(scores[index])
    return labels, score_by_id


def instance_metrics(
    prediction,
    target,
    score_by_id=None,
    iou_threshold=0.5,
):
    """Compute global detection counts and per-plot MWCov ingredients.

    Instances use positive IDs; zero and negative values are background. Each
    prediction keeps its highest-IoU target. Confidence ordering makes duplicate
    predictions false positives after the first accepted match.
    """

    prediction = np.asarray(prediction).reshape(-1).astype(np.int64, copy=False)
    target = np.asarray(target).reshape(-1).astype(np.int64, copy=False)
    if prediction.shape != target.shape:
        raise ValueError("instance prediction and target shapes differ")

    gt_ids, gt_counts = np.unique(target[target > 0], return_counts=True)
    pred_ids, pred_counts = np.unique(prediction[prediction > 0], return_counts=True)
    num_gt, num_pred = len(gt_ids), len(pred_ids)

    best_gt_index = np.full(num_pred, -1, dtype=np.int64)
    best_pred_iou = np.zeros(num_pred, dtype=np.float64)
    best_gt_iou = np.zeros(num_gt, dtype=np.float64)
    both = (target > 0) & (prediction > 0)
    if both.any() and num_gt and num_pred:
        pair_pred = np.searchsorted(pred_ids, prediction[both])
        pair_gt = np.searchsorted(gt_ids, target[both])
        pair_code = pair_pred * num_gt + pair_gt
        codes, intersections = np.unique(pair_code, return_counts=True)
        pair_pred = codes // num_gt
        pair_gt = codes % num_gt
        unions = pred_counts[pair_pred] + gt_counts[pair_gt] - intersections
        ious = intersections / np.maximum(unions, 1)
        for pred_index, gt_index, iou in zip(pair_pred, pair_gt, ious):
            if iou > best_pred_iou[pred_index]:
                best_pred_iou[pred_index] = iou
                best_gt_index[pred_index] = gt_index
            if iou > best_gt_iou[gt_index]:
                best_gt_iou[gt_index] = iou

    if score_by_id is None:
        score_by_id = {}
    scores = np.array(
        [score_by_id.get(int(instance_id), 0.0) for instance_id in pred_ids]
    )
    order = np.argsort(-scores, kind="stable")
    matched_gt = set()
    true_positive = 0
    false_positive = 0
    for pred_index in order:
        gt_index = int(best_gt_index[pred_index])
        if (
            gt_index >= 0
            and best_pred_iou[pred_index] >= iou_threshold
            and gt_index not in matched_gt
        ):
            matched_gt.add(gt_index)
            true_positive += 1
        else:
            false_positive += 1
    false_negative = num_gt - true_positive
    weighted_coverage = (
        float(np.sum(gt_counts * best_gt_iou) / np.sum(gt_counts)) if num_gt else np.nan
    )
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "num_gt": num_gt,
        "num_pred": num_pred,
        "weighted_coverage": weighted_coverage,
    }


def summarize_detection(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

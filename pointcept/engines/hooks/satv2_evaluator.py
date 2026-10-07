"""Combined semantic/tree-instance evaluator for SegmentAnyTreeV2."""

from __future__ import annotations

import numpy as np
import torch

import pointcept.utils.comm as comm
from pointcept.utils.satv2_metrics import (
    instance_metrics,
    masks_to_instance_labels,
    semantic_confusion_matrix,
    semantic_iou,
    summarize_detection,
)

from .builder import HOOKS
from .default import HookBase


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@HOOKS.register_module()
class SaTV2Evaluator(HookBase):
    def __init__(self, iou_threshold=0.5, evaluate_test=False):
        self.iou_threshold = iou_threshold
        self.evaluate_test = evaluate_test

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.evaluate(self.trainer.val_loader, split="val", update_checkpoint=True)
            if self.evaluate_test:
                test_loader = getattr(self.trainer, "test_loader", None)
                if test_loader is None:
                    raise RuntimeError(
                        "SaTV2Evaluator(evaluate_test=True) requires cfg.evaluate_test=True "
                        "and a data.test split."
                    )
                self.evaluate(test_loader, split="test", update_checkpoint=False)

    def evaluate(self, loader=None, split="val", update_checkpoint=True):
        if loader is None:
            loader = self.trainer.val_loader
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start SaTv2 {} Evaluation >>>>>>>>>>>>>>>>".format(split)
        )
        self.trainer.model.eval()
        num_classes = int(self.trainer.cfg.data.num_classes)
        local = {
            "confusion": np.zeros((num_classes, num_classes), dtype=np.int64),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "coverage": [],
            "loss_sum": 0.0,
            "scenes": 0,
            "skipped_samples": [],
        }
        for index, input_dict in enumerate(loader):
            local["skipped_samples"].extend(
                input_dict.pop("_skipped_samples", [])
            )
            if input_dict.pop("_skip_batch", False):
                continue
            if len(input_dict["offset"]) != 1:
                raise ValueError("SaTV2Evaluator requires validation batch size 1")
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.trainer.model(input_dict)

            semantic = output["pred_semantic"]
            masks = output["pred_masks"]
            scores = output["pred_scores"]
            target_semantic = input_dict["segment"]
            target_instance = input_dict["instance"]
            if "inverse" in input_dict:
                inverse = input_dict["inverse"].long()
                semantic = semantic[inverse]
                masks = masks[:, inverse.cpu()]
                target_semantic = input_dict["origin_segment"]
                target_instance = input_dict["origin_instance"]

            semantic = _to_numpy(semantic)
            target_semantic = _to_numpy(target_semantic)
            masks = _to_numpy(masks)
            scores = _to_numpy(scores)
            target_instance = _to_numpy(target_instance)
            prediction, score_by_id = masks_to_instance_labels(masks, scores)
            metrics = instance_metrics(
                prediction,
                target_instance,
                score_by_id=score_by_id,
                iou_threshold=self.iou_threshold,
            )
            local["confusion"] += semantic_confusion_matrix(
                semantic,
                target_semantic,
                num_classes=num_classes,
                ignore_index=self.trainer.cfg.data.ignore_index,
            )
            for key in ("tp", "fp", "fn"):
                local[key] += metrics[key]
            if np.isfinite(metrics["weighted_coverage"]):
                local["coverage"].append(metrics["weighted_coverage"])
            local["loss_sum"] += float(output["loss"].item())
            local["scenes"] += 1
            self.trainer.logger.info(
                "{} scene [{}/{}], loss {:.4f}".format(
                    split.capitalize(), index + 1, len(loader), output["loss"].item()
                )
            )

        comm.synchronize()
        gathered = comm.gather(local, dst=0)
        if not comm.is_main_process():
            return
        confusion = sum(item["confusion"] for item in gathered)
        tp = sum(item["tp"] for item in gathered)
        fp = sum(item["fp"] for item in gathered)
        fn = sum(item["fn"] for item in gathered)
        coverages = [value for item in gathered for value in item["coverage"]]
        scene_count = sum(item["scenes"] for item in gathered)
        loss_sum = sum(item["loss_sum"] for item in gathered)
        skipped_samples = [
            sample
            for item in gathered
            for sample in item["skipped_samples"]
        ]
        if skipped_samples:
            self.trainer.logger.info(
                "Skipped %d empty %s samples: %s",
                len(skipped_samples),
                split,
                ", ".join(
                    f"{sample.get('name') or '<unnamed>'}:"
                    f"{sample.get('reason', 'empty_sample')}"
                    f"(points={sample.get('points', 0)})"
                    for sample in skipped_samples
                ),
            )
        detection = summarize_detection(tp, fp, fn)
        class_iou, mean_iou = semantic_iou(confusion)
        coverage = float(np.mean(coverages)) if coverages else float("nan")
        self.trainer.logger.info(
            "SaTv2 {} result: precision/recall/F1/Cov/mIoU "
            "{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
                split,
                detection["precision"],
                detection["recall"],
                detection["f1"],
                coverage,
                mean_iou,
            )
        )
        for class_index, name in enumerate(self.trainer.cfg.data.names):
            self.trainer.logger.info(
                f"{split.capitalize()} class {class_index}-{name}: "
                f"IoU {class_iou[class_index]:.4f}"
            )
        self.trainer.storage.put_scalar(
            f"{split}_loss", loss_sum / max(scene_count, 1)
        )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                f"{split}/loss", loss_sum / max(scene_count, 1), current_epoch
            )
            self.trainer.writer.add_scalar(
                f"{split}/f1", detection["f1"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"{split}/mIoU", mean_iou, current_epoch
            )
        if update_checkpoint:
            self.trainer.comm_info["current_metric_value"] = detection["f1"]
            self.trainer.comm_info["current_metric_name"] = "F1"
        self.trainer.logger.info(
            "<<<<<<<<<<<<<<<< End SaTv2 {} Evaluation <<<<<<<<<<<<<<<<".format(split)
        )

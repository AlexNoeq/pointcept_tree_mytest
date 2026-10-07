"""Pointcept tester reporting the metrics used by SegmentAnyTreeV2."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

import pointcept.utils.comm as comm
from pointcept.datasets import collate_fn
from pointcept.utils.satv2_metrics import (
    instance_metrics,
    masks_to_instance_labels,
    semantic_confusion_matrix,
    semantic_iou,
    summarize_detection,
)

from .test import TESTERS, TesterBase


@TESTERS.register_module()
class SaTV2Tester(TesterBase):
    def __init__(self, iou_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)

    def test(self):
        if self.test_loader.batch_size != 1:
            raise ValueError("SaTV2Tester requires test batch size 1")
        self.model.eval()
        num_classes = int(self.cfg.data.num_classes)
        scenes = []
        skipped_samples = []
        for data_dict in self.test_loader:
            skipped_samples.extend(data_dict.pop("_skipped_samples", []))
            if data_dict.pop("_skip_batch", False):
                continue
            name = data_dict["name"][0]
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(data_dict)
            semantic = output["pred_semantic"]
            masks = output["pred_masks"]
            target_semantic = data_dict["segment"]
            target_instance = data_dict["instance"]
            if "inverse" in data_dict:
                inverse = data_dict["inverse"].long()
                semantic = semantic[inverse]
                masks = masks[:, inverse.cpu()]
                target_semantic = data_dict["origin_segment"]
                target_instance = data_dict["origin_instance"]
            masks = masks.numpy()
            scores = output["pred_scores"].numpy()
            prediction, score_by_id = masks_to_instance_labels(masks, scores)
            scenes.append(
                {
                    "name": name,
                    "confusion": semantic_confusion_matrix(
                        semantic.cpu().numpy(),
                        target_semantic.cpu().numpy(),
                        num_classes=num_classes,
                        ignore_index=self.cfg.data.ignore_index,
                    ),
                    "instance": instance_metrics(
                        prediction,
                        target_instance.cpu().numpy(),
                        score_by_id=score_by_id,
                        iou_threshold=self.iou_threshold,
                    ),
                }
            )
            self.logger.info(f"Tested {name}")

        comm.synchronize()
        gathered = comm.gather(scenes, dst=0)
        gathered_skipped = comm.gather(skipped_samples, dst=0)
        if not comm.is_main_process():
            return
        scenes = [scene for rank_scenes in gathered for scene in rank_scenes]
        skipped_samples = [
            sample
            for rank_samples in gathered_skipped
            for sample in rank_samples
        ]
        if skipped_samples:
            self.logger.info(
                "Skipped %d empty test samples: %s",
                len(skipped_samples),
                ", ".join(
                    f"{sample.get('name') or '<unnamed>'}:"
                    f"{sample.get('reason', 'empty_sample')}"
                    f"(points={sample.get('points', 0)})"
                    for sample in skipped_samples
                ),
            )
        confusion = sum(
            (scene["confusion"] for scene in scenes),
            np.zeros((num_classes, num_classes), dtype=np.int64),
        )
        tp = sum(scene["instance"]["tp"] for scene in scenes)
        fp = sum(scene["instance"]["fp"] for scene in scenes)
        fn = sum(scene["instance"]["fn"] for scene in scenes)
        coverages = [
            scene["instance"]["weighted_coverage"]
            for scene in scenes
            if np.isfinite(scene["instance"]["weighted_coverage"])
        ]
        detection = summarize_detection(tp, fp, fn)
        class_iou, mean_iou = semantic_iou(confusion)
        result = {
            **detection,
            "coverage": float(np.mean(coverages)) if coverages else None,
            "semantic_iou": class_iou.tolist(),
            "semantic_miou": mean_iou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_scenes": len(scenes),
        }
        os.makedirs(self.cfg.save_path, exist_ok=True)
        output_path = os.path.join(self.cfg.save_path, "satv2_metrics.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        self.logger.info("SaTv2 metrics: " + json.dumps(result))

"""Normalization layers with sparse-feature edge-case handling."""

import torch.nn.functional as F
from torch import nn


class SparseBatchNorm1d(nn.BatchNorm1d):
    """Keep BatchNorm behavior except for a singleton sparse feature row.

    Sparse convolutions flatten their active voxels to ``[N, C]``. BatchNorm
    cannot estimate batch statistics when ``N == 1`` during training, so that
    one case uses the layer's existing running statistics. Inheriting from
    ``BatchNorm1d`` keeps parameter names and buffers compatible with existing
    checkpoints.
    """

    def forward(self, input):
        if (
            self.training
            and input.ndim == 2
            and input.shape[0] == 1
            and self.track_running_stats
            and self.running_mean is not None
            and self.running_var is not None
        ):
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(input)

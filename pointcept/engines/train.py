"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path
import itertools

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry
from pointcept.datasets.dataloader import DistributedImbalancedSampler

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.test_loader = None
        if getattr(self.cfg, "evaluate_test", False):
            self.logger.info("=> Building test dataset & dataloader for training-time evaluation ...")
            self.test_loader = self.build_test_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self._gradient_accumulation_counter = 0
        self._empty_sample_log_interval = max(
            int(getattr(self.cfg.model, "sparse_log_interval", 100)), 1
        )
        self._pending_empty_samples = []

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        if input_dict is None:
            input_dict = {"_skip_batch": True, "_skipped_samples": []}
        local_skip_batch = bool(input_dict.get("_skip_batch", False))
        self._pending_empty_samples.extend(input_dict.get("_skipped_samples", []))
        if comm.get_world_size() > 1:
            # Keep all DDP ranks on the same forward/backward schedule.
            has_valid_sample = torch.tensor(
                0 if local_skip_batch else 1,
                dtype=torch.int32,
                device="cuda",
            )
            dist.all_reduce(has_valid_sample, op=dist.ReduceOp.MIN)
            global_batch_valid = bool(has_valid_sample.item())
        else:
            global_batch_valid = not local_skip_batch
        if not global_batch_valid:
            self.comm_info["model_output_dict"] = {}
            self._report_skipped_samples()
            return

        input_dict.pop("_skip_batch", None)
        input_dict.pop("_skipped_samples", None)
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Only clear gradients on first accumulation step
        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()

        # Forward pass
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = (
                output_dict["loss"] / self.cfg.gradient_accumulation_steps
            )  # scale loss

        # Backward pass
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._gradient_accumulation_counter += 1

        # Perform optimizer step only when enough gradients have accumulated
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            if self.cfg.enable_amp:
                self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.scaler.step(self.optimizer)

                # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
                # Fix torch warning scheduler step before optimizer step.
                scale = self.scaler.get_scale()
                self.scaler.update()
                if scale <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.optimizer.step()
                self.scheduler.step()

            # Reset grad accumulation counter
            self._gradient_accumulation_counter = 0

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict
        self._report_skipped_samples()

    def _report_skipped_samples(self, force=False):
        if (
            not force
            and (self.comm_info["iter"] + 1) % self._empty_sample_log_interval != 0
        ):
            return
        gathered = comm.all_gather(self._pending_empty_samples)
        skipped_samples = [
            record for records in gathered for record in records
        ]
        if skipped_samples and comm.is_main_process():
            summary = ", ".join(
                f"{record.get('name') or '<unnamed>'}:"
                f"{record.get('reason', 'empty_sample')}"
                f"(points={record.get('points', 0)})"
                for record in skipped_samples
            )
            self.logger.info(
                "Skipped empty input samples in the last %d training batches: %s",
                self._empty_sample_log_interval,
                summary,
            )
        self._pending_empty_samples.clear()

    def after_epoch(self):
        self._report_skipped_samples(force=True)
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_test_loader(self):
        """Build an optional test loader used by evaluation hooks during training."""
        test_data = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
        else:
            test_sampler = None
        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )
        return torch.utils.data.DataLoader(
            test_data,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            persistent_workers=self.cfg.num_worker_per_gpu > 0,
        )

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader)
            * self.cfg.eval_epoch
            // self.cfg.gradient_accumulation_steps
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("PartialSampledTrainer")
class PartialSampledTrainer(Trainer):
    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        sampled_dataset_index = self.cfg.data.sampled_dataset_index
        sampled_dataset_limit = self.cfg.data.sampled_dataset_limit

        train_sampler = DistributedImbalancedSampler(
            dataset=train_data,
            sampled_dataset_index=sampled_dataset_index,
            sampled_dataset_limit=sampled_dataset_limit,
            num_replicas=comm.get_world_size(),
            rank=comm.get_rank(),
            shuffle=True,
            seed=self.cfg.seed if self.cfg.seed is not None else 0,
        )

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader

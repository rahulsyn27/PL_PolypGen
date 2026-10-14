from copy import deepcopy
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from data.utils.datasets import BaseDataset
from src.utils.functional import evaluate_model, get_optimal_cuda_device
from src.utils.metrics import Metrics
from src.utils.models import DecoupledModel


class FedPerClient:
    def __init__(
        self,
        model: DecoupledModel,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler],
        args: DictConfig,
        dataset: BaseDataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
    ):
        self.client_id: int = None
        self.args = args
        if device is None:
            self.device = get_optimal_cuda_device(use_cuda=self.args.common.use_cuda)
        else:
            self.device = device
        self.dataset = dataset
        self.model = model.to(self.device)
        self.regular_model_params: OrderedDict[str, torch.Tensor]
        self.personal_params_name: list[str] = []
        self.regular_params_name = list(key for key, _ in self.model.named_parameters())
        if self.args.common.buffers == "local":
            self.personal_params_name.extend(
                [name for name, _ in self.model.named_buffers()]
            )
        elif self.args.common.buffers == "drop":
            self.init_buffers = deepcopy(OrderedDict(self.model.named_buffers()))

        self.optimizer = optimizer_cls(params=self.model.parameters())
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.init_lr_scheduler_state: dict = None
        self.lr_scheduler_cls = None
        if lr_scheduler_cls is not None:
            self.lr_scheduler_cls = lr_scheduler_cls
            self.lr_scheduler = self.lr_scheduler_cls(optimizer=self.optimizer)
            self.init_lr_scheduler_state = deepcopy(self.lr_scheduler.state_dict())

        # [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices
        # Please don't bother with the [0], which is only for avoiding raising runtime error by setting Subset(indices=[]) with `DataLoader(shuffle=True)`
        self.trainset = Subset(self.dataset, indices=[0])
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.args.common.batch_size, shuffle=True
        )
        self.valloader = DataLoader(self.valset, batch_size=self.args.common.batch_size)
        self.testloader = DataLoader(
            self.testset, batch_size=self.args.common.batch_size
        )
        self.testing = False

        self.local_epoch = self.args.common.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.eval_results = {}

        self.return_diff = return_diff

    def load_data_indices(self):
        """This function is for loading data indices for No.`self.client_id`
        client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train_with_eval(self):
        """Wraps `fit()` with `evaluate()` and collect model evaluation
        results.

        A model evaluation results dict: {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        eval_results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }

        eval_results["before"] = evaluate_model(
            self.model, self.trainloader, self.valloader, self.testloader, self.device
        )
        self.fit()
        eval_results["after"] = evaluate_model(
            self.model, self.trainloader, self.valloader, self.testloader, self.device
        )
        return eval_results

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def finetune(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.test.client.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def upload(self):
        if self.return_diff:
            return self.regular_model_params
        else:
            return self.model.state_dict()

    def refresh(self):
        self.model.load_state_dict(self.regular_model_params)
        self.optimizer.load_state_dict(self.init_optimizer_state)
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)


from collections import OrderedDict


class FedRepClient(FedPerClient):
    def __init__(self, **commons):
        super().__init__(**commons)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for E in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                if E < self.local_epoch - self.args.fedrep.train_body_epoch:
                    self.model.base.zero_grad()
                else:
                    self.model.classifier.zero_grad()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def finetune(self):
        self.model.train()
        self.dataset.train()
        full_model = False
        if full_model:
            # fine-tune the full model
            for E in range(self.args.common.test.client.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # freeze body, train head
                    if (
                        E
                        < self.args.common.test.client.finetune_epoch
                        - self.args.fedrep.train_body_epoch
                    ):
                        self.model.base.zero_grad()
                    # freeze head, train body
                    else:
                        self.model.classifier.zero_grad()
                    self.optimizer.step()
        else:
            # fine-tune the classifier only
            for _ in range(self.args.common.test.client.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.model.base.zero_grad()
                    self.optimizer.step()
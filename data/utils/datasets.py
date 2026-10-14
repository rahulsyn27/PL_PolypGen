import json
import os
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
import torch
import torchvision
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io.image import ImageReadMode, read_image
from src.utils.polypgen import PolypGenDataset


class BaseDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        classes: List[int],
        train_data_transform: Optional[transforms.Compose] = None,
        train_target_transform: Optional[transforms.Compose] = None,
        test_data_transform: Optional[transforms.Compose] = None,
        test_target_transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data = data
        self.targets = targets
        self.classes = classes
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        self.test_data_transform = test_data_transform
        self.test_target_transform = test_target_transform
        self.data_transform = self.train_data_transform
        self.target_transform = self.train_target_transform

        # rescale data to fit in [0, 1.0] if needed
        self._rescale_data()

    def _rescale_data(self):
        max_val = self.data.max()
        if max_val > 1.0:
            self.data /= 255.0

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets

    def train(self):
        self.data_transform = self.train_data_transform
        self.target_transform = self.train_target_transform

    def eval(self):
        self.data_transform = self.test_data_transform
        self.target_transform = self.test_target_transform

    def __len__(self):
        return len(self.targets)


class USPS(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.USPS(root / "raw", True, download=True)
        test_part = torchvision.datasets.USPS(root / "raw", False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long()
        test_targets = torch.Tensor(test_part.targets).long()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(10)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class SVHN(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.SVHN(root / "raw", "train", download=True)
        test_part = torchvision.datasets.SVHN(root / "raw", "test", download=True)
        train_data = torch.Tensor(train_part.data).float()
        test_data = torch.Tensor(test_part.data).float()
        train_targets = torch.Tensor(train_part.labels).long()
        test_targets = torch.Tensor(test_part.labels).long()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(10)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class CIFAR10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(10)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class CIFAR100(BaseDataset):
    def __init__(
        self,
        root,
        args,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        data = torch.cat([train_data, test_data])
        targets = torch.cat([train_targets, test_targets])
        classes = list(range(100))

        super_class = False
        if isinstance(args, (Namespace, DictConfig)):
            super_class = args.super_class
        elif isinstance(args, dict):
            super_class = args["super_class"]

        if super_class:
            # super_class: [sub_classes]
            CIFAR100_SUPER_CLASS = {
                0: ["beaver", "dolphin", "otter", "seal", "whale"],
                1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
                3: ["bottle", "bowl", "can", "cup", "plate"],
                4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                5: ["clock", "keyboard", "lamp", "telephone", "television"],
                6: ["bed", "chair", "couch", "table", "wardrobe"],
                7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                8: ["bear", "leopard", "lion", "tiger", "wolf"],
                9: ["cloud", "forest", "mountain", "plain", "sea"],
                10: ["bridge", "castle", "house", "road", "skyscraper"],
                11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
                13: ["crab", "lobster", "snail", "spider", "worm"],
                14: ["baby", "boy", "girl", "man", "woman"],
                15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            }
            mapping = {}
            for super_cls, sub_cls in CIFAR100_SUPER_CLASS.items():
                for cls in sub_cls:
                    mapping[cls] = super_cls
            new_targets = []
            for cls in targets:
                new_targets.append(mapping[train_part.classes[cls]])
            targets = torch.tensor(new_targets, dtype=torch.long)
            classes = list(range(20))

        super().__init__(
            data=data,
            targets=targets,
            classes=classes,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


DATASETS: Dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "svhn": SVHN,
    "usps": USPS,
    "polypgen": PolypGenDataset,
}
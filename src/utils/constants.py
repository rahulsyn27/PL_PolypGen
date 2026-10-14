import json
import os
from enum import Enum
from pathlib import Path

from torch import optim

FLBENCH_ROOT = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = FLBENCH_ROOT / "out"
TEMP_DIR = FLBENCH_ROOT / "temp"


class MODE(Enum):
    SERIAL = 0
    PARALLEL = 1


DEFAULTS = {
    "method": "fedavg",
    "dataset": {"name": "cifar10"},
    "model": {
        "name": "lenet5",
        "use_torchvision_pretrained_weights": True,
        "external_model_weights_path": None,
    },
    "lr_scheduler": {
        "name": None,
        "step_size": 10,
        "gamma": 0.1,
        "T_max": 10,
        "eta_min": 0,
        "factor": 0.3334,
        "total_iters": 5,
        "mode": "min",
        "patience": 10,
        "threshold": 1.0e-4,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 0,
        "eps": 1.0e-8,
        "last_epoch": -1,
    },
    "optimizer": {
        "name": "sgd",
        "lr": 0.01,
        "dampening": 0,
        "weight_decay": 0,
        "momentum": 0,
        "alpha": 0.99,
        "nesterov": False,
        "betas": [0.9, 0.999],
        "amsgrad": False,
    },
    "mode": "serial",
    "parallel": {
        "ray_cluster_addr": None,
        "num_cpus": None,
        "num_gpus": None,
        "num_workers": 2,
    },
    "common": {
        "seed": 42,
        "join_ratio": 0.1,
        "global_epoch": 100,
        "local_epoch": 5,
        "batch_size": 32,
        "reset_optimizer_on_global_epoch": True,
        "straggler_ratio": 0,
        "straggler_min_local_epoch": 0,
        "buffers": "global",
        "client_side_evaluation": True,
        "test": {
            "client": {
                "interval": 100,
                "finetune_epoch": 0,
                "train": False,
                "val": False,
                "test": True,
            },
            "server": {
                "interval": -1,
                "train": False,
                "val": False,
                "test": False,
                "model_in_train_mode": False,
            },
        },
        "verbose_gap": 10,
        "monitor": None,
        "use_cuda": True,
        "save_log": True,
        "save_model": False,
        "save_learning_curve_plot": True,
        "save_metrics": True,
        "delete_useless_run": True,
    },
}


INPUT_CHANNELS = {
    "cifar10": 3,
    "cifar100": 3,
    "svhn": 3,
    "usps": 1,
    "polypgen": 3,
}


# (C, H, W)
DATA_SHAPE = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "svhn": (3, 32, 32),
    "usps": (1, 16, 16),
    "polypgen": (3, 224, 224),
}

NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "usps": 10,
    "polypgen": 2,
}


DATA_MEAN = {
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "svhn": [0.4377, 0.4438, 0.4728],
    "usps": [0.1700],
    "polypgen": [0.485, 0.456, 0.406],
}


DATA_STD = {
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "svhn": [0.1201, 0.1231, 0.1052],
    "usps": [0.2581],
    "polypgen": [0.229, 0.224, 0.225],
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}

LR_SCHEDULERS = {
    "step": optim.lr_scheduler.StepLR,
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "constant": optim.lr_scheduler.ConstantLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
}
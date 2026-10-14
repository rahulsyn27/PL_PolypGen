from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig

from src.client.fedavg import FedAvgClient
from src.server.fedavg import FedAvgServer
from src.server.fednova import fednova_aggregate


class FedNovaServer(FedAvgServer):
    algorithm_name = "FedNova"
    all_model_params_personalized = False
    return_diff = False
    client_cls = FedAvgClient

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.client_taus = {}  # client_id -> tau (local steps)

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        return parser.parse_args(args_list)

    def package(self, client_id: int):
        """Package parameters for client, including local_epoch as tau."""
        package = super().package(client_id)
        package["tau"] = self.client_local_epoches[client_id]
        return package

    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """FedNova aggregation: normalize by tau before weighted averaging."""
        client_weights = [package["weight"] for package in client_packages.values()]
        client_taus = [package.get("tau", self.client_local_epoches[cid]) 
                       for cid in client_packages.keys()]
        
        # Convert to list of state_dicts for fednova_aggregate
        client_updates = []
        if self.return_diff:
            for package in client_packages.values():
                client_updates.append(package["model_params_diff"])
        else:
            for package in client_packages.values():
                client_updates.append(package["regular_model_params"])
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # FedNova aggregation
        aggregated = fednova_aggregate(client_updates, client_taus, normalized_weights)
        
        if self.return_diff:
            for name, global_param in self.public_model_params.items():
                global_param.data -= aggregated[name]
        else:
            for name, global_param in self.public_model_params.items():
                global_param.data = aggregated[name]
        
        self.model.load_state_dict(self.public_model_params, strict=False)
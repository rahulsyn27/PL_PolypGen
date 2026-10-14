from typing import Dict, List, Any, Optional
from copy import deepcopy
import torch
import torch.nn as nn

from src.server.fednova import fednova_aggregate


def fedavg_aggregate(
    client_params_list: List[Dict[str, torch.Tensor]],
    sample_weights_list: List[float],
) -> Dict[str, torch.Tensor]:
    """
    Standard FedAvg parameter aggregation using dataset sample counts as weights.
    """
    total_samples = float(sum(sample_weights_list))
    num_clients = len(client_params_list)
    if total_samples <= 0:
        weights = [1.0 / num_clients] * num_clients
    else:
        weights = [n / total_samples for n in sample_weights_list]

    aggregated: Dict[str, torch.Tensor] = {}
    param_keys = list(client_params_list[0].keys())

    for key in param_keys:
        accum = None
        for i in range(num_clients):
            if key in client_params_list[i]:
                tensor = client_params_list[i][key].float()
                w = weights[i]
                if accum is None:
                    accum = w * tensor
                else:
                    accum += w * tensor
        if accum is not None:
            aggregated[key] = accum.to(dtype=client_params_list[0][key].dtype)

    return aggregated


class FedAvgServer:
    """
    FedAvg Server supporting standard sample-weighted aggregation and
    delegation to FedNova (Fix P3) when steps_completed is provided.
    """
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model

    def aggregate(
        self,
        client_packages: List[Dict[str, Any]],
        use_fednova: bool = False,
    ) -> None:
        """
        Aggregate client weights. If use_fednova=True, normalizes each client's
        weight update by steps_completed before summing.
        """
        if not client_packages:
            return

        client_params = [pkg["model_params"] for pkg in client_packages]
        sample_weights = [float(pkg.get("sample_count", 1.0)) for pkg in client_packages]

        if use_fednova:
            steps_completed = [pkg.get("steps_completed", 1) for pkg in client_packages]
            global_dict = self.global_model.state_dict()
            aggregated = fednova_aggregate(
                client_params_list=client_params,
                steps_completed_list=steps_completed,
                sample_weights_list=sample_weights,
                global_model_params=global_dict,
            )
        else:
            aggregated = fedavg_aggregate(client_params, sample_weights)

        current_state = self.global_model.state_dict()
        for key, val in aggregated.items():
            if key in current_state:
                current_state[key].copy_(val)
        self.global_model.load_state_dict(current_state)

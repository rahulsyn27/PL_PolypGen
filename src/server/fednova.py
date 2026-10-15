from typing import Dict, List, Any, Optional
from copy import deepcopy
import torch
import torch.nn as nn


def fednova_aggregate(
    client_params_list: List[Dict[str, torch.Tensor]],
    steps_completed_list: List[int],
    sample_weights_list: List[float],
    global_model_params: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    FedNova Server Aggregation (Fix P3).
    
    Standard FedAvg uses dataset sizes for weighted averaging, which causes clients with
    more local steps (due to H3 system heterogeneity) to dominate and drift the global objective.
    
    FedNova dynamically normalizes each client's weight update by their exact number of local
    steps (steps_completed) before computing the weighted sum, scaling by effective steps tau_eff.
    
    Args:
        client_params_list: List of state_dict dicts from participating clients (W_{t+1}^k)
        steps_completed_list: List of exact optimizer steps tau_k completed by each client (H3)
        sample_weights_list: List of dataset sample counts n_k for each client
        global_model_params: Optional state_dict of current global model W_t
    
    Returns:
        Aggregated global model state_dict W_{t+1}
    """
    num_clients = len(client_params_list)
    assert num_clients > 0, "No client parameters provided for aggregation."
    assert len(steps_completed_list) == num_clients, "Mismatched steps_completed and client params count."
    assert len(sample_weights_list) == num_clients, "Mismatched sample_weights and client params count."

    # Normalized dataset proportions p_k = n_k / sum(n_i)
    total_samples = float(sum(sample_weights_list))
    if total_samples <= 0:
        p_k = [1.0 / num_clients] * num_clients
    else:
        p_k = [n / total_samples for n in sample_weights_list]

    # Effective local steps: tau_eff = sum_k (p_k * tau_k)
    # Ensure minimum tau of 1 to avoid division by zero
    taus = [max(1, int(tau)) for tau in steps_completed_list]
    tau_eff = sum(p * tau for p, tau in zip(p_k, taus))

    # Identify parameter keys present across clients
    param_keys = list(client_params_list[0].keys())
    aggregated_params: Dict[str, torch.Tensor] = {}

    # If global_model_params is provided, use exact gradient displacement formulation:
    # Delta W = tau_eff * sum_k [ p_k * (W_t - W_k) / tau_k ]
    # W_{t+1} = W_t - Delta W
    if global_model_params is not None:
        for key in param_keys:
            w_global = global_model_params[key].float()
            delta_accum = torch.zeros_like(w_global)

            for i in range(num_clients):
                if key in client_params_list[i]:
                    w_client = client_params_list[i][key].float()
                    # Displacement d_k = (W_t - W_k) / tau_k
                    displacement = (w_global - w_client) / float(taus[i])
                    delta_accum += p_k[i] * displacement

            # Global step
            aggregated_params[key] = (w_global - tau_eff * delta_accum).to(
                dtype=global_model_params[key].dtype
            )
    else:
        # Direct normalized weight averaging:
        # Effective weight for client i: alpha_i = p_k[i] * (tau_eff / tau_i)
        # Normalizing so weights sum to 1 across clients
        effective_weights = [p * (tau_eff / tau) for p, tau in zip(p_k, taus)]
        sum_eff = sum(effective_weights)
        if sum_eff > 0:
            effective_weights = [w / sum_eff for w in effective_weights]

        for key in param_keys:
            accum = None
            total_w = 0.0
            for i in range(num_clients):
                if key in client_params_list[i]:
                    tensor = client_params_list[i][key].float()
                    weight = effective_weights[i]
                    if accum is None:
                        accum = weight * tensor
                    else:
                        accum += weight * tensor
                    total_w += weight

            if accum is not None and total_w > 0:
                aggregated_params[key] = (accum / total_w).to(
                    dtype=client_params_list[0][key].dtype
                )
            else:
                aggregated_params[key] = client_params_list[0][key].clone()

    return aggregated_params


class FedNovaServer:
    """
    FedNova Server implementing Fix P3.
    Intercepts client updates with steps_completed and executes normalized aggregation.
    """
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model

    def aggregate(
        self,
        client_packages: List[Dict[str, Any]],
        p1_fedrep: bool = False,
        p2_fedbn: bool = False,
    ) -> None:
        """
        Modified aggregate() accepting steps_completed (tracked by H3 module) from each client.
        Dynamically divides each client's weight update by their exact number of local steps.
        
        Args:
            client_packages: List of dicts, each containing:
                             - 'model_params': client state_dict
                             - 'steps_completed': local steps (tau)
                             - 'sample_count': dataset size
            p1_fedrep: Whether Fix P1 is enabled (classifier heads stay local)
            p2_fedbn: Whether Fix P2 is enabled (batchnorm stats stay local)
        """
        if not client_packages:
            return

        client_params_list = [pkg["model_params"] for pkg in client_packages]
        steps_completed_list = [pkg.get("steps_completed", 1) for pkg in client_packages]
        sample_weights_list = [float(pkg.get("sample_count", 1.0)) for pkg in client_packages]

        global_dict = self.global_model.state_dict()
        aggregated = fednova_aggregate(
            client_params_list=client_params_list,
            steps_completed_list=steps_completed_list,
            sample_weights_list=sample_weights_list,
            global_model_params=global_dict,
        )

        # Update global model parameters without overwriting layers intended to be skipped
        current_state = self.global_model.state_dict()
        for key, param in aggregated.items():
            if key in current_state:
                current_state[key].copy_(param)
        self.global_model.load_state_dict(current_state)

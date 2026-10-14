from copy import deepcopy
from typing import Dict, Any, List, Optional
import torch


def is_head_param(key: str) -> bool:
    """Check if parameter belongs to the final classification head (P1 - FedRep)."""
    head_indicators = ["head", "linear_head", "classifier", "fc"]
    return any(indicator in key.lower() for indicator in head_indicators)


def is_bn_param(key: str) -> bool:
    """Check if parameter belongs to Batch Normalization layers (P2 - FedBN)."""
    bn_indicators = ["bn", "running_mean", "running_var", "num_batches_tracked"]
    return any(indicator in key.lower() for indicator in bn_indicators)


def package_client_weights(
    model: torch.nn.Module,
    client_id: int,
    sample_count: int,
    steps_completed: int,
    p1_fedrep: bool = False,
    p2_fedbn: bool = False,
) -> Dict[str, Any]:
    """
    Package client weights by selectively filtering state_dict before sending to the server.
    
    Args:
        model: Trained client model
        client_id: ID of the client
        sample_count: Number of training samples held by the client
        steps_completed: Number of local optimizer steps taken (tracked by H3 module)
        p1_fedrep: If True, keep classification head local (exclude from transmitted dict)
        p2_fedbn: If True, keep BN layers local (exclude from transmitted dict)
    
    Returns:
        Dictionary payload containing filtered state_dict, sample count, and steps_completed
    """
    raw_state_dict = model.state_dict()
    filtered_state_dict = {}

    for key, val in raw_state_dict.items():
        # P1: Keep classification head private
        if p1_fedrep and is_head_param(key):
            continue
        # P2: Keep Batch Normalization private
        if p2_fedbn and is_bn_param(key):
            continue

        filtered_state_dict[key] = val.detach().cpu().clone()

    return {
        "client_id": client_id,
        "sample_count": sample_count,
        "steps_completed": steps_completed,
        "model_params": filtered_state_dict,
    }


def unpack_server_weights(
    model: torch.nn.Module,
    server_params: Dict[str, torch.Tensor],
    p1_fedrep: bool = False,
    p2_fedbn: bool = False,
) -> None:
    """
    Update client model with aggregated server weights without overwriting private local layers.
    """
    current_state = model.state_dict()
    for key, val in server_params.items():
        if p1_fedrep and is_head_param(key):
            continue
        if p2_fedbn and is_bn_param(key):
            continue
        if key in current_state:
            current_state[key].copy_(val)
    model.load_state_dict(current_state)

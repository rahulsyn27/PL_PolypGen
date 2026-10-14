def filter_state_dict(state_dict: dict, apply_P1: bool = False, apply_P2: bool = False) -> dict:
    """
    Filter a PyTorch model's state_dict for federated learning weight packaging.
    
    Args:
        state_dict: Model's state dictionary
        apply_P1: If True, remove 'fc.' and 'classifier.' keys (FedRep - keep representation only)
        apply_P2: If True, remove 'bn.', 'running_mean', 'running_var' keys (FedBN - keep BN stats local)
    
    Returns:
        Filtered state dictionary
    """
    filtered = {}
    
    for key, value in state_dict.items():
        if apply_P1:
            if 'fc.' in key or 'classifier.' in key:
                continue
        
        if apply_P2:
            if 'bn.' in key or 'running_mean' in key or 'running_var' in key:
                continue
        
        filtered[key] = value
    
    return filtered
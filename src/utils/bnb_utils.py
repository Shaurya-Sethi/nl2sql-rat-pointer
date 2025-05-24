import torch
import logging

logger = logging.getLogger(__name__)

def ensure_bnb_state(optimizer, device=None):
    """
    For every parameter in a bitsandbytes optimizer that lacks momentum ('state1'),
    variance ('state2'), or 'step' buffers, create them to prevent 'state1' KeyError
    and TypeError with 'step'.

    Args:
        optimizer: The bitsandbytes optimizer instance.
        device: Optional. The device to place new tensors on. If None, uses param's device.
    """
    created_buffers_count = 0
    if not hasattr(optimizer, 'param_groups'):
        logger.warning("Optimizer does not have param_groups. Cannot ensure BnB state.")
        return

    for group_idx, group in enumerate(optimizer.param_groups):
        if "params" not in group:
            logger.warning(f"Group {group_idx} in optimizer does not have 'params' key. Skipping.")
            continue
            
        for p_idx, p in enumerate(group["params"]):
            if p not in optimizer.state:
                # This can happen if parameters are added to the optimizer *after*
                # its initialization and before any steps or state loading.
                # bitsandbytes usually initializes state lazily on the first step or upon loading.
                optimizer.state[p] = {}
                logger.debug(f"Param {p_idx} in group {group_idx} (LR: {group.get('lr', 'N/A')}) "
                             f"was not in optimizer.state. Initializing empty state dict for it.")

            state = optimizer.state[p]
            param_device = device or p.device
            
            current_param_created_any_buffer = False

            if "state1" not in state:
                # 'state1' is typically momentum or EMA
                state["state1"] = torch.zeros_like(p.data, dtype=torch.float32, device=param_device)
                current_param_created_any_buffer = True
            
            if "state2" not in state:
                # 'state2' is typically variance or RMS
                state["state2"] = torch.zeros_like(p.data, dtype=torch.float32, device=param_device)
                current_param_created_any_buffer = True

            if "step" not in state:
                state["step"] = 0  # Use a plain Python int for step
                current_param_created_any_buffer = True
            elif isinstance(state["step"], torch.Tensor):
                # Ensure existing tensor steps are converted to int
                logger.debug(f"Converting tensor step to int for param {p_idx} in group {group_idx}.")
                state["step"] = int(state["step"].item())
                # We don't count this as a newly created buffer, but as a correction.
            
            if current_param_created_any_buffer:
                created_buffers_count +=1 # Count params for which at least one buffer was made

    if created_buffers_count > 0:
        logger.info(f"Initialised missing BnB state (state1/state2/step) for {created_buffers_count} parameter(s).")
    else:
        # This log can be noisy if called every batch, consider adjusting if needed
        logger.debug("ensure_bnb_state: All parameters checked appear to have state1/state2/step buffers, or steps were already int.") 

def cleanup_bnb_step_tensors(optimizer):
    """
    Converts any 'step' entries in the optimizer state that are tensors to Python integers.
    This is to ensure compatibility with bitsandbytes optimizers that expect int steps.
    Args:
        optimizer: The optimizer whose state needs to be cleaned up.
    """
    converted_count = 0
    if not hasattr(optimizer, 'state') or not optimizer.state:
        logger.debug("Optimizer has no state or state is empty, skipping step tensor cleanup.")
        return

    for param_state in optimizer.state.values():
        if "step" in param_state and isinstance(param_state["step"], torch.Tensor):
            try:
                param_state["step"] = int(param_state["step"].item())
                converted_count += 1
            except Exception as e:
                logger.error(f"Failed to convert tensor step to int: {e}. Current step value: {param_state['step']}")
    
    if converted_count > 0:
        logger.info(f"Converted 'step' from tensor to int for {converted_count} parameter states during checkpoint cleanup.")
    else:
        logger.debug("cleanup_bnb_step_tensors: No tensor steps found or conversion was not needed.") 
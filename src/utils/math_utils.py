"""Mathematical utilities for CVaR annealing and risk calculations."""


def get_annealed_cvar(
    current_step: int, 
    start_step: int, 
    end_step: int, 
    max_cvar: float
) -> float:
    """
    Pure function. Calculates a linearly annealed CVaR multiplier.
    - Before start_step: returns 0.0 (No penalty, pure Alpha exploration)
    - Between start_step and end_step: returns linearly scaled value
    - After end_step: returns max_cvar (Full risk protection)
    """
    if start_step >= end_step:
        return max_cvar
        
    if current_step <= start_step:
        return 0.0
        
    if current_step >= end_step:
        return max_cvar
        
    # Linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    # y1 = 0.0, y2 = max_cvar, x1 = start_step, x2 = end_step
    return 0.0 + (current_step - start_step) * (max_cvar - 0.0) / (end_step - start_step)

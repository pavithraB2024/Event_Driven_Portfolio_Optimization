"""Training utilities for learning rate scheduling and convergence helpers."""


def calculate_cooled_lr(base_lr: float, cooling_factor: float = 0.1) -> float:
    """
    Calculates a reduced learning rate for fine-tuning.
    Default cooling factor is 0.1 (10x reduction).
    """
    return float(base_lr * cooling_factor)

def should_resume_finetune(current_sharpe: float, target_sharpe: float = 2.10) -> bool:
    """
    Determines if training should continue based on target Sharpe.
    """
    return current_sharpe < target_sharpe

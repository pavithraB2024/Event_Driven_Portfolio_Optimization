import pytest
from src.utils.training_utils import calculate_cooled_lr, should_resume_finetune

def test_calculate_cooled_lr():
    # Test reduction by 10x
    assert calculate_cooled_lr(3e-4, 0.1) == pytest.approx(3e-5)
    # Test reduction by 2x
    assert calculate_cooled_lr(1e-4, 0.5) == 5e-5

def test_should_resume_finetune():
    # Should resume if sharpe is high but we want better
    assert should_resume_finetune(2.01, 2.10) is True
    # Should NOT resume if already surpassed target
    assert should_resume_finetune(2.15, 2.10) is False

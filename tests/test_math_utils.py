import pytest
from src.utils.math_utils import get_annealed_cvar

def test_get_annealed_cvar_before_start():
    # Before start_step, should be 0.0 (pure alpha)
    assert get_annealed_cvar(current_step=500, start_step=1000, end_step=2000, max_cvar=1.0) == 0.0

def test_get_annealed_cvar_after_end():
    # After end_step, should be max_cvar (full defense)
    assert get_annealed_cvar(current_step=2500, start_step=1000, end_step=2000, max_cvar=2.5) == 2.5

def test_get_annealed_cvar_during_anneal():
    # Halfway through the anneal period, should be half of max_cvar
    # Step 1500 is halfway between 1000 and 2000
    assert get_annealed_cvar(current_step=1500, start_step=1000, end_step=2000, max_cvar=1.0) == 0.5
    
    # 25% through
    assert get_annealed_cvar(current_step=1250, start_step=1000, end_step=2000, max_cvar=2.0) == 0.5

def test_get_annealed_cvar_exact_boundaries():
    # Exactly at start_step
    assert get_annealed_cvar(current_step=1000, start_step=1000, end_step=2000, max_cvar=1.0) == 0.0
    # Exactly at end_step
    assert get_annealed_cvar(current_step=2000, start_step=1000, end_step=2000, max_cvar=1.0) == 1.0

def test_get_annealed_cvar_invalid_bounds():
    # If start_step >= end_step, should return max_cvar to be safe
    assert get_annealed_cvar(current_step=1500, start_step=2000, end_step=1000, max_cvar=1.0) == 1.0

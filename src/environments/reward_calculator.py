"""Reward shaping with log-utility, DSR, and risk-adjusted components."""
import numpy as np
from typing import Tuple, List, Optional, NamedTuple, Literal

# ───────────────────────────────────────────────────────────────────
# Pure Functions — Mathematical Rewards and Penalties
# ───────────────────────────────────────────────────────────────────

def differential_sharpe_ratio(
    return_t: float,
    ema_return: float,
    ema_return_sq: float,
    eta: float = 0.01,
) -> Tuple[float, float, float]:
    """Compute the Differential Sharpe Ratio (Moody & Saffell, 2001)."""
    new_ema_return = ema_return + eta * (return_t - ema_return)
    new_ema_return_sq = ema_return_sq + eta * (return_t ** 2 - ema_return_sq)

    denom = (new_ema_return_sq - new_ema_return ** 2)
    if denom <= 1e-12:
        dsr = 0.0
    else:
        dsr_numer = (
            new_ema_return_sq * (return_t - ema_return)
            - 0.5 * new_ema_return * (return_t ** 2 - ema_return_sq)
        )
        dsr = dsr_numer / (denom ** 1.5 + 1e-12)

    return float(np.clip(dsr, -5.0, 5.0)), new_ema_return, new_ema_return_sq


def weight_diversification_bonus(
    weights: np.ndarray,
    coeff: float = 0.001,
) -> float:
    """Entropy-based bonus that rewards diversified allocation."""
    n = len(weights)
    if n <= 1:
        return 0.0
    w = np.clip(weights, 1e-8, 1.0)
    entropy = -np.sum(w * np.log(w))
    max_entropy = np.log(n)
    return coeff * (entropy / max_entropy) if max_entropy > 0 else 0.0


def log_utility_reward(old_value: float, new_value: float) -> float:
    """Calculate logarithmic rate of return."""
    if new_value > 0 and old_value > 0:
        return float(np.log(new_value) - np.log(old_value))
    return -10.0  # Penalty for bankruptcy


class RewardConfig(NamedTuple):
    """Typed reward configuration — validated at construction."""
    base_type: Literal['log_utility', 'sharpe', 'return', 'risk_adjusted', 'aggressive', 'dsr']
    oscillation_beta: float = 0.001  
    drawdown_penalty: float = 0.002  
    cvar_kappa: float = 0.0          
    dd_threshold: float = 0.05       

    def validate(self) -> None:
        assert self.base_type in ('log_utility', 'sharpe', 'return', 'risk_adjusted', 'aggressive', 'dsr')
        assert 0.0 <= self.oscillation_beta <= 0.1, 'oscillation_beta must be in [0, 0.1]'
        assert 0.0 <= self.drawdown_penalty <= 0.1, 'drawdown_penalty must be in [0, 0.1]'
        assert 0.0 <= self.cvar_kappa <= 1.0, 'cvar_kappa must be in [0, 1]'
        total_penalty = self.oscillation_beta + self.drawdown_penalty + self.cvar_kappa
        assert total_penalty <= 1.0, (
            f'Total penalty weight {total_penalty:.3f} > 1.0. '
            'Excessive penalisation will dominate the return signal.'
        )

# ───────────────────────────────────────────────────────────────────
# Reward Calculator Architecture
# ───────────────────────────────────────────────────────────────────

class BaseRewardCalculator:
    """
    Base class for reward calculation.
    Handles universal penalties (oscillation, drawdown) and diversification bonuses.
    Subclasses should override `_calculate_base`.
    """
    def __init__(
        self,
        config: RewardConfig,
        diversification_coeff: float = 0.0,
        horizon_blend_lambda: float = 1.0
    ):
        self.config = config
        self.diversification_coeff = diversification_coeff
        self.horizon_blend_lambda = horizon_blend_lambda

    def reset(self):
        """Reset internal state (if any)."""
        pass

    def calculate(
        self,
        old_value: float,
        new_value: float,
        portfolio_return: float,
        transaction_costs: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        prev_prev_weights: np.ndarray,
        peak_value: float,
        initial_balance: float,
        reward_history: List[float],
        risk_free_rate: float
    ) -> float:
        # Aggressive reward skips all penalties
        if self.config.base_type == 'aggressive':
            return (new_value - old_value) / max(old_value, 1.0)

        # 1. Base Reward
        base_reward = self._calculate_base(
            old_value, new_value, portfolio_return, transaction_costs,
            reward_history, risk_free_rate, peak_value
        )

        # 2. Universal Penalties
        oscillation = np.sum((weights - 2.0 * prev_weights + prev_prev_weights) ** 2)
        osc_penalty = self.config.oscillation_beta * oscillation

        current_dd = (peak_value - new_value) / peak_value if peak_value > 0 else 0.0
        dd_penalty = self.config.drawdown_penalty * max(0.0, current_dd - self.config.dd_threshold)

        div_bonus = weight_diversification_bonus(weights, self.diversification_coeff)

        step_reward = base_reward - osc_penalty - dd_penalty + div_bonus

        # 3. Blending
        if self.horizon_blend_lambda < 1.0:
            horizon_return = (new_value - initial_balance) / initial_balance
            return self.horizon_blend_lambda * step_reward + (1.0 - self.horizon_blend_lambda) * horizon_return

        return step_reward

    def _calculate_base(
        self,
        old_value: float,
        new_value: float,
        portfolio_return: float,
        transaction_costs: float,
        reward_history: List[float],
        risk_free_rate: float,
        peak_value: float
    ) -> float:
        raise NotImplementedError


class LogUtilityRewardCalculator(BaseRewardCalculator):
    def _calculate_base(self, old_value, new_value, *args, **kwargs) -> float:
        return log_utility_reward(old_value, new_value)


class ReturnRewardCalculator(BaseRewardCalculator):
    def _calculate_base(self, old_value, new_value, portfolio_return, transaction_costs, *args, **kwargs) -> float:
        return portfolio_return - (transaction_costs / old_value)


class SharpeRewardCalculator(BaseRewardCalculator):
    def _calculate_base(self, old_value, new_value, portfolio_return, transaction_costs, reward_history, risk_free_rate, *args, **kwargs) -> float:
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_return = portfolio_return - daily_rf
        recent_returns = reward_history[-20:] if len(reward_history) >= 20 else reward_history
        volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.01
        return excess_return / (volatility + 1e-6)


class RiskAdjustedRewardCalculator(BaseRewardCalculator):
    def _calculate_base(self, old_value, new_value, portfolio_return, transaction_costs, reward_history, risk_free_rate, peak_value, **kwargs) -> float:
        if new_value > 0:
            base_reward = np.log(new_value) - np.log(old_value)
        else:
            base_reward = -10.0
        current_dd = (peak_value - new_value) / peak_value if peak_value > 0 else 0.0
        excess_dd = max(0.0, current_dd - self.config.dd_threshold)
        base_reward -= self.config.cvar_kappa * excess_dd
        return base_reward


class DSRRewardCalculator(BaseRewardCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dsr_ema_return = 0.0
        self._dsr_ema_return_sq = 0.0

    def reset(self):
        self._dsr_ema_return = 0.0
        self._dsr_ema_return_sq = 0.0

    def _calculate_base(self, old_value, new_value, portfolio_return, *args, **kwargs) -> float:
        dsr_val, self._dsr_ema_return, self._dsr_ema_return_sq = differential_sharpe_ratio(
            return_t=portfolio_return,
            ema_return=self._dsr_ema_return,
            ema_return_sq=self._dsr_ema_return_sq,
            eta=0.01,
        )
        return dsr_val

def get_reward_calculator(
    reward_type: str,
    config: Optional[RewardConfig] = None,
    oscillation_penalty: float = 0.001,
    drawdown_penalty: float = 0.002,
    diversification_coeff: float = 0.0,
    horizon_blend_lambda: float = 1.0
) -> BaseRewardCalculator:
    if config is None:
        config = RewardConfig(
            base_type=reward_type, # type: ignore
            oscillation_beta=oscillation_penalty,
            drawdown_penalty=drawdown_penalty,
            cvar_kappa=0.0,
            dd_threshold=0.05
        )
    
    mapping = {
        'log_utility': LogUtilityRewardCalculator,
        'return': ReturnRewardCalculator,
        'sharpe': SharpeRewardCalculator,
        'risk_adjusted': RiskAdjustedRewardCalculator,
        'dsr': DSRRewardCalculator,
        'aggressive': BaseRewardCalculator # Aggressive logic is completely in Base
    }
    
    if config.base_type not in mapping:
        raise ValueError(f"Unknown reward_type: {config.base_type}")
        
    return mapping[config.base_type](
        config=config,
        diversification_coeff=diversification_coeff,
        horizon_blend_lambda=horizon_blend_lambda
    )

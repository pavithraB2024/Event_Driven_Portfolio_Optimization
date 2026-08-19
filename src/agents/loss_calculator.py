"""Loss computation strategies for SAC critic and actor updates."""
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

# ───────────────────────────────────────────────────────────────────
# Pure Functions — SAC Loss Mathematics
# ───────────────────────────────────────────────────────────────────

def compute_nlp_dynamic_alpha(
    base_alpha: torch.Tensor,      # [B, 1] or scalar
    event_emb: torch.Tensor,       # [B, D]
    multiplier: float = 0.1,
) -> torch.Tensor:
    """Scale SAC entropy alpha by NLP event severity. Pure function."""
    severity = torch.mean(torch.abs(event_emb), dim=-1, keepdim=True)
    out = base_alpha + multiplier * severity
    if out.dim() == 2 and out.shape[0] == 1 and out.shape[1] == 1:
        out = out.squeeze()
    return out.clamp(min=0.01, max=1.0)


def compute_cvar_loss(quantiles, targets, tau=0.1, multiplier=5.0):
    """
    Compute Risk-Sensitive CVaR loss for IQN Critics.
    Pure function: Weights errors in the lower tau quantiles more heavily.
    """
    errors = (quantiles - targets)**2
    B, N = errors.shape
    
    tail_idx = max(int(tau * N), 1)
    
    tail_errors = errors[:, :tail_idx] * multiplier
    normal_errors = errors[:, tail_idx:]
    
    combined = torch.cat([tail_errors, normal_errors], dim=1)
    return combined.mean()


def compute_volatility_exploration_penalty(
    log_std: torch.Tensor,
    vix_norm: torch.Tensor,
    penalty_coef: float = 0.1
) -> torch.Tensor:
    """
    Penalizes high action variance (exploration) when market volatility is high.
    Pure function.
    """
    penalty = penalty_coef * (log_std.exp() * vix_norm)
    return penalty.mean()

# ───────────────────────────────────────────────────────────────────
# Loss Calculator Interfaces
# ───────────────────────────────────────────────────────────────────

class LossCalculator(ABC):
    @abstractmethod
    def compute_z_target(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        z_next: torch.Tensor,
        next_log_probs: torch.Tensor,
        alpha: torch.Tensor,
        gamma: float,
        nextgen_ctx: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def compute_critic_loss(
        self,
        z1_pred: torch.Tensor,
        z2_pred: torch.Tensor,
        z_target: torch.Tensor,
        tau1: torch.Tensor,
        tau2: torch.Tensor,
        weights: Optional[torch.Tensor],
        quantile_huber_loss_fn: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns iqn_loss, td_err1, td_err2"""
        pass

    @abstractmethod
    def compute_policy_loss(
        self,
        alpha: torch.Tensor,
        log_probs: torch.Tensor,
        min_q_new: torch.Tensor,
        weights: Optional[torch.Tensor],
        nextgen_ctx: Optional[Dict[str, Any]],
        stock_log_std: Optional[torch.Tensor]
    ) -> torch.Tensor:
        pass


class StandardSACLossCalculator(LossCalculator):
    def compute_z_target(self, rewards, dones, z_next, next_log_probs, alpha, gamma, nextgen_ctx):
        z_target = rewards + (1 - dones) * gamma * (z_next - alpha * next_log_probs)
        return z_target, alpha

    def compute_critic_loss(self, z1_pred, z2_pred, z_target, tau1, tau2, weights, quantile_huber_loss_fn):
        iqn_loss1, td_err1 = quantile_huber_loss_fn(
            z1_pred, z_target, tau=tau1, weights=weights,
            cvar_tau=1.0, cvar_multiplier=1.0
        )
        iqn_loss2, td_err2 = quantile_huber_loss_fn(
            z2_pred, z_target, tau=tau2, weights=weights,
            cvar_tau=1.0, cvar_multiplier=1.0
        )
        return iqn_loss1 + iqn_loss2, td_err1, td_err2

    def compute_policy_loss(self, alpha, log_probs, min_q_new, weights, nextgen_ctx, stock_log_std):
        alpha_val = alpha.detach() if isinstance(alpha, torch.Tensor) else alpha
        if alpha_val.dim() > 1 and alpha_val.size(1) > 1:
            alpha_val = alpha_val.mean()
            
        policy_td = (alpha_val * log_probs - min_q_new).squeeze(-1)
        if weights is not None:
            return (policy_td * weights).mean()
        return policy_td.mean()


class NextGenCVaRLossCalculator(LossCalculator):
    def __init__(self, cvar_tau: float, cvar_multiplier: float, use_nlp_alpha: bool, nlp_alpha_multiplier: float, device: torch.device):
        self.cvar_tau = cvar_tau
        self.cvar_multiplier = cvar_multiplier
        self.use_nlp_alpha = use_nlp_alpha
        self.nlp_alpha_multiplier = nlp_alpha_multiplier
        self.device = device

    def compute_z_target(self, rewards, dones, z_next, next_log_probs, alpha, gamma, nextgen_ctx):
        if self.use_nlp_alpha:
            if nextgen_ctx and 'event_emb' in nextgen_ctx and nextgen_ctx['event_emb'] is not None:
                event_emb = nextgen_ctx['event_emb']
            else:
                event_emb = torch.zeros(rewards.shape[0], 768, device=self.device)
            alpha = compute_nlp_dynamic_alpha(alpha, event_emb, self.nlp_alpha_multiplier)
            
        z_target = rewards + (1 - dones) * gamma * (z_next - alpha * next_log_probs)
        return z_target, alpha

    def compute_critic_loss(self, z1_pred, z2_pred, z_target, tau1, tau2, weights, quantile_huber_loss_fn):
        iqn_loss1, td_err1 = quantile_huber_loss_fn(
            z1_pred, z_target, tau=tau1, weights=weights,
            cvar_tau=self.cvar_tau, cvar_multiplier=self.cvar_multiplier
        )
        iqn_loss2, td_err2 = quantile_huber_loss_fn(
            z2_pred, z_target, tau=tau2, weights=weights,
            cvar_tau=self.cvar_tau, cvar_multiplier=self.cvar_multiplier
        )
        return iqn_loss1 + iqn_loss2, td_err1, td_err2

    def compute_policy_loss(self, alpha, log_probs, min_q_new, weights, nextgen_ctx, stock_log_std):
        alpha_val = alpha.detach() if isinstance(alpha, torch.Tensor) else alpha
        if alpha_val.dim() > 1 and alpha_val.size(1) > 1:
            alpha_val = alpha_val.mean()
            
        policy_td = (alpha_val * log_probs - min_q_new).squeeze(-1)
        if weights is not None:
            policy_loss = (policy_td * weights).mean()
        else:
            policy_loss = policy_td.mean()

        if nextgen_ctx and 'vix' in nextgen_ctx and 'vix_mean' in nextgen_ctx and stock_log_std is not None:
            v_vix = nextgen_ctx['vix']
            v_vix_mean = nextgen_ctx['vix_mean']
            vix_norm = torch.sigmoid((v_vix - v_vix_mean) / (v_vix_mean.clamp(min=1e-6))).view(-1, 1)
            vol_penalty = compute_volatility_exploration_penalty(stock_log_std, vix_norm, penalty_coef=0.1)
            policy_loss = policy_loss + vol_penalty

        return policy_loss

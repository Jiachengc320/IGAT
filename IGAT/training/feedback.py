"""Attack-outcome coefficients from Eqs. (9) and (10)."""
import torch


def attack_feedback_weights(logits_adv, labels, beta):
    attack_success = (logits_adv.argmax(dim=1) != labels)
    success_rate = attack_success.float().mean().detach()
    weights = torch.where(
        attack_success,
        1 + beta * success_rate,
        1 - beta * (1 - success_rate),
    ).detach()
    return weights


def feedback_weighted_average(paired_term, weights):
    return (weights * paired_term).mean()

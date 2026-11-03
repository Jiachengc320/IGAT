"""Gaussian Lie coordinates and paired regularization, Eqs. (5) and (6)."""
import torch


def gaussian_lie_coordinates(features, eps=1e-6):
    """Return log covariance and transformed means from manuscript Eq. (5).

    Spatial positions supply the samples for each channel's Gaussian. Adding
    eps to population variance ensures strictly positive diagonal covariance.
    """
    mu = features.mean(dim=[2, 3])
    variance = features.var(dim=[2, 3], unbiased=False) + eps
    delta = variance - 1.0
    log_variance = torch.log(variance)
    near_one = delta.abs() < 1e-3
    # Both torch.where branches must have finite derivatives, including at s=1.
    denominator = torch.where(near_one, torch.ones_like(delta), delta)
    quotient = log_variance / denominator
    small_delta = torch.where(near_one, delta, torch.zeros_like(delta))
    continuation = (1 - small_delta / 2 + small_delta.square() / 3
                    - small_delta.pow(3) / 4 + small_delta.pow(4) / 5)
    h = torch.where(near_one, continuation, quotient)
    return log_variance, mu * h


def paired_lie_term(x_natural, x_adv, eps=1e-6):
    """Paired Gaussian-coordinate inner product, Eq. (6)."""
    log_nat, mean_nat = gaussian_lie_coordinates(x_natural, eps)
    log_adv, mean_adv = gaussian_lie_coordinates(x_adv, eps)
    covariance_term = 0.5 * (log_nat * log_adv).sum(dim=1)
    mean_term = (mean_nat * mean_adv).sum(dim=1)
    return covariance_term + mean_term

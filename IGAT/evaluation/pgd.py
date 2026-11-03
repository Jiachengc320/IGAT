"""Projected-gradient-descent attack under an L-infinity perturbation budget."""
import torch
import torch.nn.functional as F


def pgd_linf_attack(model, x, y, eps, steps, step_size, random_init=True):
    """Generate adversarial inputs with projected sign-gradient ascent.

    ``x`` must contain inputs in [0, 1], and ``model`` must return class logits.
    This function does not update model parameters.
    """
    x_adv = x.detach().clone()
    if random_init:
        delta = torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv + delta, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
    return x_adv.detach()

# IG-AT: Information Geometry-aware Adversarial Training

**Code for the paper *Research on Adversarial Defense Algorithm based on Information Geometry*.**

IG-AT studies adversarial robustness through the statistical geometry of natural and adversarial representations. The method describes feature representations using Gaussian means and diagonal covariances, maps these statistics to Lie-algebra coordinates, and constructs a paired geometric regularizer. Attack outcomes determine the coefficients assigned to individual pairs.

## Repository Structure

```text
IGAT/
├── README.md
├── \_\_init\_\_.py
├── training/
│   ├── \_\_init\_\_.py
│   ├── lie.py          # Gaussian Lie coordinates and paired geometric term
│   └── feedback.py     # Attack-outcome weights and weighted averaging
└── evaluation/
    ├── \_\_init\_\_.py
    └── pgd.py          # L-infinity projected gradient descent attack
```

## Correspondence to the Paper

|Paper|Implementation|Description|
|-|-|-|
|Eq. (5)|`training/lie.py` — `gaussian\_lie\_coordinates`|Computes per-channel log-variance coordinates and covariance-transformed means.|
|Eq. (6)|`training/lie.py` — `paired\_lie\_term`|Computes the paired Gaussian-coordinate inner product for natural and adversarial features.|
|Eq. (9)|`training/feedback.py` — `attack\_feedback\_weights`|Computes sample weights from adversarial classification outcomes and the batch attack-success rate.|
|Eq. (10)|`training/feedback.py` — `feedback\_weighted\_average`|Averages the pairwise terms after applying their attack-outcome weights.|
|Eq. (11)|`evaluation/pgd.py` — `pgd\_linf\_attack`|Generates adversarial inputs using projected sign-gradient ascent under an L-infinity perturbation constraint.|

### Gaussian Lie coordinates — Eq. (5)

For a feature tensor of shape `\[batch, channels, height, width]`, the spatial dimensions provide observations for each channel. Let `mu` denote the channel mean and `s` its stabilized variance. The coordinate functions are

$$
\\ell = \\log s, \\qquad \\psi = \\mu \\odot h(s),
\\qquad h(s) = \\begin{cases}
\\frac{\\log s}{s-1}, \& s \\ne 1, \\
1, \& s = 1.
\\end{cases}
$$

The implementation uses a continuous numerical approximation near `s = 1` for stable evaluation and differentiation. These vectors correspond to the nonzero statistical coordinates of the Lie-algebra representation in Eq. (5).

### Paired geometric term — Eq. (6)

Given natural and adversarial feature statistics, the paired term is

$$
D\_{\\mathrm{sim}}(x,x^{\\mathrm{adv}})
= \\frac{1}{2}\\langle\\ell\_n,\\ell\_a\\rangle

* \\langle\\psi\_n,\\psi\_a\\rangle.
$$

`paired\_lie\_term` returns one value per sample, with the inner products summed across channels.

### Attack-outcome feedback — Eqs. (9)–(10)

For sample `i`, let `I\_i = 1` when the adversarial prediction differs from its ground-truth label, and `I\_i = 0` otherwise. For a batch of size `B`,

$$
r = \\frac{1}{B}\\sum\_{i=1}^{B} I\_i, \\qquad
w\_i = 1 + \\beta(I\_i+r-1).
$$

The weighted batch statistic is

$$
L\_{\\mathrm{sim}}^{w}
= \\frac{1}{B}\\sum\_{i=1}^{B} w\_i D\_{\\mathrm{sim}}(x\_i,x\_i^{\\mathrm{adv}}).
$$

`attack\_feedback\_weights` takes adversarial logits, ground-truth labels, and the feedback coefficient `beta`. `feedback\_weighted\_average` performs the batch mean. The feedback weights are treated as detached coefficients.

### Projected gradient descent — Eq. (11)

`pgd\_linf\_attack` uses cross-entropy gradients, an optional random initialization, projection onto the input-centered L-infinity ball, and clipping to the valid input range `\[0, 1]`. It accepts a model, inputs, labels, perturbation budget, step count, and step size; it returns adversarial inputs without updating model parameters.

## Code Updates

This repository will be updated progressively. Additional implementations, evaluation utilities, and supporting code will be provided in future updates.


import torch


def ddpm_schedules(beta1: float, beta2: float, T: int) -> dict[str, torch.Tensor]:
    """
    Pre-compute all diffusion schedules used in DDPM training and sampling.
    """

    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # Linear variance schedule:
    # beta_t increases linearly from beta1 to beta2
    # β_t = β_start + (β_end – β_start) * (t / T)
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1

    sqrt_beta_t = torch.sqrt(beta_t)

    # alpha_t = 1 - beta_t
    alpha_t = 1 - beta_t

    # Original implementation:
    # alphabar_t = torch.cumprod(alpha_t, dim=0)
    #
    # Problem:
    # Multiplying many values smaller than 1 may cause numerical underflow.
    #
    # Stable solution:
    # prod(x) = exp(sum(log(x)))
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    # ============================================================
    # Forward diffusion process: q(x_t | x_0)
    # ============================================================

    # sqrt(alphabar_t)
    sqrtab = torch.sqrt(alphabar_t)

    # sqrt(1 - alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)

    # ============================================================
    # Reverse diffusion process: p(x_{t-1} | x_t)
    # ============================================================

    # 1 / sqrt(alpha_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    # (1 - alpha_t) / sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # α_t
        "oneover_sqrta": oneover_sqrta,  # 1/sqrt(α_t)
        "sqrt_beta_t": sqrt_beta_t,  # sqrt(β_t)
        "alphabar_t": alphabar_t,  # ᾱ_t
        "sqrtab": sqrtab,  # sqrt(ᾱ_t)
        "sqrtmab": sqrtmab,  # sqrt(1 - ᾱ_t)
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-α_t)/sqrt(1-ᾱ_t)
    }


def q_sample(x_0, t, sqrtab, sqrtmab, noise=None):
    """Add Gaussian noise to image x_0 at timestep t and return noisy image x_t."""

    # Generate random Gaussian noise if not provided
    if noise is None:
        noise = torch.randn_like(x_0)

    """
    Why do we use multiple `None` dimensions?
    -----------------------------------------
    Reason:
        - x_0 has shape [batch, channels, height, width]
          Example: [32, 3, 28, 28]

        - t has shape [batch]
          Example: [32]

        - Therefore:
            noise_schedule_dict["sqrtab"][t]
          returns a tensor with shape [32]

        - This causes a shape mismatch when multiplying with x_0.

    Solution:
        Add extra singleton dimensions using `None`
        (equivalent to unsqueeze).

        Example:
            [32] -> [32, 1, 1, 1]

        This enables broadcasting across channels, height, and width.
    """
    # sqrtab = noise_schedule_dict["sqrtab"][t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    # sqrtmab = noise_schedule_dict["sqrtmab"][t].unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # sqrtab = noise_schedule_dict["sqrtab"][t, None, None, None]
    # sqrtmab = noise_schedule_dict["sqrtmab"][t, None, None, None]

    _sqrtab = sqrtab[t].view(-1, 1, 1, 1)
    _sqrtmab = sqrtmab[t].view(-1, 1, 1, 1)

    x_t = _sqrtab * x_0 + _sqrtmab * noise

    return x_t, noise

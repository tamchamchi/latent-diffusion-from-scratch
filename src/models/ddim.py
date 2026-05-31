import torch
import torch.nn as nn

from .ddpm import DDPM


class DDIM(DDPM):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
        eta: float = 0.0,
    ) -> None:
        super(DDIM, self).__init__(eps_model, betas, n_T, criterion)
        self.eta = eta

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            # predicted x_0
            x0_t = (x_i - eps * (1 - self.alphabar_t[i]).sqrt()) / self.alphabar_t[
                i
            ].sqrt()

            # direction pointing to x_i
            c1 = (
                self.eta
                * (
                    (1 - self.alphabar_t[i] / self.alphabar_t[i - 1])
                    * (1 - self.alphabar_t[i - 1])
                    / (1 - self.alphabar_t[i])
                ).sqrt()
            )

            # random noise to add
            c2 = ((1 - self.alphabar_t[i - 1]) - c1**2).sqrt()

            x_i = self.alphabar_t[i - 1].sqrt() * x0_t + c1 * z + c2 * eps

        return x_i

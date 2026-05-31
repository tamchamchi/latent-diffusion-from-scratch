import torch
import torch.nn as nn

from .utils import ddpm_schedules, q_sample


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        self.schedules = ddpm_schedules(betas[0], betas[1], n_T)

        for k, v in self.schedules.items():
            self.register_buffer(k, v)
        """
        Tại sao phải sử dụng register_buffer?
        ------------------------------------
        Lý do 1: Tự động đồng bộ Device (CPU <-> GPU).
        Khi đưa model lên GPU `model.to(cuda)`. Nếu không có register_buffer thì 

        """
        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x):
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        eps = torch.randn_like(x)

        """
        Error: RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)

        """
        x_t, _ = q_sample(x, _ts, self.sqrtab, self.sqrtmab, eps)

        predicted_eps = self.eps_model(x_t, _ts / self.n_T)

        return self.criterion(eps, predicted_eps)

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i

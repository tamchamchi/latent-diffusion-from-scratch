import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.configs import get_config
from src.models.ddpm import DDPM
from src.models.unet import NaiveUnet
from src.utils import set_random_seed, setup_logger

config = get_config()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_filepath = os.path.join(
    config.log_dir,
    f"training_{current_time}.log",
)

logger = setup_logger("train", log_file=log_filepath, level=logging.DEBUG)


def train_cifar10(config) -> None:

    eps_model = NaiveUnet(config.in_channels, config.out_channels, n_feat=config.n_feat)

    ddpm = DDPM(eps_model=eps_model, betas=config.beta, n_T=config.n_T)

    if config.load_pth is not None:
        ddpm.load_state_dict(torch.load(config.load_pth))

    ddpm.to(config.device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        config.data_dir,
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    optim = torch.optim.Adam(ddpm.parameters(), lr=config.lr)

    for i in range(config.n_epoch):
        logger.info(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for batch_idx, (x, _) in enumerate(pbar):
            optim.zero_grad()
            x = x.to(config.device)
            loss = ddpm(x)
            loss.backward()
            optim.step()

            current_loss = loss.item()

            if loss_ema is None:
                loss_ema = current_loss
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * current_loss

            pbar.set_description(f"loss: {loss_ema:.4f}")

            if batch_idx % config.log_interval == 0:
                logger.debug(
                    f"Epoch: {i} | Batch: {batch_idx}/{len(dataloader)} | "
                    f"Loss: {current_loss:.4f} | Loss EMA: {loss_ema:.4f}"
                )

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 32, 32), config.device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            logger.info(f"Saving sample image for epoch {i}...")
            save_image(grid, f"{config.output_dir}/ddpm_sample_cifar{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"{config.output_dir}/ddpm_cifar.pth")


if __name__ == "__main__":
    set_random_seed(0)
    train_cifar10(config)

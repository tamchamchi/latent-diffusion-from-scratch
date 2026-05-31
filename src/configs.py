import os

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # DDPM config
    beta: tuple[float, float] = (1e-4, 0.02)
    n_T: int = 1000

    # Unet config
    n_feat: int = 128
    in_channels: int = 3
    out_channels: int = 3

    # training config
    device: str = "cuda:0"
    n_epoch: int = 10
    batch_size: int = 512
    lr: float = 1e-5
    num_workers: int = 8
    load_pth: str | None = None
    log_interval: int = 10

    # path config
    data_dir: str = "./data"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


def get_config() -> Config:
    """
    Returns a Config instance with default values.
    You can modify this function to load config from a file or environment variables if needed.
    """
    return Config()

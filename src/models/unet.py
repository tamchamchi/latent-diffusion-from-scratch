"""
Simple Unet Structure.
This U-Net architecture is designed specifically for Diffusion Models.
The key distinguishing feature of a Diffusion U-Net is its ability to be Time-Conditioned.
"""

import torch
import torch.nn as nn


class Conv3(nn.Module):
    """
    Core Convolutional Block (Residual Block).
    The name 'Conv3' comes from the use of a 3x3 Kernel Size (the standard size for maintaining resolution).
    """

    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        # 1. Main Block (Projector): Synchronizes the number of input channels to the desired output channels.
        self.main = nn.Sequential(
            # kernel=3, stride=1, padding=1 -> Preserves the spatial dimensions (Width and Height) of the feature map.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            # GroupNorm: Performs better than BatchNorm when the batch size is small (common in generative models).
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        # 2. Conv Block: Extracts deeper features without changing the number of channels.
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        # Flag to determine whether to use a Skip Connection (Residual path).
        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First, project the input to the correct number of channels.
        x = self.main(x)

        if self.is_res:
            # If using Residual: Add the projected input to the output of the deeper conv block.
            x = x + self.conv(x)
            # Crucial Technique: Divide by 1.414 (sqrt(2)) for Variance Scaling.
            # This stabilizes the variance and prevents Exploding Gradients in very deep networks.
            return x / 1.414
        else:
            # If not using Residual: Just pass through the conv block.
            return self.conv(x)


class UnetDown(nn.Module):
    """
    Encoder Block (Downsampling): Reduces the spatial resolution and increases the number of channels.
    Helps the network learn global, abstract features.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        layers = [
            # Use Conv3 block for feature extraction.
            Conv3(in_channels, out_channels),
            # Use MaxPool2d to halve the spatial dimensions (e.g., 32x32 -> 16x16).
            nn.MaxPool2d(2),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    """
    Decoder Block (Upsampling): Restores the spatial resolution from deep features.
    Receives 'skip' signals from the Encoder branch to preserve fine-grained, high-frequency details.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            # ConvTranspose2d: Doubles the spatial dimensions (e.g., 16x16 -> 32x32).
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            # Conv3 blocks to refine the features after upsampling.
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Concatenate the current feature map (x) with the feature map from the Down branch (skip).
        # Concatenation happens along the Channel dimension (dim=1).
        # Note: 'in_channels' of this layer must equal (channels_x + channels_skip).
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class TimeSiren(nn.Module):
    """
    Time Embedding Module.
    Converts a scalar timestep 't' into a high-dimensional vector, conditioning the U-Net on the noise level.
    Uses the SIREN (Sinusoidal Representation Networks) architecture with a Sine activation function.
    """

    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()
        # Project the scalar 't' into the embedding dimension.
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape 't' into a column vector (batch_size, 1).
        x = x.view(-1, 1)
        # Apply Sine function to create periodic signals (similar to Positional Encoding in Transformers).
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    """
    The complete U-Net architecture.
    Termed 'Naive' because the Time Injection mechanism is implemented in a relatively simple, additive manner.
    """

    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat  # Base number of channels

        # Input layer: Projects the input image channels (e.g., 3 for RGB) to the base feature channels (256).
        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        # Encoder Branch (Down-path): Spatial dimensions decrease, channel counts increase.
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        # Bottleneck (The bottom of the 'U'): Compresses the feature map into a 1x1 vector representation.
        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        # Time embedding module.
        self.timeembed = TimeSiren(2 * n_feat)

        # First layer of the Decoder branch: Expands the vector back into a spatial feature map.
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        # Decoder Branch (Up-path): Gradually increases the spatial dimensions.
        # Note: in_channels is 4*n_feat because it must handle the concatenated channels from the Down branch.
        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)

        # Output layer: Projects the features back to the required image channels (e.g., 3 for RGB).
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # --- 1. Initial Processing ---
        x = self.init_conv(x)

        # --- 2. Encoder Branch (Save intermediate outputs for Skip Connections) ---
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        # --- 3. Bottleneck & Time Injection ---
        thro = self.to_vec(down3)
        # Compute the time vector and reshape it for broadcasting across spatial dimensions.
        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)

        # First Time Injection: Add the time vector directly to the bottleneck feature map.
        thro = self.up0(thro + temb)

        # --- 4. Decoder Branch (with Skip Connections) ---
        # Second Time Injection at the first Up block.
        up1 = self.up1(thro, down3) + temb
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        # --- 5. Output Layer ---
        # Global Skip Connection: Concatenate the final feature map with the original input before final projection.
        out = self.out(torch.cat((up3, x), 1))

        return out

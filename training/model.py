import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helper ----------
def _match_spatial(x, ref):
    if x.shape[-3:] != ref.shape[-3:]:
        x = F.interpolate(x, size=ref.shape[-3:], mode='trilinear', align_corners=False)
    return x

# ---------- building blocks ----------
class ResidualBlock3D(nn.Module):
    def __init__(self, channels, groups=4, dilation=1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=pad, dilation=dilation, bias=False)
        self.gn1   = nn.GroupNorm(groups, channels)
        self.act1  = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=pad, dilation=dilation, bias=False)
        self.gn2   = nn.GroupNorm(groups, channels)

    def forward(self, x):
        y = self.act1(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act1(y + x)

# ---------- UNet ----------
class FilterPredictorUNet(nn.Module):
    """
    Tiny UNet for PSF logits, 1 downsample (11x11x11 safe).
    Keeps same input/output logic as original FilterPredictor.
    """
    def __init__(self, input_channel=2, base=16, output_channel=1):
        super().__init__()
        # encoder
        self.enc1_in = nn.Sequential(
            nn.Conv3d(input_channel, base, 3, padding=1, bias=False),
            nn.GroupNorm(4, base),
            nn.PReLU(),
        )
        self.enc1 = ResidualBlock3D(base, groups=4, dilation=1)
        self.down1 = nn.Conv3d(base, base*2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock3D(base*2, groups=4, dilation=1)

        # bottleneck
        self.bott_d2 = ResidualBlock3D(base*2, groups=4, dilation=2)
        self.bott_d4 = ResidualBlock3D(base*2, groups=4, dilation=4)
        self.bott_fuse = nn.Sequential(
            nn.Conv3d(base*2*3, base*2, 1, bias=False),
            nn.GroupNorm(4, base*2),
            nn.PReLU(),
        )

        # decoder
        self.up1  = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base + base, base, 3, padding=1, bias=False),
            nn.GroupNorm(4, base),
            nn.PReLU(),
            ResidualBlock3D(base, groups=4, dilation=1)
        )

        # head (logits)
        self.head = nn.Conv3d(base, output_channel, 1)

        # --- initialization ---
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # use relu gain (PReLU has no entry)
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_mu, distances):
        if x_mu.dim() == 4:  # [B,D,H,W]
            x_mu = x_mu.unsqueeze(1)
        B, _, D, H, W = x_mu.shape
        

        expected_size = D * H * W
        if distances.dim() == 4:
            distances = distances.view(1, 1, D, H, W)
        elif distances.numel() == expected_size:
            distances = distances.view(1, 1, D, H, W)
        elif distances.numel() == B * expected_size:
            distances = distances.view(B, 1, D, H, W)
        else:
            raise ValueError(f"Distances shape mismatch: {distances.shape}")

        if distances.shape[0] == 1 and B > 1:
            distances = distances.expand(B, -1, -1, -1, -1)

        x = torch.cat((x_mu, distances), dim=1)  # same logic

        e1_in = self.enc1_in(x)
        e1 = self.enc1(e1_in)
        e2 = self.enc2(self.down1(e1))

        b2 = self.bott_d2(e2)
        b4 = self.bott_d4(e2)
        bott = self.bott_fuse(torch.cat([e2, b2, b4], dim=1))

        u1 = _match_spatial(self.up1(bott), e1)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)  # [B,1,D,H,W]
        return logits


# fastunet:

class SiLU(nn.Module):
    def forward(self, x):  # marginally faster than nn.SiLU on some builds
        return x * torch.sigmoid(x)

# -----------------------
# fast building blocks
# -----------------------
class DepthwiseSeparableConv3d(nn.Module):
    """
    Depthwise 3D conv (groups=C_in) + pointwise 1x1x1 conv
    No norm; bias=True; PReLU for cheap nonlinearity.
    """
    def __init__(self, in_ch, out_ch, k=3, stride=1, dilation=1, act='prelu'):
        super().__init__()
        pad = (k // 2) * dilation
        self.dw = nn.Conv3d(in_ch, in_ch, k, stride=stride, padding=pad,
                            dilation=dilation, groups=in_ch, bias=True)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=True)
        self.act = nn.PReLU(out_ch) if act == 'prelu' else SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(x)

class InvertedResidual3D(nn.Module):
    """
    MobileNetV2-style block for 3D:
      expand (1x1) → depthwise (3x3) → project (1x1)
    Linear bottleneck (no act on the last proj).
    Optional residual if shapes match and stride=1.
    """
    def __init__(self, in_ch, out_ch, expand=2, stride=1, dilation=1, act='prelu'):
        super().__init__()
        hid = max(in_ch * expand, 8)
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.expand = nn.Conv3d(in_ch, hid, 1, bias=True)
        self.act1   = nn.PReLU(hid) if act == 'prelu' else SiLU()
        pad = dilation
        self.dw = nn.Conv3d(hid, hid, 3, stride=stride, padding=pad,
                            dilation=dilation, groups=hid, bias=True)
        self.act2 = nn.PReLU(hid) if act == 'prelu' else SiLU()
        self.proj = nn.Conv3d(hid, out_ch, 1, bias=True)

    def forward(self, x):
        y = self.expand(x); y = self.act1(y)
        y = self.dw(y);     y = self.act2(y)
        y = self.proj(y)    # linear bottleneck
        return x + y if self.use_res else y

# -----------------------
# Fast UNet (1 down, 1 up)
# -----------------------
class FastUNet3D(nn.Module):
    """
    Fast UNet for predicting PSF logits on 31^3.
    - Input:  (x_mu, distances) with shapes [B,1,D,H,W] and [B|1,1,D,H,W]
    - Output: logits [B,1,D,H,W]
    - Architecture:
        stem → enc(IRB x2) → down(s=2) → bottleneck(IRB x2) → up(trilinear) → dec(IRB) → head(1x1)
    Notes:
    - No norms, all ops bias=True.
    - Depthwise separable / inverted residual blocks dominate.
    - Designed for D=H=W=31 but works for any small odd window (pads via _match_spatial).
    """
    def __init__(self, input_channel=2, base=16, output_channel=1, expand=2):
        super().__init__()
        C = base
        # stem
        self.stem = DepthwiseSeparableConv3d(input_channel, C, k=3, stride=1)
        # encoder @31^3
        self.enc1 = InvertedResidual3D(C,   C,   expand=expand, stride=1)
        self.enc2 = InvertedResidual3D(C,   C,   expand=expand, stride=1)
        # downsample to ~16^3 (uses depthwise stride-2)
        self.down = InvertedResidual3D(C,   2*C, expand=expand, stride=2)
        # bottleneck @~16^3
        self.b1   = InvertedResidual3D(2*C, 2*C, expand=expand, stride=1, dilation=1)
        self.b2   = InvertedResidual3D(2*C, 2*C, expand=expand, stride=1, dilation=1)
        # upsample back to 31^3 (trilinear + conv)
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # fuse skip (reduce channels before IRB to save FLOPs)
        self.fuse = nn.Conv3d(2*C + C, C, 1, bias=True)
        # decoder @31^3
        self.dec  = InvertedResidual3D(C,   C,   expand=expand, stride=1)
        # head
        self.head = nn.Conv3d(C, output_channel, 1, bias=True)

        # light init for 3D
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_mu, distances):
        # Accept [B,1,D,H,W] for both. (Also accepts flat distances; we reshape.)
        if x_mu.dim() == 4:
            x_mu = x_mu.unsqueeze(1)
        B, _, D, H, W = x_mu.shape

        # distances normalization/shape — follow your prior logic, but fast
        expected = D * H * W
        if distances.dim() == 5:
            d = distances
        elif distances.numel() == expected:
            d = distances.view(1, 1, D, H, W)
        elif distances.numel() == B * expected:
            d = distances.view(B, 1, D, H, W)
        else:
            raise ValueError(f"Distances shape mismatch: {distances.shape}")
        if d.shape[0] == 1 and B > 1:
            d = d.expand(B, -1, -1, -1, -1)

        # concatenate inputs
        x = torch.cat([x_mu, d], dim=1)

        # encoder
        s  = self.stem(x)       # [B,C,31,31,31]
        e1 = self.enc1(s)
        e1 = self.enc2(e1)
        e2 = self.down(e1)      # [B,2C,~16,~16,~16]

        # bottleneck
        b  = self.b1(e2)
        b  = self.b2(b)

        # decoder
        u  = self.up(b)                  # [B,2C,~32,~32,~32]
        u  = _match_spatial(u, e1)       # → [B,2C,31,31,31]
        cat = torch.cat([u, e1], dim=1)  # [B,3C,31,31,31]
        cat = self.fuse(cat)             # [B,C,31,31,31]
        d1  = self.dec(cat)              # [B,C,31,31,31]
        logits = self.head(d1)           # [B,1,31,31,31]
        return logits


class SimpleResidualBlock3D(nn.Module):
    """
    Residual block without norms (as provided in your snippet).
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.act   = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))) + x)

class FilterPredictor(nn.Module):
    """
    Enhanced Filter Predictor with repeating residual blocks (no norms),
    matching your snippet. Input = concat(μ, -distances).
    """
    def __init__(self, input_channel=2, in_between_channel=16, output_channel=1, num_residual_blocks=4):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channel, in_between_channel, kernel_size=3, padding=1, bias=True),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            *[SimpleResidualBlock3D(in_between_channel) for _ in range(num_residual_blocks)]
        )
        self.final_conv = nn.Conv3d(in_between_channel, output_channel, kernel_size=1, bias=True)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _shape_distances(self, distances, B, D, H, W):
        expected_size = D * H * W
        if distances.dim() == 5:
            d = distances
        elif distances.dim() == 4:
            d = distances.view(1, 1, D, H, W)
        elif distances.numel() == expected_size:
            d = distances.view(1, 1, D, H, W)
        elif distances.numel() == B * expected_size:
            d = distances.view(B, 1, D, H, W)
        else:
            raise ValueError(f"'distances' has {distances.numel()} elements, expected {expected_size} or {B * expected_size}")
        if d.shape[0] == 1 and B > 1:
            d = d.expand(B, -1, -1, -1, -1)
        return d

    def forward(self, x_mu, distances):
        # accept [B,1,D,H,W] or [B,D,H,W]
        if x_mu.dim() == 4:
            x_mu = x_mu.unsqueeze(1)
        B, _, D, H, W = x_mu.shape

        d = self._shape_distances(distances, B, D, H, W)

        combined_input = torch.cat((x_mu, d), dim=1)  # (B,2,D,H,W)

        out = self.initial_conv(combined_input)
        out = self.residual_blocks(out)
        out = self.final_conv(out)  # logits or linear filter
        return out


# experiment the model 
if __name__ == "__main__":
    model = FilterPredictor(input_channel=2, in_between_channel=16, output_channel=1, num_residual_blocks=4)
    x_mu = torch.randn(2, 1, 32, 32, 32)  # [B,1,D,H,W]
    distances = torch.randn(1, 1, 32, 32, 32)  # [1,1,D,H,W]
    logits = model(x_mu, distances)
    print(logits.shape)  # should be [2,1,32,32,32]

    # get number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

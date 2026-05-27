import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(1, channels, 1, 1)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, channels, 1, 1)))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, padding=0, bias=True)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):

    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, bias=True)
        self.sca = SCA(dw_channel // 2)
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma

class NAF_BPN_FusionNet(nn.Module):
    def __init__(self, num_basis=15, ksz=7, width=32, enc_blk_nums=[1, 1, 1], middle_blk_num=1, dec_blk_nums=[1, 1, 1]):
        super().__init__()
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_len = 2

        self.intro = nn.Conv2d(3, width, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.coeff_head = nn.Sequential(
            nn.Conv2d(width, width, 3, 1, 1),
            nn.Conv2d(width, num_basis, 1),
            nn.Softmax(dim=1)
        )

        final_chan = width * (2 ** len(enc_blk_nums))
        self.basis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(final_chan, final_chan // 2, 1),
            nn.Flatten(),
            nn.Linear(final_chan // 2, num_basis * self.burst_len * ksz * ksz)
        )

        self._init_bpn_weights()

    def _init_bpn_weights(self):
        with torch.no_grad():
            basis_linear = self.basis_head[-1]
            nn.init.constant_(basis_linear.weight, 0)
            nn.init.constant_(basis_linear.bias, 0)

            bias_view = basis_linear.bias.view(self.num_basis, self.burst_len, self.ksz ** 2)
            center_idx = (self.ksz ** 2) // 2

            bias_view[:, 0, center_idx] = 5.0
            bias_view[:, 1, center_idx] = 5.0

    def apply_basis_fusion(self, img_2d, img_3d, basis, coeffs):
        B, N, _, K, _ = basis.shape
        pad = K // 2
        src = torch.cat([img_2d, img_3d], dim=1)
        kernels = basis.view(B * N, 2, K, K)
        input_stack = src.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(1, B * N * 2, src.shape[2], src.shape[3])
        conv_res = F.conv2d(input_stack, kernels, padding=pad, groups=B * N)
        conv_res = conv_res.view(B, N, src.shape[2], src.shape[3])
        return torch.sum(conv_res * coeffs, dim=1, keepdim=True)

    def forward(self, img_2d, img_3d, noisy_t, noisy_tm1):
        diff_noise = torch.abs(noisy_t - noisy_tm1)
        md_noise = F.avg_pool2d(diff_noise, kernel_size=5, stride=1, padding=2)

        diff_algo = torch.abs(img_3d - img_2d)
        md_algo = F.avg_pool2d(diff_algo, kernel_size=5, stride=1, padding=2)

        unet_inputs = torch.cat([noisy_t, md_noise, md_algo], dim=1)

        x = self.intro(unet_inputs)

        enc_skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_skips.append(x)
            x = down(x)

        x = self.middle_blks(x)
        feat_bottleneck = x

        for decoder, up, enc_skip in zip(self.decoders, self.ups, enc_skips[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        coeffs = self.coeff_head(x)

        basis_weights = self.basis_head(feat_bottleneck)
        basis_kernels = basis_weights.view(-1, self.num_basis, self.burst_len, self.ksz, self.ksz)

        basis_kernels = F.softmax(
            basis_kernels.view(-1, self.num_basis, self.burst_len * self.ksz ** 2),
            dim=-1
        )
        basis_kernels = basis_kernels.view(-1, self.num_basis, self.burst_len, self.ksz, self.ksz)

        fusion_out = self.apply_basis_fusion(img_2d, img_3d, basis_kernels, coeffs)
        return fusion_out, md_algo

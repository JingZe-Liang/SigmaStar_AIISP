from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FusionDataset
from net import NAF_BPN_FusionNet


@dataclass
class TrainConfig:
    """训练脚本的集中配置。"""

    # =================【输入 / 输出配置】=================
    # data_path：训练集根目录。
    # checkpoint_dir：模型权重保存目录。
    # log_dir：TensorBoard 日志输出目录。
    # val_files：固定验证集文件列表。
    # 这几项决定“从哪里读训练/验证数据、往哪里写训练结果”。
    # ====================================================
    data_path: str = "./H5"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    val_files: list[str] = field(default_factory=lambda: ["./scene_1/shard_0.h5", "./scene_12/shard_0.h5"])
    num_basis: int = 15
    kernel_size: int = 7
    width: int = 32
    base_lr: float = 1e-3
    total_epochs: int = 800
    warmup_epochs: int = 20
    batch_size: int = 16
    num_workers: int = 12
    weight_decay: float = 1e-3
    grad_clip_norm: float = 0.1
    patch_size: int = 256
    lambda_grad: float = 0.5
    lambda_anchor: float = 2.0
    beta: float = 10.0
    alpha: float = 0.9998
    save_every: int = 50
    preview_every: int = 20


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class MaskedGradientLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_max = torch.max(mask.view(mask.size(0), -1), dim=1, keepdim=True)[0]
        mask_max = mask_max.unsqueeze(2).unsqueeze(3).clamp(min=1e-5)
        mask_normalized = mask / mask_max

        diff_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        diff_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        diff_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        diff_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = torch.abs(diff_pred_x - diff_target_x)
        loss_y = torch.abs(diff_pred_y - diff_target_y)
        mask_x = mask_normalized[:, :, :, 1:]
        mask_y = mask_normalized[:, :, 1:, :]
        return torch.mean(loss_x * mask_x) + torch.mean(loss_y * mask_y)


def to_srgb(x: torch.Tensor) -> torch.Tensor:
    safe_x = torch.clamp(x, min=0.0)
    return torch.where(
        safe_x <= 0.0031308,
        12.92 * safe_x,
        1.055 * torch.pow(safe_x + 1e-6, 1.0 / 2.4) - 0.055,
    )


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred_np = np.clip(pred.detach().cpu().numpy().squeeze(), 0, 1)
    target_np = np.clip(target.detach().cpu().numpy().squeeze(), 0, 1)
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
        target_np = target_np[0]
    return float(psnr_func(target_np, pred_np, data_range=1.0)), float(ssim_func(target_np, pred_np, data_range=1.0))


def evaluate_fixed_set(model: nn.Module, device: torch.device, val_files: list[str]) -> float:
    model.eval()
    all_psnr: list[float] = []

    print("\n[Validation] 开始执行固定验证集评估...")
    with torch.no_grad():
        for h5_path_str in val_files:
            h5_path = Path(h5_path_str)
            if not h5_path.exists():
                print(f"[警告] 验证文件不存在，跳过：{h5_path}")
                continue

            with h5py.File(h5_path, "r", swmr=True) as h5_file:
                num_frames = int(h5_file["clean"].shape[0])
                for frame_idx in range(num_frames):
                    # 【输入标记】离线验证时固定读取 2dnr / 3dnr / noisy / clean 四类数据。
                    img_2dnr_np = h5_file["2dnr"][frame_idx].astype(np.float32) / 4095.0
                    img_3dnr_np = h5_file["3dnr"][frame_idx].astype(np.float32) / 4095.0
                    gt_raw_np = h5_file["clean"][frame_idx].astype(np.float32) / 4095.0
                    noisy_t_np = h5_file["noisy"][frame_idx, 1, :, :].astype(np.float32) / 4095.0
                    noisy_tm1_np = (
                        h5_file["noisy"][frame_idx - 1, 1, :, :].astype(np.float32) / 4095.0
                        if frame_idx > 0
                        else noisy_t_np.copy()
                    )

                    img_2dnr = torch.from_numpy(img_2dnr_np).unsqueeze(0).unsqueeze(0).to(device)
                    img_3dnr = torch.from_numpy(img_3dnr_np).unsqueeze(0).unsqueeze(0).to(device)
                    noisy_t = torch.from_numpy(noisy_t_np).unsqueeze(0).unsqueeze(0).to(device)
                    noisy_tm1 = torch.from_numpy(noisy_tm1_np).unsqueeze(0).unsqueeze(0).to(device)

                    pred_raw, _ = model(img_2dnr, img_3dnr, noisy_t, noisy_tm1)
                    pred_srgb = np.clip(to_srgb(pred_raw).cpu().squeeze().numpy(), 0, 1)
                    gt_srgb = np.clip(to_srgb(torch.from_numpy(gt_raw_np)).numpy(), 0, 1)
                    all_psnr.append(float(psnr_func(gt_srgb, pred_srgb, data_range=1.0)))

    model.train()
    avg_psnr = float(np.mean(all_psnr)) if all_psnr else 0.0
    print(f"[Validation] 完成，平均 sRGB PSNR: {avg_psnr:.4f} dB\n")
    return avg_psnr


def train() -> None:
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # 【输出配置】训练过程中的 TensorBoard 曲线和可视化写到 log_dir。
    writer = SummaryWriter(log_dir=config.log_dir)
    model = NAF_BPN_FusionNet(num_basis=config.num_basis, ksz=config.kernel_size, width=config.width).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_epochs - config.warmup_epochs, eta_min=1e-7)

    charb_criterion = CharbonnierLoss(eps=1e-3).to(device)
    masked_grad_criterion = MaskedGradientLoss().to(device)

    dataset = FusionDataset(root_dir=config.data_path, patch_size=config.patch_size, is_training=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    global_step = 0
    best_val_psnr = 0.0

    try:
        for epoch in range(config.total_epochs):
            if epoch < config.warmup_epochs:
                current_lr = config.base_lr * ((epoch + 1) / config.warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            model.train()
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.total_epochs}", mininterval=10, ncols=110)
            last_preview_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

            for batch_data in progress_bar:
                # 【输入标记】当前 batch 的输入顺序固定为：
                # img_2dnr, img_3dnr, noisy_t, noisy_tm1, targets
                img_2dnr, img_3dnr, noisy_t, noisy_tm1, targets = [tensor.to(device) for tensor in batch_data]
                optimizer.zero_grad()

                output_raw, motion_mask = model(img_2dnr, img_3dnr, noisy_t, noisy_tm1)
                output_srgb = to_srgb(output_raw)
                targets_srgb = to_srgb(targets)
                img_2dnr_srgb = to_srgb(img_2dnr)
                img_3dnr_srgb = to_srgb(img_3dnr)

                loss_base = charb_criterion(output_srgb, targets_srgb)
                loss_grad = masked_grad_criterion(output_srgb, targets_srgb, motion_mask.detach())
                anchor_diff = output_srgb - img_2dnr_srgb
                anchor_map = torch.sqrt(anchor_diff * anchor_diff + 1e-6)
                loss_anchor = torch.mean(motion_mask.detach() * anchor_map)

                main_loss = loss_base + config.lambda_grad * loss_grad + config.lambda_anchor * loss_anchor

                with torch.no_grad():
                    loss_2d = charb_criterion(img_2dnr_srgb, targets_srgb) + config.lambda_grad * masked_grad_criterion(
                        img_2dnr_srgb, targets_srgb, motion_mask
                    )
                    loss_3d = charb_criterion(img_3dnr_srgb, targets_srgb) + config.lambda_grad * masked_grad_criterion(
                        img_3dnr_srgb, targets_srgb, motion_mask
                    )
                    baseline_reference = loss_2d + loss_3d

                current_beta = config.beta * (config.alpha ** global_step)
                display_loss = main_loss + current_beta * baseline_reference

                main_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

                writer.add_scalar("Train/Main_Loss", float(main_loss.item()), global_step)
                writer.add_scalar("Train/Display_Loss", float(display_loss.item()), global_step)
                writer.add_scalar("Train/Anchor_Loss", float(loss_anchor.item()), global_step)
                writer.add_scalar("Train/Baseline_Reference", float(baseline_reference.item()), global_step)

                global_step += 1
                running_loss += float(main_loss.item())
                last_preview_batch = (img_2dnr, img_3dnr, output_raw.detach(), targets.detach())
                progress_bar.set_postfix({"Loss": f"{main_loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.5f}"})

            if epoch >= config.warmup_epochs:
                scheduler.step()

            writer.add_scalar("Train/Epoch_Loss", running_loss / max(len(dataloader), 1), epoch + 1)

            if (epoch + 1) % config.preview_every == 0 and last_preview_batch is not None:
                img_2dnr, img_3dnr, pred_raw, targets = last_preview_batch
                pred_srgb = to_srgb(torch.clamp(pred_raw, 0, 1))
                target_srgb = to_srgb(targets)
                cur_psnr, cur_ssim = calculate_metrics(pred_srgb, target_srgb)
                writer.add_scalar("Val_Patch/PSNR", cur_psnr, epoch + 1)
                writer.add_scalar("Val_Patch/SSIM", cur_ssim, epoch + 1)

                compare_grid = vutils.make_grid(
                    torch.cat([to_srgb(img_2dnr[0:1]), to_srgb(img_3dnr[0:1]), pred_srgb[0:1], target_srgb[0:1]], dim=3),
                    normalize=False,
                )
                writer.add_image(f"Visual/Result_Epoch_{epoch + 1}", compare_grid, epoch + 1)

            if (epoch + 1) % config.save_every == 0:
                # 【输出标记】这里保存常规 checkpoint，便于中断恢复和阶段对比。
                checkpoint_path = checkpoint_dir / f"naf_bpn_sz{config.kernel_size}_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                current_val_psnr = evaluate_fixed_set(model, device, config.val_files)
                writer.add_scalar("Val_Offline/sRGB_PSNR", current_val_psnr, epoch + 1)

                if current_val_psnr > best_val_psnr:
                    best_val_psnr = current_val_psnr
                    best_checkpoint_path = checkpoint_dir / f"naf_bpn_sz{config.kernel_size}_best.pth"
                    torch.save(model.state_dict(), best_checkpoint_path)
    finally:
        writer.close()


if __name__ == "__main__":
    train()

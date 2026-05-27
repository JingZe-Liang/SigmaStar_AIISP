import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from tqdm import tqdm

from net import NAF_BPN_FusionNet


@dataclass
class TestConfig:
    """测试脚本配置。"""

    # =================【输入 / 输出配置】=================
    # model_path：待加载的模型权重路径。
    # data_root：测试数据集根目录。
    # save_csv：指标结果 CSV 输出路径。
    # num_files_to_test：本次抽样测试多少个 H5 文件。
    # ====================================================
    model_path: str = ""
    data_root: str = ""
    save_csv: str = "test_metrics.csv"
    num_files_to_test: int = 10
    num_basis: int = 15
    kernel_size: int = 7
    width: int = 32


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_srgb(x: torch.Tensor) -> torch.Tensor:
    safe_x = torch.clamp(x, min=0.0)
    return torch.where(
        safe_x <= 0.0031308,
        12.92 * safe_x,
        1.055 * torch.pow(safe_x + 1e-6, 1.0 / 2.4) - 0.055,
    )


def compute_metrics(img: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    img = np.clip(img, 0, 1)
    ref = np.clip(ref, 0, 1)
    return float(psnr_func(ref, img, data_range=1.0)), float(ssim_func(ref, img, data_range=1.0))


def load_checkpoint(model: torch.nn.Module, model_path: Path) -> None:
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def validate_config(config: TestConfig) -> tuple[Path, Path, Path]:
    model_path = Path(config.model_path)
    data_root = Path(config.data_root)
    save_csv = Path(config.save_csv)
    if not config.model_path:
        raise ValueError("请先在 TestConfig.model_path 中填写模型权重路径。")
    if not config.data_root:
        raise ValueError("请先在 TestConfig.data_root 中填写测试数据目录。")
    if not model_path.exists():
        raise FileNotFoundError(f"模型权重不存在：{model_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"测试数据目录不存在：{data_root}")
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    return model_path, data_root, save_csv


def test() -> None:
    config = TestConfig()
    model_path, data_root, save_csv = validate_config(config)

    model = NAF_BPN_FusionNet(num_basis=config.num_basis, ksz=config.kernel_size, width=config.width).to(DEVICE)
    load_checkpoint(model, model_path)
    model.eval()

    all_h5 = sorted(data_root.rglob("*.h5"))
    if not all_h5:
        raise FileNotFoundError(f"未在 {data_root} 中找到任何 .h5 文件。")

    selected_h5 = all_h5 if config.num_files_to_test <= 0 or config.num_files_to_test >= len(all_h5) else random.sample(all_h5, config.num_files_to_test)
    final_results: list[dict[str, float | str]] = []

    with torch.no_grad():
        for h5_path in selected_h5:
            with h5py.File(h5_path, "r", swmr=True) as h5_file:
                num_frames = int(h5_file["clean"].shape[0])
                metrics = {key: [] for key in ["r2p", "r2s", "r3p", "r3s", "rap", "ras", "s2p", "s2s", "s3p", "s3s", "sap", "sas"]}

                for frame_idx in tqdm(range(num_frames), desc="Frames", leave=False):
                    # 【输入标记】测试阶段固定读取 2dnr / 3dnr / noisy / clean。
                    img_2dnr_np = h5_file["2dnr"][frame_idx].astype(np.float32) / 4095.0
                    img_3dnr_np = h5_file["3dnr"][frame_idx].astype(np.float32) / 4095.0
                    gt_raw_np = h5_file["clean"][frame_idx].astype(np.float32) / 4095.0
                    noisy_t_np = h5_file["noisy"][frame_idx, 1, :, :].astype(np.float32) / 4095.0
                    noisy_tm1_np = (
                        h5_file["noisy"][frame_idx - 1, 1, :, :].astype(np.float32) / 4095.0
                        if frame_idx > 0
                        else noisy_t_np.copy()
                    )

                    img_2dnr = torch.from_numpy(img_2dnr_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    img_3dnr = torch.from_numpy(img_3dnr_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    noisy_t = torch.from_numpy(noisy_t_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    noisy_tm1 = torch.from_numpy(noisy_tm1_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

                    pred_raw, _ = model(img_2dnr, img_3dnr, noisy_t, noisy_tm1)
                    pred_raw_np = pred_raw.squeeze().cpu().numpy()

                    gt_srgb = to_srgb(torch.from_numpy(gt_raw_np)).numpy()
                    img_2dnr_srgb = to_srgb(torch.from_numpy(img_2dnr_np)).numpy()
                    img_3dnr_srgb = to_srgb(torch.from_numpy(img_3dnr_np)).numpy()
                    pred_srgb = to_srgb(pred_raw.cpu()).squeeze().numpy()

                    for key_name, img_np, ref_np, psnr_key, ssim_key in [
                        ("2D-RAW", img_2dnr_np, gt_raw_np, "r2p", "r2s"),
                        ("3D-RAW", img_3dnr_np, gt_raw_np, "r3p", "r3s"),
                        ("AI-RAW", pred_raw_np, gt_raw_np, "rap", "ras"),
                        ("2D-sRGB", img_2dnr_srgb, gt_srgb, "s2p", "s2s"),
                        ("3D-sRGB", img_3dnr_srgb, gt_srgb, "s3p", "s3s"),
                        ("AI-sRGB", pred_srgb, gt_srgb, "sap", "sas"),
                    ]:
                        psnr_value, ssim_value = compute_metrics(img_np, ref_np)
                        metrics[psnr_key].append(psnr_value)
                        metrics[ssim_key].append(ssim_value)

                final_results.append(
                    {
                        "File": h5_path.name,
                        "Scene": h5_path.parent.name,
                        "Raw_2D_PSNR": float(np.mean(metrics["r2p"])),
                        "Raw_2D_SSIM": float(np.mean(metrics["r2s"])),
                        "Raw_3D_PSNR": float(np.mean(metrics["r3p"])),
                        "Raw_3D_SSIM": float(np.mean(metrics["r3s"])),
                        "Raw_AI_PSNR": float(np.mean(metrics["rap"])),
                        "Raw_AI_SSIM": float(np.mean(metrics["ras"])),
                        "sRGB_2D_PSNR": float(np.mean(metrics["s2p"])),
                        "sRGB_2D_SSIM": float(np.mean(metrics["s2s"])),
                        "sRGB_3D_PSNR": float(np.mean(metrics["s3p"])),
                        "sRGB_3D_SSIM": float(np.mean(metrics["s3s"])),
                        "sRGB_AI_PSNR": float(np.mean(metrics["sap"])),
                        "sRGB_AI_SSIM": float(np.mean(metrics["sas"])),
                    }
                )

    result_df = pd.DataFrame(final_results)
    # 【输出标记】测试统计结果最终写入 save_csv。
    result_df.to_csv(save_csv, index=False)


if __name__ == "__main__":
    test()

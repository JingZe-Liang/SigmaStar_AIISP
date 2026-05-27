import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm


INPUT_DIR = Path(r"D:\DeepLearning\RAWProcessing\dataset\scene12_gt")
JSON_PATH = Path(r"D:\DeepLearning\RAWProcessing\metadata.json")
OUTPUT_VIDEO = Path(r"D:\DeepLearning\RAWProcessing\dataset\scene12_gt\scene12_gt.avi")

FPS = 24
EXPOSURE_COMPENSATION = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def natural_sort_key(text: str) -> list[int | str]:
    """按数字语义排序，避免 frame10 排在 frame2 前面。"""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"([0-9]+)", text)]


class TeamStandardISP:
    """将 RAW Bayer 数据转换为可视化 sRGB 帧。"""

    def __init__(self, json_path: Path) -> None:
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as file:
                self.meta = json.load(file)
        else:
            self.meta = {}

        self.black_level = 16.0
        self.white_level = 4095.0
        self.cv_bayer_code = cv2.COLOR_BayerGB2BGR

        self.g1_slice = (slice(0, None, 2), slice(0, None, 2))
        self.b_slice = (slice(0, None, 2), slice(1, None, 2))
        self.r_slice = (slice(1, None, 2), slice(0, None, 2))
        self.g2_slice = (slice(1, None, 2), slice(1, None, 2))
        self.wb_gains: list[float] | None = None

    def _compute_auto_white_balance(self, img: np.ndarray) -> list[float]:
        """采用 Gray World 假设估计白平衡增益。"""
        r_mean = float(np.mean(img[self.r_slice]))
        b_mean = float(np.mean(img[self.b_slice]))
        g_mean = float((np.mean(img[self.g1_slice]) + np.mean(img[self.g2_slice])) / 2.0)
        r_gain = g_mean / (r_mean + 1e-6)
        b_gain = g_mean / (b_mean + 1e-6)
        print(f"--- [AWB] R_gain={r_gain:.3f}, B_gain={b_gain:.3f} ---")
        return [r_gain, 1.0, b_gain]

    def process_frame(self, raw_data: np.ndarray, is_first_frame: bool = False) -> np.ndarray:
        """把单帧 RAW 数据映射成 8-bit BGR 可视化图像。"""
        img = (raw_data - self.black_level) / (self.white_level - self.black_level)
        img = np.clip(img, 0, 1)

        if is_first_frame or self.wb_gains is None:
            self.wb_gains = self._compute_auto_white_balance(img)

        img[self.r_slice] *= self.wb_gains[0]
        img[self.b_slice] *= self.wb_gains[2]
        img = np.clip(img * EXPOSURE_COMPENSATION, 0, 1)

        img_u16 = (img * 65535).astype(np.uint16)
        bgr = cv2.cvtColor(img_u16, self.cv_bayer_code)
        bgr_float = bgr.astype(np.float32) / 65535.0
        bgr_gamma = np.power(bgr_float, 1 / 2.2)
        return (np.clip(bgr_gamma, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    isp = TeamStandardISP(JSON_PATH)

    files = [file for file in INPUT_DIR.iterdir() if file.is_file() and file.suffix.lower() == ".safetensors"]
    files.sort(key=lambda path: natural_sort_key(path.name))

    if not files:
        print(f"[错误] 未在目录中找到 .safetensors 文件：{INPUT_DIR}")
        return

    sample_tensor = load_file(str(files[0]))["raw"]
    height, width = sample_tensor.shape

    print(f"--- 序列信息：帧数={len(files)}, 分辨率={width}x{height} (fp32 RAW) ---")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (width, height))

    for index, filename in enumerate(tqdm(files, desc="Safetensors ISP 渲染")):
        raw_tensor = load_file(str(filename))["raw"]
        raw_np = raw_tensor.to(torch.float32).cpu().numpy()

        frame_bgr = isp.process_frame(raw_np, is_first_frame=(index == 0))
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"\n[任务完成] 视频已生成：{OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()

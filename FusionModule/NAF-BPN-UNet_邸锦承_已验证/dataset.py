from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    """用于 2DNR / 3DNR 融合训练的数据集。"""

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int | None = 256,
        is_training: bool = True,
        motion_prob: float = 0.7,
        motion_trials: int = 3,
    ) -> None:
        # =================【输入配置】=================
        # root_dir：训练或验证时要读取的 H5 数据根目录。
        # patch_size：每次从整帧中裁剪多大的 patch 喂给模型。
        # is_training / motion_prob：控制是否启用随机裁剪和运动区域优先采样。
        # ============================================
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.is_training = is_training
        self.motion_prob = motion_prob
        self.motion_trials = motion_trials

        self.h5_files = sorted(self.root_dir.rglob("*.h5"))
        if not self.h5_files:
            raise FileNotFoundError(f"未在 {self.root_dir} 及其子目录中找到 .h5 文件。")

        self.samples: list[tuple[Path, int]] = []
        for h5_path in self.h5_files:
            with h5py.File(h5_path, "r") as h5_file:
                num_frames = int(h5_file["clean"].shape[0])
            self.samples.extend((h5_path, frame_idx) for frame_idx in range(num_frames))

        print(f"共找到 {len(self.h5_files)} 个 H5 文件，累计 {len(self.samples)} 帧样本。")

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_crop_origin(self, height: int, width: int, diff_map: np.ndarray | None) -> tuple[int, int]:
        """返回裁剪左上角坐标。"""
        if self.patch_size is None:
            return 0, 0

        patch_height = self.patch_size
        patch_width = self.patch_size

        if height < patch_height or width < patch_width:
            raise ValueError(f"patch_size={self.patch_size} 大于图像尺寸 {width}x{height}，无法裁剪。")

        max_top = height - patch_height
        max_left = width - patch_width

        if not self.is_training:
            return max_top // 2, max_left // 2

        def sample_random_origin() -> tuple[int, int]:
            top = int(np.random.randint(0, max_top + 1))
            left = int(np.random.randint(0, max_left + 1))
            return top, left

        if diff_map is None or np.random.rand() >= self.motion_prob:
            return sample_random_origin()

        best_top, best_left = 0, 0
        best_energy = -1.0

        for _ in range(self.motion_trials):
            top, left = sample_random_origin()
            patch = diff_map[top : top + patch_height, left : left + patch_width]
            motion_energy = float(np.mean(patch))

            if motion_energy > best_energy:
                best_energy = motion_energy
                best_top, best_left = top, left

            if best_energy > 0.05:
                break

        return best_top, best_left

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h5_path, frame_idx = self.samples[idx]

        with h5py.File(h5_path, "r") as h5_file:
            # 【输入标记】固定从 H5 中读取 2dnr、3dnr、noisy 和 clean。
            img_2dnr_np = h5_file["2dnr"][frame_idx].astype(np.float32) / 4095.0
            img_3dnr_np = h5_file["3dnr"][frame_idx].astype(np.float32) / 4095.0
            img_clean_np = h5_file["clean"][frame_idx].astype(np.float32) / 4095.0
            noisy_t_np = h5_file["noisy"][frame_idx, 1, :, :].astype(np.float32) / 4095.0

            if frame_idx > 0:
                noisy_tm1_np = h5_file["noisy"][frame_idx - 1, 1, :, :].astype(np.float32) / 4095.0
            else:
                noisy_tm1_np = noisy_t_np.copy()

        if self.patch_size is not None:
            diff_map = np.abs(noisy_t_np - noisy_tm1_np)
            height, width = img_2dnr_np.shape
            top, left = self._choose_crop_origin(height, width, diff_map)
            bottom = top + self.patch_size
            right = left + self.patch_size

            img_2dnr_np = img_2dnr_np[top:bottom, left:right]
            img_3dnr_np = img_3dnr_np[top:bottom, left:right]
            img_clean_np = img_clean_np[top:bottom, left:right]
            noisy_t_np = noisy_t_np[top:bottom, left:right]
            noisy_tm1_np = noisy_tm1_np[top:bottom, left:right]

        img_2dnr = torch.from_numpy(img_2dnr_np).unsqueeze(0)
        img_3dnr = torch.from_numpy(img_3dnr_np).unsqueeze(0)
        noisy_t = torch.from_numpy(noisy_t_np).unsqueeze(0)
        noisy_tm1 = torch.from_numpy(noisy_tm1_np).unsqueeze(0)
        img_clean = torch.from_numpy(img_clean_np).unsqueeze(0)
        return img_2dnr, img_3dnr, noisy_t, noisy_tm1, img_clean

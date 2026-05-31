import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import random


class CRVDDataset(Dataset):
    """
    CRVD数据集加载器 说明 - 支持单帧和序列两种模式的raw图像降噪训练

    功能：
        从分散的文件夹中加载noisy-GT配对数据，支持两种加载模式：
        1. 单帧模式：每帧作为独立样本，随机选择noisy版本
        2. 序列模式：连续帧作为完整序列，保持时序一致性

    参数：
        noisy_root (str): noisy图像根目录，如 "D:/data/indoor_raw_noisy"
        gt_root (str): GT图像根目录，如 "D:/data/indoor_raw_gt"
        scenes (list[int]): 场景列表，训练用[1-6]，测试用[7-11]
        iso_levels (list[int]): ISO级别列表，可选[1600, 3200, 6400, 12800, 25600]
        num_frames (int): 每个scene的帧数，固定7帧（默认值）
        num_noisy_versions (int): 每帧的noisy版本数，固定10个（默认值）
        sequence_mode (bool): 是否启用序列模式（默认True）
            - True: 返回完整帧序列 (seq_len, H, W)
            - False: 返回单帧 (H, W)，与原始版本行为一致

    返回：
        - 单帧模式: tuple[np.ndarray, np.ndarray] = (noisy, gt)
            * noisy: (H, W) float32，随机从noisy0-9中选一个
            * gt: (H, W) float32
        - 序列模式: tuple[np.ndarray, np.ndarray] = (noisy_seq, gt_seq)
            * noisy_seq: (num_frames, H, W) float32，整个序列使用同一个noisy版本
            * gt_seq: (num_frames, H, W) float32

    样本总数计算：
        单帧模式：len(dataset) = len(scenes) × len(iso_levels) × num_frames
        序列模式：len(dataset) = len(scenes) × len(iso_levels)
        例如 scenes=[1,2], iso=[1600,3200]:
            - 单帧: 2×2×7 = 28个样本
            - 序列: 2×2 = 4个样本（每个样本包含7帧）
    没有进行归一化 ， 至于黑电评矫正貌似数据集里面已经做了 因为很多raw数据值都低于论文里所声称的240

    索引顺序：
        单帧模式：先遍历scene → 再遍历ISO → 最后遍历frame
        序列模式：先遍历scene → 再遍历ISO（每个组合为一个完整序列）
        训练时使用 DataLoader(shuffle=True) 打乱顺序
        序列模式下shuffle打乱的是完整序列，不会破坏帧间时序关系

    使用示例：
        # 单帧模式（兼容原始版本）
        >>> dataset = CRVDDataset(
        ...     noisy_root="F：/CRVD_dataset/indoor_raw_noisy",
        ...     gt_root="F:/CRVD_dataset/indoor_raw_gt",
        ...     scenes=[1, 2, 3, 4, 5, 6],
        ...     iso_levels=[1600, 3200, 6400],
        ...     sequence_mode=False  # 单帧模式
        ... )
        >>> noisy, gt = dataset[0]  # 返回单帧: (H, W), (H, W)

        # 序列模式（时序依赖处理）
        >>> dataset_seq = CRVDDataset(
        ...     noisy_root="D:/data/indoor_raw_noisy",
        ...     gt_root="D:/data/indoor_raw_gt",
        ...     scenes=[1, 2, 3, 4, 5, 6],
        ...     iso_levels=[1600, 3200, 6400],
        ...     sequence_mode=True  # 序列模式
        ... )
        >>> noisy_seq, gt_seq = dataset_seq[0]  # 返回序列: (7, H, W), (7, H, W)

    注意事项：
        - 单帧模式：每次调用__getitem__会随机选择noisy0-9中的一个版本
        - 序列模式：整个序列使用同一个随机选择的noisy版本（保持时序一致性）
        - 图像自动转为float32格式
        - 文件路径错误会抛出FileNotFoundError
        - 序列模式下，DataLoader的batch_size处理的是完整序列数，不是帧数

    与网络输入的关系：
        单帧模式：
            1. 当前返回: (H, W) numpy array
            2. DataLoader自动batch后: (B, H, W) torch tensor
            3. 网络通常需要: (B, C, H, W)
               - 方案: batch_noisy.unsqueeze(1) 变成 (B, 1, H, W)

        序列模式：
            1. 当前返回: (num_frames, H, W) numpy array
            2. DataLoader自动batch后: (B, num_frames, H, W) torch tensor
            3. 网络通常需要: (B, C, num_frames, H, W) 或 (B, num_frames, C, H, W)
               - 方案: 根据网络架构调整维度，常见：
                 batch_noisy.unsqueeze(2) → (B, num_frames, 1, H, W)
                 或 batch_noisy.permute(0, 2, 1, 3) → (B, 1, num_frames, H, W)

    典型训练流程（序列模式）：
        # 1. 创建数据集
        train_dataset = CRVDDataset(
            noisy_root="...", gt_root="...",
            scenes=[1, 2, 3, 4, 5, 6],
            iso_levels=[1600, 3200, 6400, 12800, 25600],
            sequence_mode=True
        )

        # 2. 创建DataLoader（shuffle=True安全，打乱的是完整序列）
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,      # 同时处理4个序列
            shuffle=True,      # 打乱序列顺序，不破坏帧间依赖
            num_workers=4
        )

        # 3. 训练循环
        for epoch in range(num_epochs):
            for noisy_seq_batch, gt_seq_batch in train_loader:
                # noisy_seq_batch: [4, 7, H, W], gt_seq_batch: [4, 7, H, W]
                noisy_seq_batch = noisy_seq_batch.unsqueeze(2)  # [4, 7, 1, H, W]
                gt_seq_batch = gt_seq_batch.unsqueeze(2)        # [4, 7, 1, H, W]

                # 逐帧处理，保持时序依赖
                prev_outputs = None
                for frame_idx in range(7):
                    noisy_frame = noisy_seq_batch[:, frame_idx]  # [4, 1, H, W]
                    gt_frame = gt_seq_batch[:, frame_idx]        # [4, 1, H, W]

                    if frame_idx == 0:
                        outputs = model.first_frame(noisy_frame)
                    else:
                        outputs = model(noisy_frame, prev_outputs)

                    prev_outputs = outputs
                    loss = criterion(outputs, gt_frame)
                    # 反向传播...

    关键提醒：
        - 序列模式需要网络支持时序依赖处理（如RNN、3D CNN、Transformer等）
        - 验证集建议shuffle=False保持可复现性
        - num_workers>0可多进程加速，但Windows下可能有问题
        - 序列模式下，每个epoch看到不同的序列组合，有助于模型泛化
        - 序列内部noisy版本一致，模拟真实视频的时序噪声特性
    """

    def __init__(
            self,
            noisy_root: str,
            gt_root: str,
            scenes: list[int],
            iso_levels: list[int],
            num_frames: int = 7,
            num_noisy_versions: int = 10,
            sequence_mode: bool = True,  # 新增参数：是否使用序列模式
    ) -> None:
        self.noisy_root = Path(noisy_root)
        self.gt_root = Path(gt_root)
        self.scenes = scenes
        self.iso_levels = iso_levels
        self.num_frames = num_frames
        self.num_noisy_versions = num_noisy_versions
        self.sequence_mode = sequence_mode  # 是否返回序列

        self.samples = self._build_sample_list()

    def _build_sample_list(self) -> list[dict]:
        samples = []
        for scene in self.scenes:
            for iso in self.iso_levels:
                if self.sequence_mode:
                    # 序列模式：每个样本是一个完整的帧序列
                    # 对于7帧序列，只有一个样本（包含frame1-7）
                    samples.append({
                        "scene": scene,
                        "iso": iso,
                        "frame": 1,  # 起始帧
                        "sequence": True
                    })
                else:
                    # 原始模式：每个样本是单帧
                    for frame_idx in range(1, self.num_frames + 1):
                        samples.append({
                            "scene": scene,
                            "iso": iso,
                            "frame": frame_idx,
                            "sequence": False
                        })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        scene = sample["scene"]
        iso = sample["iso"]
        start_frame = sample["frame"]  # 起始帧号
        is_sequence = sample["sequence"]

        if not is_sequence:
            # 原始模式：返回单帧（保持完全兼容）
            noisy_version = random.randint(0, self.num_noisy_versions - 1)

            gt_path = self.gt_root / f"scene{scene}" / f"ISO{iso}" / f"frame{start_frame}_clean_and_slightly_denoised.tiff"
            noisy_path = (
                    self.noisy_root / f"scene{scene}" / f"ISO{iso}" / f"frame{start_frame}_noisy{noisy_version}.tiff"
            )

            gt = cv2.imread(str(gt_path), -1)
            noisy = cv2.imread(str(noisy_path), -1)

            if gt is None:
                raise FileNotFoundError(f"GT not found: {gt_path}")
            if noisy is None:
                raise FileNotFoundError(f"Noisy not found: {noisy_path}")

            gt = gt.astype(np.float32)
            noisy = noisy.astype(np.float32)

            return noisy, gt
        else:
            # 序列模式：返回连续帧序列
            # 为整个序列选择一个固定的noisy版本（保持一致）
            noisy_version = random.randint(0, self.num_noisy_versions - 1)

            noisy_sequence = []
            gt_sequence = []

            for i in range(self.num_frames):
                frame_idx = start_frame + i
                gt_path = self.gt_root / f"scene{scene}" / f"ISO{iso}" / f"frame{frame_idx}_clean_and_slightly_denoised.tiff"
                noisy_path = (
                        self.noisy_root / f"scene{scene}" / f"ISO{iso}" / f"frame{frame_idx}_noisy{noisy_version}.tiff"
                )

                gt = cv2.imread(str(gt_path), -1)
                noisy = cv2.imread(str(noisy_path), -1)

                if gt is None:
                    raise FileNotFoundError(f"GT not found: {gt_path}")
                if noisy is None:
                    raise FileNotFoundError(f"Noisy not found: {noisy_path}")

                gt_sequence.append(gt.astype(np.float32))
                noisy_sequence.append(noisy.astype(np.float32))

            # 返回形状: (7, H, W) 的序列
            return np.stack(noisy_sequence, axis=0), np.stack(gt_sequence, axis=0)


def main() -> None:
    noisy_root = "E:/CRVD_datasetindoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"

    # 测试1：原始模式（单帧）
    print("=== 原始模式（单帧）===")
    dataset_single = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1, 2],
        iso_levels=[1600, 3200],
        num_frames=7,
        sequence_mode=False  # 单帧模式
    )

    print(f"Total samples (single): {len(dataset_single)}")  # 应该输出: 2×2×7 = 28
    noisy, gt = dataset_single[0]
    print(f"Noisy shape (single): {noisy.shape}")  # (H, W)
    print(f"GT shape (single): {gt.shape}")  # (H, W)

    # 测试2：序列模式
    print("\n=== 序列模式（帧序列）===")
    dataset_seq = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1, 2],
        iso_levels=[1600, 3200],
        num_frames=7,
        sequence_mode=True  # 序列模式
    )

    print(f"Total samples (sequence): {len(dataset_seq)}")  # 应该输出: 2×2 = 4（每个场景-ISO组合为一个序列）
    noisy_seq, gt_seq = dataset_seq[0]
    print(f"Noisy shape (sequence): {noisy_seq.shape}")  # (7, H, W)
    print(f"GT shape (sequence): {gt_seq.shape}")  # (7, H, W)
    print(f"Frame 1 range: [{noisy_seq[0].min():.2f}, {noisy_seq[0].max():.2f}]")


if __name__ == "__main__":
    main()
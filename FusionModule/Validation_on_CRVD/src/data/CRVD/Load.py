import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import random


class CRVDDataset(Dataset):
    """
    CRVD数据集加载器 说明 - 用于raw图像降噪训练

    功能：
        从分散的文件夹中加载noisy-GT配对数据，自动索引所有样本

    参数：
        noisy_root (str): noisy图像根目录，如 "D:/data/indoor_raw_noisy"---我简单微调了数据结构，师兄 ，我删掉了一级文件夹
        gt_root (str): GT图像根目录，如 "D:/data/indoor_raw_gt"
        scenes (list[int]): 场景列表，训练用[1-6]，测试用[7-11]
        iso_levels (list[int]): ISO级别列表，可选[1600, 3200, 6400, 12800, 25600]
        num_frames (int): 每个scene的帧数，固定7帧（默认值）
        num_noisy_versions (int): 每帧的noisy版本数，固定10个（默认值）

    返回：
        tuple[np.ndarray, np.ndarray]: (noisy, gt)配对
            - noisy: (H, W) float32，随机从noisy0-9中选一个
            - gt: (H, W) float32

    样本总数：
        len(dataset) = len(scenes) × len(iso_levels) × num_frames
        例如：scenes=[1,2], iso=[1600,3200] → 2×2×7 = 28个样本

    索引顺序：
        先遍历scene → 再遍历ISO → 最后遍历frame
        例如：scene1/ISO1600/frame1-7 → scene1/ISO3200/frame1-7 → scene2/...
        训练时使用 DataLoader(shuffle=True) 打乱顺序

    使用示例：
        >>> dataset = CRVDDataset(
        ...     noisy_root="F：/CRVD_dataset/indoor_raw_noisy",
        ...     gt_root="F:/CRVD_dataset/indoor_raw_gt",
        ...     scenes=[1, 2, 3, 4, 5, 6],
        ...     iso_levels=[1600, 3200, 6400]
        ... )
        >>> noisy, gt = dataset[0]
        >>> loader = DataLoader(dataset, batch_size=4, shuffle=True)

    注意事项：
        - 每次调用__getitem__会随机选择noisy0-9中的一个版本
        - 图像自动转为float32格式
        - 文件路径错误会抛出FileNotFoundError
    """
    """
    补充说明 

    数据格式：
        - 返回的图像是raw sensor数据，Bayer格式（GBRG排列）
        - 像素值范围：[240, 4095]（黑电平240，白电平2^12-1）
        - 尺寸：(H, W) 单通道，需根据网络输入格式进行pack或处理

    与网络输入的关系：
        1. 当前返回：(H, W) numpy array
        2. DataLoader自动batch后：(B, H, W) torch tensor
        3. 网络通常需要：(B, C, H, W)
           - 方案1：在Dataset的__getitem__中添加 np.expand_dims(img, 0) 变成(1, H, W)
           - 方案2：在网络forward前 unsqueeze(1)
           - 方案3：pack成RGBG四通道(4, H/2, W/2)

    典型训练流程：
        # 1. 创建训练集和验证集
        train_dataset = CRVDDataset(
            noisy_root="...", gt_root="...",
            scenes=[1, 2, 3, 4, 5, 6],
            iso_levels=[1600, 3200, 6400, 12800, 25600]
        )
        val_dataset = CRVDDataset(
            noisy_root="...", gt_root="...",
            scenes=[7, 8],
            iso_levels=[1600, 3200, 6400, 12800, 25600]
        )

        # 2. 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

        # 3. 训练循环
        for epoch in range(num_epochs):
            for noisy, gt in train_loader:
                noisy, gt = noisy.cuda(), gt.cuda()
                # 根据网络需求调整维度
                noisy = noisy.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                gt = gt.unsqueeze(1)

                output = model(noisy)
                loss = criterion(output, gt)
                # 反向传播...

    关键提醒：
        - 每次epoch数据顺序不同（因为随机选noisy版本+shuffle）
        - 验证集建议shuffle=False保持可复现性
        - num_workers>0可多进程加速，但Windows下可能有问题
    """


    def __init__(
        self,
        noisy_root: str,
        gt_root: str,
        scenes: list[int],
        iso_levels: list[int],
        num_frames: int = 7,
        num_noisy_versions: int = 10,
    ) -> None:
        self.noisy_root = Path(noisy_root)
        self.gt_root = Path(gt_root)
        self.scenes = scenes
        self.iso_levels = iso_levels
        self.num_frames = num_frames
        self.num_noisy_versions = num_noisy_versions

        self.samples = self._build_sample_list()

    def _build_sample_list(self) -> list[dict]:
        samples = []
        for scene in self.scenes:
            for iso in self.iso_levels:
                for frame_idx in range(1, self.num_frames + 1):
                    samples.append({"scene": scene, "iso": iso, "frame": frame_idx})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[idx]
        scene = sample["scene"]
        iso = sample["iso"]
        frame_idx = sample["frame"]

        noisy_version = random.randint(0, self.num_noisy_versions - 1)

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

        gt = gt.astype(np.float32)
        noisy = noisy.astype(np.float32)

        return noisy, gt


def main() -> None:
    noisy_root = "path/to/indoor_raw_noisy"
    gt_root = "path/to/indoor_raw_gt"

    dataset = CRVDDataset(
        noisy_root=noisy_root, gt_root=gt_root, scenes=[1, 2], iso_levels=[1600, 3200], num_frames=7
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Scenes: {dataset.scenes}")
    print(f"ISO levels: {dataset.iso_levels}")

    noisy, gt = dataset[0]
    print(f"\nFirst sample:")
    print(f"Noisy shape: {noisy.shape}, dtype: {noisy.dtype}")
    print(f"GT shape: {gt.shape}, dtype: {gt.dtype}")
    print(f"Noisy range: [{noisy.min():.2f}, {noisy.max():.2f}]")
    print(f"GT range: [{gt.min():.2f}, {gt.max():.2f}]")


if __name__ == "__main__":
    main()
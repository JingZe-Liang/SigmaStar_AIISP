# ==========================================================
# 第一部分：环境修复补丁 (针对 Python 版本与库兼容性)
# ==========================================================
import sys
import builtins
import typing


def apply_patches():
    # 1. 修复 Python 3.10+ 中 Torch/OpenCV 的 Callable 校验错误
    if sys.version_info >= (3, 10):
        if not hasattr(typing, 'ParamSpec'):
            typing.ParamSpec = typing.TypeVar
        orig_getitem = typing.Callable.__getitem__

        def safe_getitem(self, params):
            if not isinstance(params, tuple) or len(params) != 2:
                return orig_getitem(self, params)
            args, res = params
            if hasattr(args, '__constraint__') or str(type(args)) == "<class 'typing.TypeVar'>":
                params = ([args], res)
            return orig_getitem(self, params)

        typing.Callable.__getitem__ = safe_getitem

    # 2. 修复 Python < 3.9 的 list[int] 报错
    if sys.version_info < (3, 9):
        class MetaList(type):
            def __getitem__(self, item): return builtins.list

        class NewList(builtins.list, metaclass=MetaList): pass

        builtins.list = NewList


apply_patches()

# ==========================================================
# 第二部分：导入库
# ==========================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 处理项目导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset


# ==========================================================
# 第三部分：核心工具函数 (Bayer 拆分与合并)
# ==========================================================

def split_bayer_channels(raw):
    """将单通道 Bayer (H, W) 拆分为 4 个子通道 (H/2, W/2)"""
    ch_r = raw[0::2, 0::2]  # Red
    ch_gr = raw[0::2, 1::2]  # Green on Red row
    ch_gb = raw[1::2, 0::2]  # Green on Blue row
    ch_b = raw[1::2, 1::2]  # Blue
    return [ch_r, ch_gr, ch_gb, ch_b]


def merge_bayer_masks(masks):
    """将 4 个子通道的掩膜重新拼回原始分辨率 (H, W)"""
    h_half, w_half = masks[0].shape
    full_mask = np.zeros((h_half * 2, w_half * 2), dtype=np.uint8)
    full_mask[0::2, 0::2] = masks[0]
    full_mask[0::2, 1::2] = masks[1]
    full_mask[1::2, 0::2] = masks[2]
    full_mask[1::2, 1::2] = masks[3]
    return full_mask


# ==========================================================
# 第四部分：三种不同的运动检测方法
# ==========================================================

# 方法 1: 分通道检测 (源自: 不处理分通道的噪声.py)
def detect_motion_split_channels(gt_seq, var_threshold=20):
    num_frames, H, W = gt_seq.shape
    norm_seq = (gt_seq / (gt_seq.max() + 1e-6) * 255).astype(np.uint8)
    subtractors = [
        cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=var_threshold, detectShadows=False) for _ in
        range(4)]

    motion_masks = []
    for f in range(num_frames):
        channels = split_bayer_channels(norm_seq[f])
        channel_masks = [(subtractors[i].apply(channels[i]) > 0).astype(np.uint8) for i in range(4)]
        full_res_mask = merge_bayer_masks(channel_masks)
        full_res_mask = cv2.medianBlur(full_res_mask, 3)
        motion_masks.append(full_res_mask)
    return np.stack(motion_masks, axis=0)


# 方法 2: 不分通道直接检测 (源自: 不分通道的.py)
def detect_motion_direct(seq_data, varThreshold=30):
    num_frames, H, W = seq_data.shape
    max_val = np.max(seq_data) if np.max(seq_data) > 0 else 1.0
    processed_seq = (seq_data / max_val * 255).astype(np.uint8)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=varThreshold, detectShadows=False)

    motion_masks = []
    for f in range(num_frames):
        mask = fgbg.apply(processed_seq[f])
        motion_masks.append((mask > 0).astype(np.uint8))
    return np.stack(motion_masks, axis=0)


# 方法 3: 4通道打包检测 (源自: MD.py)
def detect_motion_packed_4channel(seq_data, varThreshold=25):
    num_frames, H, W = seq_data.shape
    max_val = np.max(seq_data) if np.max(seq_data) > 0 else 1.0
    norm_seq = (seq_data / max_val * 255).astype(np.uint8)
    subtractors = [
        cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=varThreshold, detectShadows=False) for _ in
        range(4)]

    motion_masks = []
    for f in range(num_frames):
        # 使用 MD.py 中的 pack_raw 逻辑
        packed_frame = np.zeros((H // 2, W // 2, 4), dtype=np.uint8)
        chs = split_bayer_channels(norm_seq[f])
        for i in range(4): packed_frame[:, :, i] = chs[i]

        packed_mask = np.zeros_like(packed_frame, dtype=np.uint8)
        for i in range(4):
            mask = subtractors[i].apply(packed_frame[:, :, i])
            packed_mask[:, :, i] = (mask > 0).astype(np.uint8)

        full_mask = merge_bayer_masks([packed_mask[:, :, i] for i in range(4)])
        full_mask = cv2.medianBlur(full_mask, 3)
        motion_masks.append(full_mask)
    return np.stack(motion_masks, axis=0)


# ==========================================================
# 第五部分：可视化执行
# ==========================================================
if __name__ == "__main__":
    dataset = CRVDDataset(
        noisy_root=r"D:\A_Data\CRVD_dataset\indoor_raw_noisy",
        gt_root=r"D:\A_Data\CRVD_dataset\indoor_raw_gt",
        scenes=[1], iso_levels=[1600], sequence_mode=True
    )

    noisy, _ = dataset[0]

    # --- 你可以在这里切换不同的方法进行对比 ---
    print("正在运行分通道检测...")
    results_split = detect_motion_split_channels(noisy)

    print("正在运行直接检测...")
    results_direct = detect_motion_direct(noisy)

    # 可视化对比
    methods = [("Split Channels", results_split), ("Direct (Non-split)", results_direct)]

    for title, motion_labels in methods:
        fig, axes = plt.subplots(2, 7, figsize=(20, 7))
        plt.suptitle(f"Method: {title}", fontsize=15)
        for i in range(7):
            disp_img = noisy[i] / (np.max(noisy[i]) + 1e-6)
            axes[0, i].imshow(np.power(disp_img, 0.45), cmap='gray')
            axes[0, i].set_title(f"GT {i}")
            axes[0, i].axis('off')

            axes[1, i].imshow(motion_labels[i], cmap='hot')
            axes[1, i].set_title(f"Motion {i}")
            axes[1, i].axis('off')
        plt.tight_layout()

    plt.show()
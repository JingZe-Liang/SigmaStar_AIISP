from pathlib import Path

import numpy as np
import rawpy
import torch
from safetensors.torch import save_file
from tqdm import tqdm


INPUT_DIR = Path(r"D:\DeepLearning\RAWProcessing\A001_03161251_C007_Bayer_NPY")
OUTPUT_DIR = Path(r"D:\DeepLearning\RAWProcessing\Standard_1080p_12bit_GBRG_safetensors")
TARGET_HEIGHT = 1080
TARGET_WIDTH = 1920

DEBUG_PRINT = True
PATCH_Y = 100
PATCH_X = 100
PATCH_SIZE = 6


def choose_aligned_start(full_size: int, target_size: int, expected_parity: int) -> int:
    """在合法范围内选一个尽量居中的裁剪起点，并满足 CFA 对齐要求。"""
    if target_size > full_size:
        raise ValueError(f"目标尺寸 {target_size} 大于原始尺寸 {full_size}，无法裁剪。")

    max_start = full_size - target_size
    center_start = max_start // 2

    candidates = [start for start in range(max_start + 1) if start % 2 == expected_parity]
    if not candidates:
        raise ValueError("未找到满足 CFA 对齐要求的合法裁剪起点。")

    return min(candidates, key=lambda start: abs(start - center_start))


def crop_aligned_to_gbrg(data: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, int, int]:
    """中心裁剪并对齐到 GBRG 排布。

    原始数据默认按 RGGB 理解，这里通过控制裁剪起点奇偶性，把输出对齐到团队约定的 GBRG。
    """
    start_h = choose_aligned_start(data.shape[0], target_h, expected_parity=1)
    start_w = choose_aligned_start(data.shape[1], target_w, expected_parity=0)
    cropped = data[start_h : start_h + target_h, start_w : start_w + target_w]
    return cropped, start_h, start_w


def load_raw_file(file_path: Path) -> np.ndarray:
    if file_path.suffix.lower() == ".npy":
        return np.load(file_path).astype(np.float32)
    if file_path.suffix.lower() == ".dng":
        with rawpy.imread(str(file_path)) as raw:
            return raw.raw_image.copy().astype(np.float32)
    raise ValueError(f"不支持的文件类型：{file_path.suffix}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [file for file in INPUT_DIR.iterdir() if file.is_file() and file.suffix.lower() in {".npy", ".dng"}]
    files.sort(key=lambda path: path.name)

    print(f"找到 {len(files)} 个文件，开始执行 RGGB 16-bit -> GBRG 12-bit Safetensors 转换。")

    for index, file_path in enumerate(tqdm(files)):
        try:
            raw_data = load_raw_file(file_path)

            if DEBUG_PRINT and index == 0:
                raw_patch = raw_data[PATCH_Y : PATCH_Y + PATCH_SIZE, PATCH_X : PATCH_X + PATCH_SIZE]

            # 有些 .npy / .dng 仍保留 16-bit 范围，这里统一压到 12-bit。
            if float(np.max(raw_data)) > 4095.0:
                raw_data = raw_data / 16.0

            cropped, start_h, start_w = crop_aligned_to_gbrg(raw_data, TARGET_HEIGHT, TARGET_WIDTH)
            cropped = np.clip(cropped, 0, 4095)

            if DEBUG_PRINT and index == 0:
                patch_y = PATCH_Y - start_h
                patch_x = PATCH_X - start_w
                cropped_patch = cropped[patch_y : patch_y + PATCH_SIZE, patch_x : patch_x + PATCH_SIZE]

            tensor_fp32 = torch.from_numpy(cropped).to(torch.float32)

            if DEBUG_PRINT and index == 0:
                tensor_patch = tensor_fp32[patch_y : patch_y + PATCH_SIZE, patch_x : patch_x + PATCH_SIZE].numpy()
                print("\n================ DEBUG PATCH ================")
                print("\n原始 16-bit Patch:")
                print(raw_patch.astype(np.int32))
                print("\n转换后的 12-bit Patch:")
                print(cropped_patch)
                print("\nSafetensors 张量 Patch:")
                print(tensor_patch)
                print("\n原始数据 / 16 的结果:")
                print(raw_patch / 16.0)

                diff = np.abs((raw_patch / 16.0) - cropped_patch)
                print("\n误差统计:")
                print("max error:", float(diff.max()))
                print("mean error:", float(diff.mean()))
                print("=============================================\n")

            save_path = OUTPUT_DIR / f"frame{index}.safetensors"
            save_file({"raw": tensor_fp32}, str(save_path))
        except Exception as exc:
            print(f"处理失败 {file_path.name}: {exc}")

    print("\n[任务完成]")
    print("当前输出规范：12-bit | GBRG | 1080p | fp32 | Safetensors")


if __name__ == "__main__":
    main()

import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


INPUT_DIR = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\CompanyData\scene1")
JSON_PATH = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\CompanyData\scene1\metadata.json")
OUTPUT_VIDEO = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\Reports\scene1_3dnr.avi")

TEMP_IN_RAW = Path("temp_dataset_in.raw")
TEMP_OUT_RAW = Path("temp_ffmpeg_out.raw")

FPS = 15
EXPOSURE_COMPENSATION = 50
HQDN3D_FILTER = "hqdn3d=0.001:0.001:8:3"


def natural_sort_key(text: str) -> list[int | str]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"([0-9]+)", text)]


def pack_bayer_to_quadrants(raw_img: np.ndarray) -> np.ndarray:
    """把 Bayer 图像拆成四象限，方便后续把 3DNR 作用到单通道上。"""
    height, width = raw_img.shape
    half_height, half_width = height // 2, width // 2
    packed = np.zeros_like(raw_img)
    packed[0:half_height, 0:half_width] = raw_img[0::2, 0::2]
    packed[0:half_height, half_width:width] = raw_img[0::2, 1::2]
    packed[half_height:height, 0:half_width] = raw_img[1::2, 0::2]
    packed[half_height:height, half_width:width] = raw_img[1::2, 1::2]
    return packed


def unpack_quadrants_to_bayer(packed_img: np.ndarray) -> np.ndarray:
    height, width = packed_img.shape
    half_height, half_width = height // 2, width // 2
    raw_img = np.zeros_like(packed_img)
    raw_img[0::2, 0::2] = packed_img[0:half_height, 0:half_width]
    raw_img[0::2, 1::2] = packed_img[0:half_height, half_width:width]
    raw_img[1::2, 0::2] = packed_img[half_height:height, 0:half_width]
    raw_img[1::2, 1::2] = packed_img[half_height:height, half_width:width]
    return raw_img


class TeamStandardISP:
    """把 RAW 数据渲染成可视化 BGR 帧。"""

    def __init__(self, json_path: Path) -> None:
        self.black_level = 256.0
        self.white_level = 4095.0
        self.cv_bayer_code = cv2.COLOR_BayerGB2BGR
        self.g1_slice = (slice(0, None, 2), slice(0, None, 2))
        self.b_slice = (slice(0, None, 2), slice(1, None, 2))
        self.r_slice = (slice(1, None, 2), slice(0, None, 2))
        self.g2_slice = (slice(1, None, 2), slice(1, None, 2))
        self.wb_gains: list[float] | None = None

    def _compute_auto_white_balance(self, img: np.ndarray) -> list[float]:
        r_mean = float(np.mean(img[self.r_slice]))
        b_mean = float(np.mean(img[self.b_slice]))
        g_mean = float((np.mean(img[self.g1_slice]) + np.mean(img[self.g2_slice])) / 2.0)
        return [g_mean / (r_mean + 1e-6), 1.0, g_mean / (b_mean + 1e-6)]

    def process_frame(self, raw_data: np.ndarray, is_first_frame: bool = False) -> np.ndarray:
        img = np.clip((raw_data - self.black_level) / (self.white_level - self.black_level), 0, 1)
        if is_first_frame or self.wb_gains is None:
            self.wb_gains = self._compute_auto_white_balance(img)
        img[self.r_slice] *= self.wb_gains[0]
        img[self.b_slice] *= self.wb_gains[2]
        img = np.clip(img * EXPOSURE_COMPENSATION, 0, 1)
        bgr = cv2.cvtColor((img * 65535).astype(np.uint16), self.cv_bayer_code)
        return (np.clip(np.power(bgr.astype(np.float32) / 65535.0, 1 / 2.2), 0, 1) * 255).astype(np.uint8)


def get_h5_video_info(path: Path, key: str = "2dnr") -> tuple[int, int, int]:
    with h5py.File(path, "r") as h5_file:
        shape = h5_file[key].shape
    real_dims = [size for size in shape if size > 1]
    if len(real_dims) == 2:
        return 1, real_dims[0], real_dims[1]
    if len(real_dims) >= 3:
        return real_dims[0], real_dims[-2], real_dims[-1]
    raise ValueError(f"无法识别的数据形状：{shape}")


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    files = [file for file in INPUT_DIR.iterdir() if file.is_file() and file.suffix.lower() in {".h5", ".hdf5"}]
    files.sort(key=lambda path: natural_sort_key(path.name))

    total_frames = 0
    height = 0
    width = 0
    for index, filename in enumerate(files):
        num_frames, cur_height, cur_width = get_h5_video_info(filename)
        total_frames += num_frames
        if index == 0:
            height, width = cur_height, cur_width

    print(f"\n--- 序列信息：共 {len(files)} 个文件，总计 {total_frames} 帧，分辨率 {width}x{height} ---")
    print("\n>>> [1/3] 正在拆解 Bayer 阵列并导出到临时 RAW 文件...")

    with open(TEMP_IN_RAW, "wb") as raw_file:
        for filename in tqdm(files, desc="导出 RAW 序列"):
            with h5py.File(filename, "r") as h5_file:
                video_data = h5_file["2dnr"][:]

            video_data = np.squeeze(video_data)
            if video_data.ndim == 2:
                video_data = np.expand_dims(video_data, axis=0)
            while video_data.ndim > 3:
                video_data = video_data[..., 0] if video_data.shape[-1] <= 4 else video_data[:, 0, ...]

            for frame_idx in range(video_data.shape[0]):
                packed_frame = pack_bayer_to_quadrants(video_data[frame_idx])
                packed_frame = np.clip((packed_frame / 4095.0) * 65535.0, 0, 65535).astype(np.uint16)
                raw_file.write(packed_frame.tobytes())

    print("\n>>> [2/3] 正在启动 FFmpeg HQDN3D...")
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(FPS),
        "-color_range",
        "pc",
        "-i",
        str(TEMP_IN_RAW),
        "-vf",
        HQDN3D_FILTER,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "-color_range",
        "pc",
        str(TEMP_OUT_RAW),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("--- FFmpeg 完成 ---")

    print("\n>>> [3/3] 开始 ISP 渲染与视频封装...")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (width, height))

    bytes_per_frame = width * height * 2
    out_file_size = os.path.getsize(TEMP_OUT_RAW)
    out_frames_count = out_file_size // bytes_per_frame
    if out_frames_count != total_frames:
        print(f"输出帧数({out_frames_count}) 与输入帧数({total_frames}) 不一致。")

    isp = TeamStandardISP(JSON_PATH)
    with open(TEMP_OUT_RAW, "rb") as raw_stream:
        for frame_idx in tqdm(range(out_frames_count), desc="ISP 最终渲染"):
            raw_bytes = raw_stream.read(bytes_per_frame)
            if not raw_bytes:
                break

            packed_np = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(height, width)
            unpacked_np = unpack_quadrants_to_bayer(packed_np)
            raw_float = (unpacked_np.astype(np.float32) / 65535.0) * 4095.0
            frame_bgr = isp.process_frame(raw_float, is_first_frame=(frame_idx == 0))
            video_writer.write(frame_bgr)

    video_writer.release()

    if TEMP_IN_RAW.exists():
        TEMP_IN_RAW.unlink()
    if TEMP_OUT_RAW.exists():
        TEMP_OUT_RAW.unlink()

    print(f"\n[FINISH] 3DNR 视频已生成：{OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()

import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


COMPANY_DATA_ROOT = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\CompanyData")
OUTPUT_DIR = Path(r"D:\DeepLearning\VideoDenoising\NewDataSet\OpenCV3dnr")

FPS = 15
EXPOSURE_COMPENSATION = 50
HQDN3D_FILTER = "hqdn3d=0.001:0.001:8:3"

NOISY_KEY = "noisy"
DENOISE_INPUT_KEY = "2dnr"


def natural_sort_key(text):
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"([0-9]+)", text)]


def list_scene_dirs(root_dir):
    return sorted(
        [path for path in root_dir.iterdir() if path.is_dir() and path.name.lower().startswith("scene")],
        key=lambda path: natural_sort_key(path.name),
    )


def list_h5_files(scene_dir):
    files = [path for path in scene_dir.iterdir() if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}]
    files.sort(key=lambda path: path.name)
    return files


def squeeze_video_data(video_data, noisy_channel_index=None):
    """将 H5 里的视频数组压缩为 T x H x W。

    如果是 noisy 的双通道数据，可以通过 noisy_channel_index 指定要保留的通道。
    """
    video_data = np.squeeze(video_data)

    if video_data.ndim == 2:
        video_data = np.expand_dims(video_data, axis=0)

    while video_data.ndim > 3:
        if noisy_channel_index is not None and video_data.ndim == 4:
            video_data = video_data[:, noisy_channel_index, ...]
        elif video_data.shape[-1] <= 4:
            video_data = video_data[..., 0]
        else:
            video_data = video_data[:, 0, ...]

    if video_data.ndim != 3:
        raise ValueError(f"Unexpected video dimensions after squeeze: {video_data.shape}")

    return video_data


def get_h5_video_info(path, key):
    with h5py.File(path, "r") as h5_file:
        shape = h5_file[key].shape

    real_dims = [size for size in shape if size > 1]

    if len(real_dims) == 2:
        return 1, real_dims[0], real_dims[1]
    if len(real_dims) >= 3:
        return real_dims[0], real_dims[-2], real_dims[-1]

    raise ValueError(f"Cannot infer video shape from dataset {key} with shape {shape}")


def pack_bayer_to_quadrants(raw_img):
    height, width = raw_img.shape
    half_height, half_width = height // 2, width // 2

    packed = np.zeros_like(raw_img)
    packed[0:half_height, 0:half_width] = raw_img[0::2, 0::2]
    packed[0:half_height, half_width:width] = raw_img[0::2, 1::2]
    packed[half_height:height, 0:half_width] = raw_img[1::2, 0::2]
    packed[half_height:height, half_width:width] = raw_img[1::2, 1::2]
    return packed


def unpack_quadrants_to_bayer(packed_img):
    height, width = packed_img.shape
    half_height, half_width = height // 2, width // 2

    raw_img = np.zeros_like(packed_img)
    raw_img[0::2, 0::2] = packed_img[0:half_height, 0:half_width]
    raw_img[0::2, 1::2] = packed_img[0:half_height, half_width:width]
    raw_img[1::2, 0::2] = packed_img[half_height:height, 0:half_width]
    raw_img[1::2, 1::2] = packed_img[half_height:height, half_width:width]
    return raw_img


class TeamStandardISP:
    def __init__(self, json_path):
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as file:
                self.meta = json.load(file)
        else:
            self.meta = {}

        self.black_level = 256.0
        self.white_level = 4095.0
        self.cv_bayer_code = cv2.COLOR_BayerGB2BGR

        self.g1_slice = (slice(0, None, 2), slice(0, None, 2))
        self.b_slice = (slice(0, None, 2), slice(1, None, 2))
        self.r_slice = (slice(1, None, 2), slice(0, None, 2))
        self.g2_slice = (slice(1, None, 2), slice(1, None, 2))

        self.wb_gains = None

    def _compute_auto_white_balance(self, img):
        r_mean = np.mean(img[self.r_slice])
        b_mean = np.mean(img[self.b_slice])
        g_mean = (np.mean(img[self.g1_slice]) + np.mean(img[self.g2_slice])) / 2.0

        r_gain = g_mean / (r_mean + 1e-6)
        b_gain = g_mean / (b_mean + 1e-6)
        return [r_gain, 1.0, b_gain]

    def process_frame(self, raw_data, is_first_frame=False):
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


def scan_scene(scene_dir):
    h5_files = list_h5_files(scene_dir)
    if not h5_files:
        raise FileNotFoundError(f"No h5 files found in {scene_dir}")

    total_frames = 0
    height = width = None

    for h5_path in h5_files:
        noisy_frames, noisy_h, noisy_w = get_h5_video_info(h5_path, NOISY_KEY)
        denoise_frames, denoise_h, denoise_w = get_h5_video_info(h5_path, DENOISE_INPUT_KEY)

        if noisy_frames != denoise_frames:
            raise ValueError(
                f"Frame count mismatch in {h5_path.name}: {NOISY_KEY}={noisy_frames}, {DENOISE_INPUT_KEY}={denoise_frames}"
            )
        if (noisy_h, noisy_w) != (denoise_h, denoise_w):
            raise ValueError(
                f"Resolution mismatch in {h5_path.name}: "
                f"{NOISY_KEY}={noisy_w}x{noisy_h}, {DENOISE_INPUT_KEY}={denoise_w}x{denoise_h}"
            )

        total_frames += noisy_frames

        if height is None:
            height, width = noisy_h, noisy_w

    return h5_files, total_frames, height, width


def export_2dnr_to_raw(h5_files, temp_in_raw):
    with open(temp_in_raw, "wb") as raw_file:
        for h5_path in tqdm(h5_files, desc="Export 2DNR RAW", leave=False):
            with h5py.File(h5_path, "r") as h5_file:
                video_data = squeeze_video_data(h5_file[DENOISE_INPUT_KEY][:])

            for frame_idx in range(video_data.shape[0]):
                packed_frame = pack_bayer_to_quadrants(video_data[frame_idx])
                packed_frame = np.clip((packed_frame / 4095.0) * 65535.0, 0, 65535).astype(np.uint16)
                raw_file.write(packed_frame.tobytes())


def run_ffmpeg_3dnr(temp_in_raw, temp_out_raw, width, height):
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
        str(temp_in_raw),
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
        str(temp_out_raw),
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def render_compare_video(scene_dir, h5_files, temp_out_raw, output_video, total_frames, width, height):
    metadata_path = scene_dir / "metadata.json"
    noisy_isp = TeamStandardISP(str(metadata_path))
    denoised_isp = TeamStandardISP(str(metadata_path))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(str(output_video), fourcc, FPS, (width, height * 2))

    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_video}")

    bytes_per_frame = width * height * 2
    out_file_size = temp_out_raw.stat().st_size
    out_frames_count = out_file_size // bytes_per_frame

    if out_frames_count != total_frames:
        raise ValueError(
            f"FFmpeg output frame count mismatch: expected {total_frames}, got {out_frames_count} for {scene_dir.name}"
        )

    global_frame_idx = 0

    with open(temp_out_raw, "rb") as denoised_stream:
        with tqdm(total=total_frames, desc=f"Render {scene_dir.name}", leave=False) as progress_bar:
            for h5_path in h5_files:
                with h5py.File(h5_path, "r") as h5_file:
                    # noisy 通常是 [T, 2, H, W]，这里固定取当前帧通道 1。
                    noisy_data = squeeze_video_data(h5_file[NOISY_KEY][:], noisy_channel_index=1)

                for frame_idx in range(noisy_data.shape[0]):
                    raw_bytes = denoised_stream.read(bytes_per_frame)
                    if len(raw_bytes) != bytes_per_frame:
                        raise EOFError(f"Unexpected end of FFmpeg output while processing {scene_dir.name}")

                    packed_np = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(height, width)
                    denoised_raw = unpack_quadrants_to_bayer(packed_np)
                    denoised_raw = (denoised_raw.astype(np.float32) / 65535.0) * 4095.0

                    noisy_raw = noisy_data[frame_idx].astype(np.float32)
                    is_first_frame = global_frame_idx == 0

                    noisy_bgr = noisy_isp.process_frame(noisy_raw, is_first_frame=is_first_frame)
                    denoised_bgr = denoised_isp.process_frame(denoised_raw, is_first_frame=is_first_frame)

                    compare_frame = np.vstack((noisy_bgr, denoised_bgr))
                    video_writer.write(compare_frame)

                    global_frame_idx += 1
                    progress_bar.update(1)

            if denoised_stream.read(1):
                raise ValueError(f"FFmpeg output for {scene_dir.name} contains extra unread frames")

    video_writer.release()


def process_scene(scene_dir):
    h5_files, total_frames, height, width = scan_scene(scene_dir)

    print(f"\n=== {scene_dir.name} ===")
    print(f"Found {len(h5_files)} h5 files, total frames: {total_frames}, resolution: {width}x{height}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_video = OUTPUT_DIR / f"{scene_dir.name}_noisy_vs_3dnr.avi"
    temp_in_raw = OUTPUT_DIR / f"{scene_dir.name}_temp_in.raw"
    temp_out_raw = OUTPUT_DIR / f"{scene_dir.name}_temp_out.raw"

    try:
        export_2dnr_to_raw(h5_files, temp_in_raw)
        run_ffmpeg_3dnr(temp_in_raw, temp_out_raw, width, height)
        render_compare_video(scene_dir, h5_files, temp_out_raw, output_video, total_frames, width, height)
    finally:
        if temp_in_raw.exists():
            temp_in_raw.unlink()
        if temp_out_raw.exists():
            temp_out_raw.unlink()

    print(f"Saved compare video: {output_video}")


def main():
    scene_dirs = list_scene_dirs(COMPANY_DATA_ROOT)
    if not scene_dirs:
        raise FileNotFoundError(f"No scene folders found in {COMPANY_DATA_ROOT}")

    for scene_dir in scene_dirs:
        process_scene(scene_dir)

    print("\nAll scene compare videos have been generated.")


if __name__ == "__main__":
    main()

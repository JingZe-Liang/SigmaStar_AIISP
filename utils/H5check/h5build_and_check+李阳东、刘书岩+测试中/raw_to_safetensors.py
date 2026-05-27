import numpy as np
from safetensors.numpy import save_file
import os
from pathlib import Path

# ==================== 请在这里修改你的文件信息 ====================
RAW_VIDEO_PATH = r"D:\BaiduNetdiskDownload\Fortest\sc635hai/raw_stream_normal_3200x1800_16_RG_0205213025917_131x.raw"# 你的raw视频文件路径
OUTPUT_DIR = r"D:\BaiduNetdiskDownload/H5/sfts4"
ORI_WIDTH = 3200                           # 输入视频宽度（像素）
ORI_HEIGHT = 1800                          # 输入视频高度（像素）
TAR_WIDTH = 1920                           # 输出视频宽度（pixel）
TAR_HEIGHT = 1080                          # 输出视频宽度（pixel）
TOTAL_FRAMES = 150                     # 总帧数
# ================================================================

# 计算每帧的字节数：宽度 * 高度 * 每像素字节数（16位即2字节）
BYTES_PER_PIXEL = 2
ORI_FRAME_SIZE_BYTES = ORI_WIDTH * ORI_HEIGHT * BYTES_PER_PIXEL
TAR_FRAME_SIZE_BYTES = TAR_WIDTH * TAR_HEIGHT * BYTES_PER_PIXEL

CROP_LEFT=(ORI_WIDTH-TAR_WIDTH)//2
CROP_ABOVE=(ORI_HEIGHT-TAR_HEIGHT)//2

# 校验：偏移为偶数 + 目标分辨率为偶数（防止CFA错位）
assert TAR_WIDTH % 2 == 0 and TAR_HEIGHT % 2 == 0, "目标分辨率宽/高必须为偶数（GRBG CFA是2x2单元）"
assert CROP_LEFT % 2 == 0 and CROP_ABOVE % 2 == 0, "裁剪偏移必须为偶数，否则GRBG序列错位"

# 创建输出目录
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"开始处理文件: {RAW_VIDEO_PATH}")
print(f"每帧大小: {ORI_FRAME_SIZE_BYTES} 字节")

try:
    with open(RAW_VIDEO_PATH, "rb") as f:
        for frame_idx in range(TOTAL_FRAMES):
            # 1. 计算并定位到当前帧在文件中的偏移量
            offset = frame_idx * ORI_FRAME_SIZE_BYTES
            f.seek(offset)

            # 2. 读取当前帧的原始数据
            raw_frame_data = f.read(ORI_FRAME_SIZE_BYTES)

            # 3. 检查数据是否完整，防止文件被截断
            if len(raw_frame_data) < ORI_FRAME_SIZE_BYTES:
                print(f"警告: 第 {frame_idx + 1} 帧数据不完整，处理终止。")
                break

            # 4. 将字节数据转换为 NumPy 数组
            # 指定数据类型为 'uint16'，字节序为小端（'<u2'）
            ori_frame = np.frombuffer(raw_frame_data, dtype='<u2').reshape(ORI_HEIGHT, ORI_WIDTH)

            cropped_frame=ori_frame[
                CROP_ABOVE:CROP_ABOVE+TAR_HEIGHT,
                CROP_LEFT:CROP_LEFT+TAR_WIDTH
            ]

            # 校验裁剪后分辨率
            assert cropped_frame.shape == (TAR_HEIGHT, TAR_WIDTH), f"裁剪后分辨率错误，实际{cropped_frame.shape}"

            # 5. 验证GRBG CFA序列（可选，仅第一帧校验）
            # if frame_idx == 0:
            #     print("\n=== 第一帧GRBG CFA校验（关键位置）===")
            #     print(f"第0行第0列（应是G）: 像素值={cropped_frame[0,0]}")
            #     print(f"第0行第1列（应是R）: 像素值={cropped_frame[0,1]}")
            #     print(f"第1行第0列（应是B）: 像素值={cropped_frame[1,0]}")
            #     print(f"第1行第1列（应是G）: 像素值={cropped_frame[1,1]}")
            #     print(f"第2行第0列（应是G）: 像素值={cropped_frame[2,0]}")
            #     # 修正后的校验逻辑：仅验证位置符合GRBG行列规则，不强制数值相等
            #     # 偶数行（0、2、4...）第0列是G，奇数行（1、3、5...）第0列是B
            #     assert (cropped_frame[0,0] is not None) and (cropped_frame[1,0] is not None), "CFA基础位置异常"
            #     print("GRBG CFA位置规则校验通过（像素值因画面不同属正常）")
        
            # 5. （可选）如果需要，可以只保留低12位数据
            # frame = frame & 0x0FFF  # 这行注释掉了，因为大部分训练框架能直接处理16位输入
            
            # 6. 构建一个字典，键为张量名称（这里用 "frame" 是标准做法）
            tensors = {"frame": cropped_frame}

            # 7. 保存为 .safetensors 文件
            output_filename = f"frame{frame_idx + 1:04d}.safetensors"  # 格式化为 frame0001.safetensors
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            save_file(tensors, output_path)

            # 打印进度
            if (frame_idx + 1) % 10 == 0:
                print(f"已处理并保存: {frame_idx + 1}/{TOTAL_FRAMES} 帧")

    print(f"\n🎉 全部完成！共保存了 {frame_idx + 1} 个 .safetensors 文件到目录: {OUTPUT_DIR}")

except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 '{RAW_VIDEO_PATH}'，请检查路径是否正确。")
except Exception as e:
    print(f"❌ 处理过程中发生意外错误: {e}")
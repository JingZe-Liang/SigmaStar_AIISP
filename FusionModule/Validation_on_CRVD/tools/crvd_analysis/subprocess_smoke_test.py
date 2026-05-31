import subprocess
import os
from pathlib import Path


def run_vbm3d_stable():
    # 1. 核心路径定义
    # 算子所在的 bin 目录
    bin_dir = Path(r"E:/Reasearch/Code/vbm3d_1/build/bin")
    # 使用绝对路径定位 EXE，彻底解决 WinError 2
    exe_full_path = str(bin_dir / "VBM3Ddenoising.exe")

    # 2. 自动化路径预检
    # 探测数据文件的物理位置，确保 i0001.png 真的在那里
    absolute_data_sample = Path(r"E:/Reasearch/Code/vbm3d_1/vbm3d_1/data/i0001.png")

    if not absolute_data_sample.exists():
        print(f"❌ 预检失败：文件不存在 -> {absolute_data_sample}")
        return

    # 3. 构造参数
    # 因为我们要把 cwd 设为 bin_dir，所以数据路径要相对于 bin_dir 向上跳两级
    input_pattern = "../../vbm3d_1/data/i%04d.png"

    args = [
        exe_full_path,  # 参数 0：绝对路径，Windows 必中
        "-i", input_pattern,
        "-f", "1",
        "-l", "9",
        "-sigma", "20",
        "-add", "true"
    ]

    print(f"🛠️ 准备执行算子...")
    print(f"📂 工作目录: {bin_dir}")
    print(f"💻 命令: {' '.join(args)}")

    try:
        # 执行算子
        process = subprocess.run(
            args,
            cwd=str(bin_dir),  # 设置工作目录，使相对路径 ../../ 有效
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # 结果输出
        if process.returncode == 0:
            print("✅ 算子运行成功！")
            # VBM3D 默认输出 PSNR 和 RMSE 指标到标准输出
            if "PSNR" in process.stdout:
                print("--- 算法核心指标 ---")
                lines = process.stdout.split('\n')
                for line in lines[-10:]:  # 打印最后几行关键数据
                    if line.strip(): print(line)
        else:
            print(f"❌ 算子执行异常退出 (Code {process.returncode})")
            print(f"错误日志: {process.stderr}")

    except Exception as e:
        print(f"🚨 脚本异常: {e}")


if __name__ == "__main__":
    run_vbm3d_stable()
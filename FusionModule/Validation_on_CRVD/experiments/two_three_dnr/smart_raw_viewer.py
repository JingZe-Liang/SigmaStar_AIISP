import numpy as np
import cv2
import os
import re
from tkinter import filedialog, Tk, simpledialog, messagebox


def smart_parse_filename(filename):
    res = re.search(r'(\d+)x(\d+)', filename)
    w = int(res.group(1)) if res else 2560
    h = int(res.group(2)) if res else 1440
    return w, h


def run_debugger():
    root = Tk()
    root.withdraw()

    # 1. 选择文件
    file_path = filedialog.askopenfilename(title="选择 RAW 文件")
    if not file_path: return
    filename = os.path.basename(file_path)

    # 2. 确认基础参数
    w_def, h_def = smart_parse_filename(filename)
    width = simpledialog.askinteger("输入", "图像宽度 (Width):", initialvalue=w_def)
    height = simpledialog.askinteger("输入", "图像高度 (Height):", initialvalue=h_def)
    bit_depth = simpledialog.askinteger("输入", "位深 (8 或 16):", initialvalue=16)

    if not width or not height: return

    # 3. 读取数据
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    try:
        raw_data = np.fromfile(file_path, dtype=dtype)
        # 防止文件末尾有 Metadata 导致 reshape 失败
        img = raw_data[:width * height].reshape((height, width))
    except Exception as e:
        messagebox.showerror("错误", f"无法以 {width}x{height} 解析文件。\n{e}")
        return

    # 4. 自动亮度增强 (针对 darkroom)
    # 将数据拉伸到 0-255 以便显示
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img, dtype=np.uint8)

    # 5. 交互循环
    bayer_patterns = [
        (cv2.COLOR_BayerBG2BGR, "BGGR"),
        (cv2.COLOR_BayerGB2BGR, "GBRG"),
        (cv2.COLOR_BayerRG2BGR, "RGGB"),
        (cv2.COLOR_BayerGR2BGR, "GRBG")
    ]
    idx = 0

    print("\n--- 操作指南 ---")
    print("使用键盘 [A] 或 [D] 键切换 Bayer 模式")
    print("使用 [ESC] 或 [Q] 退出程序")

    cv2.namedWindow("RAW Debugger (A/D to Switch)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RAW Debugger (A/D to Switch)", 1280, 720)

    while True:
        mode_code, mode_name = bayer_patterns[idx]

        # 转换色彩
        color_display = cv2.cvtColor(img_norm, mode_code)

        # 在图上标注当前模式
        cv2.putText(color_display, f"Mode: {mode_name} (Index: {idx})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow("RAW Debugger (A/D to Switch)", color_display)

        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        if key == ord('d') or key == 83:  # 'd' 或 右方向键 (某些系统)
            idx = (idx + 1) % 4
        elif key == ord('a') or key == 81:  # 'a' 或 左方向键
            idx = (idx - 1) % 4
        elif key == 27 or key == ord('q'):  # ESC 或 q 退出
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_debugger()
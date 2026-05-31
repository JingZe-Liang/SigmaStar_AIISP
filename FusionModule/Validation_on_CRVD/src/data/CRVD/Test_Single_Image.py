import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    # 脚本在Test/Data目录
    image_path = Path(r"D:\A_Data\CRVD_dataset\indoor_raw_noisy\scene1\ISO1600\frame1_clean.tiff")

    print(f"Loading image from: {image_path}")

    # 加载图片到中间变量
    img = cv2.imread(str(image_path), -1)

    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # 转换为float32
    img_float = img.astype(np.float32)

    print(f"Image loaded successfully")
    print(f"Shape: {img_float.shape}")
    print(f"Dtype: {img_float.dtype}")
    print(f"Range: [{img_float.min():.2f}, {img_float.max():.2f}]")

    # === 在这里设置断点查看 img_float 变量 ===
    print(f"\n=== Set breakpoint here ===")
    print(f"img_float type: {type(img_float)}")

    # 显示图片
    plt.figure(figsize=(10, 8))
    if len(img_float.shape) == 2:
        plt.imshow(img_float, cmap='gray')
        plt.title(f"Raw Image (Single Channel)\nlocation: {image_path}")
    else:
        plt.imshow(img_float)
        plt.title(f"Raw Image\nShape: {img_float.shape}")
    plt.colorbar(label='Pixel Value')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
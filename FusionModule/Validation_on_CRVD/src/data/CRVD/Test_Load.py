from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"

#   For Debug

    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1,2],
        iso_levels=[1600,3200],
        num_frames=7,
        # sequence_mode = False
    )

    print(f"Total samples: {len(dataset)}")



    noisy, gt = dataset[0]   #Debug here
    print(f"\nSample 0:")
    print(f"Noisy: {noisy.shape}, range [{noisy.min():.1f}, {noisy.max():.1f}]")
    print(f"GT: {gt.shape}, range [{gt.min():.1f}, {gt.max():.1f}]")

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch_noisy, batch_gt = next(iter(loader))
    print(f"\nBatch test:")
    print(f"Batch noisy: {batch_noisy.shape}")
    print(f"Batch GT: {batch_gt.shape}")

    print("\n=== Set breakpoint here to inspect data ===")

    noisy, gt = dataset[2]


#   Visualize and check

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(noisy[0], cmap='gray')
    axes[0].set_title(f'Noisy\nRange: [{noisy.min():.0f}, {noisy.max():.0f}]')
    axes[0].axis('off')

    axes[1].imshow(gt[0], cmap='gray')
    axes[1].set_title(f'GT\nRange: [{gt.min():.0f}, {gt.max():.0f}]')
    axes[1].axis('off')

    diff = np.abs(noisy[0] - gt[0])
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Abs Difference\nMean: {diff.mean():.2f}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # 检查基本统计量
    print(f"Noisy mean: {noisy.mean():.2f}, std: {noisy.std():.2f}")
    print(f"GT mean: {gt.mean():.2f}, std: {gt.std():.2f}")
    print(f"PSNR: {10 * np.log10(4095 ** 2 / ((noisy - gt) ** 2).mean()):.2f} dB")


if __name__ == "__main__":
    main()
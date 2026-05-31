from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
from piq import psnr, ssim


from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset
from Liang.src.denoise.Denoising_Pipeline.Denoising_Pipeline_1 import Denoising1
from Liang.src.isp_utils.CRVD_OpenCV_ISP.OpenCV_ISP import OpenCV_ISP2
from Liang.utils.quick_video_eval_gt import quick_evaluate


def main() -> None:
    noisy_root = "E:/CRVD_dataset/indoor_raw_noisy"
    gt_root = "E:/CRVD_dataset/indoor_raw_gt"

    torch.manual_seed(10)
    denoise = Denoising1()

    dataset = CRVDDataset(
        noisy_root=noisy_root,
        gt_root=gt_root,
        scenes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        iso_levels=[1600, 3200],
        num_frames=7,
        # sequence_mode = False
    )
    print(f"Total samples: {len(dataset)},Maximum is 55")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Number of batches: {len(loader)}")

    ISP2 = OpenCV_ISP2()
    print("Objectivation of OpenCVISP2 finished")
    # ======================================
    #   降噪模块    --- plug here
    # ======================================

    for nosiy, gt in loader:  # 验证数据导入并进行ISP2

        sRGB_nosiy = ISP2(nosiy)
        print(sRGB_nosiy.shape)
        sRGB_gt = ISP2(gt)
        print(sRGB_gt.shape)
        sRGB_denosie = ISP2(denoise(nosiy, 12))
        print(sRGB_denosie.shape)
        results = quick_evaluate(sRGB_nosiy, sRGB_gt, sRGB_denosie)

        # ABS = np.abs(sRGB_nosiy[0,0] - sRGB_gt[0,0])
        #
        # plt.figure(figsize=(8, 5))
        # plt.imshow(sRGB_denosie[1,0], cmap="gray")
        # plt.title("Modular Fast ISP - denoisy")
        # plt.axis('off')
        # plt.show()

    # print("Plot finished")
    print("Process finished")


if __name__ == "__main__":
    main()


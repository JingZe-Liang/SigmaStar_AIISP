#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import h5py
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from typing import Tuple
from typing import Optional

from wt import WaveletThresholding


def compute_metrics(clean: np.ndarray, test: np.ndarray, data_range: int) -> Tuple[float, float]:
    """计算 PSNR 和 SSIM"""
    clean_f = clean.astype(np.float64)
    test_f = test.astype(np.float64)
    psnr_val = psnr(clean_f, test_f, data_range=data_range)
    ssim_val = ssim(clean_f, test_f, data_range=data_range)
    return psnr_val, ssim_val


def process_h5_file(
    h5_path: str,
    scene_id: int,
    shard_idx: int,
    noisy_key: str = "noisy",
    clean_key: str = "clean",
    noisy_axis: int = 0,
    cfa_pattern: str = "GBRG",
    clip: int = 4095,
    save_denoised: bool = False,
    output_dir: Optional[str] = None,
    **wt_kwargs
) -> Optional[pd.DataFrame]:
    """
    处理单个H5文件，应用小波阈值去噪并计算指标
    """
    with h5py.File(h5_path, 'r') as f:
        if noisy_key not in f or clean_key not in f:
            print(f"警告: {h5_path} 缺少 {noisy_key}/{clean_key} 数据集，跳过")
            return None

        noisy_ds = f[noisy_key]
        clean_ds = f[clean_key]

        # 解析noisy形状
        shape = noisy_ds.shape
        if len(shape) == 4:
            N, D, H, W = shape
            if noisy_axis >= D:
                raise ValueError(f"noisy_axis={noisy_axis} 超出范围 (0-{D-1})")
        elif len(shape) == 3:
            N, H, W = shape
            D = 1
        else:
            raise ValueError(f"不支持的noisy形状: {shape}")

        # 检查clean形状
        if clean_ds.ndim == 3:
            clean_N, clean_H, clean_W = clean_ds.shape
            if clean_N != N or clean_H != H or clean_W != W:
                print(f"警告: clean形状 {clean_ds.shape} 与noisy不匹配，截取最小维度")
                N = min(N, clean_N)
        elif clean_ds.ndim == 2:
            clean_H, clean_W = clean_ds.shape
            if clean_H != H or clean_W != W:
                raise ValueError(f"clean尺寸 {clean_H}x{clean_W} 与noisy {H}x{W} 不匹配")
            N = 1
        else:
            raise ValueError(f"不支持的clean维度: {clean_ds.ndim}")

        if H % 2 != 0 or W % 2 != 0:
            print(f"警告: {h5_path} 尺寸 {H}x{W} 不是偶数，跳过")
            return None

        results = []
        denoised_list = []

        # 逐帧处理
        for idx in tqdm(range(N), desc=f"  Scene{scene_id}_shard{shard_idx}", leave=False):
            # 读取噪声图像
            if len(shape) == 4:
                img = noisy_ds[idx, noisy_axis, :, :]
            else:
                img = noisy_ds[idx]
            img = np.squeeze(img).astype(np.uint16)

            # 读取干净图像
            if clean_ds.ndim == 3:
                clean = clean_ds[idx]
            elif clean_ds.ndim == 2:
                clean = clean_ds[()]
            clean = np.squeeze(clean).astype(img.dtype)

            # 计算原始噪声图PSNR
            noisy_psnr, _ = compute_metrics(clean, img, data_range=clip)

            # 小波阈值去噪
            wt = WaveletThresholding(
                img=img,
                clip=clip,
                cfa_pattern=cfa_pattern,
                **wt_kwargs
            )
            denoised = wt.execute()

            if save_denoised:
                denoised_list.append(denoised)

            # 计算去噪后指标
            denoised_psnr, denoised_ssim = compute_metrics(clean, denoised, data_range=clip)

            results.append({
                "scene": scene_id,
                "shard": shard_idx,
                "frame": idx,
                "noisy_psnr": noisy_psnr,
                "wt_psnr": denoised_psnr,
                "wt_ssim": denoised_ssim,
            })

        df = pd.DataFrame(results)

        # 保存去噪后的H5
        if save_denoised and output_dir and denoised_list:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, os.path.basename(h5_path))
            with h5py.File(out_path, 'w') as f_out:
                if len(denoised_list) == 1:
                    f_out.create_dataset('2dnr', data=denoised_list[0], compression='gzip')
                else:
                    f_out.create_dataset('2dnr', data=np.stack(denoised_list, axis=0), compression='gzip')
                f_out.create_dataset('noisy', data=noisy_ds[...])
                f_out.create_dataset('clean', data=clean_ds[...])
            print(f"    已保存去噪H5: {out_path}")

        return df


def batch_test(
    root_dir: str = "./dataset",
    scene_ids: list = [1],
    shard_pattern: str = "shard_*.h5",
    output_csv: str = "wt_results.csv",
    save_denoised_h5: bool = True,
    output_denoised_dir: str = "./denoised_h5_wt",
    noisy_axis: int = 0,
    **wt_kwargs
) -> Optional[pd.DataFrame]:
    """
    批量测试小波阈值去噪
    """
    all_dfs = []

    for scene_id in scene_ids:
        scene_dir = os.path.join(root_dir, f"scene{scene_id}")
        if not os.path.isdir(scene_dir):
            print(f"跳过不存在的目录: {scene_dir}")
            continue
        
        h5_files = sorted(glob.glob(os.path.join(scene_dir, shard_pattern)))
        shard_indices = []
        for path in h5_files:
            m = re.search(r'shard_(\d+)', os.path.basename(path))
            shard_indices.append(int(m.group(1)) if m else -1)

        print(f"场景 scene{scene_id} 找到 {len(h5_files)} 个H5文件")
        for h5_path, shard_idx in zip(h5_files, shard_indices):
            print(f"\n处理 {os.path.basename(h5_path)} (shard {shard_idx})")
            df = process_h5_file(
                h5_path=h5_path,
                scene_id=scene_id,
                shard_idx=shard_idx,
                noisy_axis=noisy_axis,
                save_denoised=save_denoised_h5,
                output_dir=output_denoised_dir,
                **wt_kwargs
            )
            if df is not None and not df.empty:
                all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"\n结果已保存至 {output_csv}")
        print("\n========== 汇总统计 ==========")
        print(f"平均 noisy_psnr : {final_df['noisy_psnr'].mean():.2f} dB")
        print(f"平均 wt_psnr    : {final_df['wt_psnr'].mean():.2f} dB")
        print(f"平均 wt_ssim    : {final_df['wt_ssim'].mean():.4f}")
        print(f"PSNR 提升       : {final_df['wt_psnr'].mean() - final_df['noisy_psnr'].mean():.2f} dB")
        return final_df
    else:
        print("没有成功处理任何图像。")
        return None


if __name__ == "__main__":
    # 小波阈值去噪参数配置
    wt_kwargs = {
        "wavelet": "db4",               # 小波基函数
        "level": 3,                     # 分解层数
        "threshold_method": "visu",     # 阈值计算方法
        "threshold_factor": 1.2,        # 阈值缩放因子（可根据噪声强度调整）
        "threshold_type": "soft",       # 软阈值（更平滑）/硬阈值（保留细节）
        "cfa_pattern": "GBRG",          # CFA模式
        "clip": 4095                    # 像素裁剪上限
    }

    # 批量测试
    batch_test(
        root_dir="./dataset",
        scene_ids=[1],
        shard_pattern="shard_*.h5",
        output_csv="wt_results_scene1.csv",
        save_denoised_h5=True,
        output_denoised_dir="./denoised_h5_wt",
        noisy_axis=0,
        **wt_kwargs
    )
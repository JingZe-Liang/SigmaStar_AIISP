from __future__ import annotations

"""H5 split discovery and DataLoader construction.

本模块负责：
1. 根据场景和分片信息发现训练/验证/测试数据分割
2. 构建PyTorch Dataset和DataLoader用于模型训练
3. 管理多进程数据加载时的随机种子同步
"""

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, get_worker_info

from MyNet.ai_fusion.dataset_h5 import H5FusionDataset, H5FusionDatasetConfig


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitStats:
    """存储数据集分割统计信息的数据类。
    
    Attributes:
        split: 数据集分割名称 ('train', 'val', 'test')
        frames: 该分割中的总帧数（样本数量）
        scenes: 该分割中包含的场景数量
        shards: 每个场景对应的分片文件映射 {scene_name: [shard_files]}
    """
    split: str
    frames: int
    scenes: int
    shards: dict[str, list[str]]


def build_split_dataset(
    root_dir: Path,
    split: SplitName,
    scenes: list[str] | None,
    crop_size: int | None,
    random_crop: bool,
    cfa_pattern: str,
    seed: int,
    hard_crop_map: dict[str, Sequence[dict[str, Any]]] | None = None,
    hard_crop_prob: float = 0.0,
) -> tuple[Subset[dict[str, Any]], SplitStats]:
    """构建指定分割的数据集。
    
    Args:
        root_dir: H5数据根目录路径
        split: 数据集分割类型 ('train', 'val', 'test')
        scenes: 要包含的场景列表，None表示所有场景
        crop_size: 裁剪尺寸（高度和宽度），None表示不裁剪
        random_crop: 是否使用随机裁剪，False则使用中心裁剪
        cfa_pattern: CFA模式字符串（如'GBRG'）
        seed: 随机种子
        
    Returns:
        tuple包含:
        - Subset: PyTorch子集数据集，仅包含选定分割的样本索引
        - SplitStats: 分割统计信息对象
        
    Note:
        返回的Subset数据集在__getitem__时会输出字典，其中关键张量形状:
        - noisy_pair: [2, H, W] (前后两帧噪声图像)
        - prev4/curr4: [4, H//2, W//2] (Bayer打包后的前后帧)
        - motion_prior: [4, H//2, W//2] (运动先验，|curr4-prev4|)
        - dnr2/dnr3/clean: [H, W] (全分辨率目标图像)
    """
    # 发现当前分割对应的分片映射 {scene_name: set(shard_filenames)}
    split_map = discover_split_shards(root_dir, split=split, scenes=scenes)
    
    # 创建基础H5融合数据集实例，配置预处理参数
    base_dataset = H5FusionDataset(
        H5FusionDatasetConfig(
            root_dir=root_dir,
            scenes=scenes,
            crop_size=crop_size,
            random_crop=random_crop,
            cfa_pattern=cfa_pattern,
            seed=seed,
            hard_crop_map=hard_crop_map,
            hard_crop_prob=hard_crop_prob,
        )
    )
    
    # 获取数据集内部样本索引列表，每个元素是_SampleIndex对象
    sample_index = getattr(base_dataset, "_index", None)
    if sample_index is None:
        raise RuntimeError("H5FusionDataset no longer exposes _index; update split filtering code.")

    # 筛选属于当前分割的样本索引
    indices: list[int] = []
    for idx, item in enumerate(sample_index):
        allowed = split_map.get(item.scene_name, set())  # 当前场景允许的分片集合
        if item.shard_path.name in allowed:  # 检查样本所在分片是否在允许列表中
            indices.append(idx)

    if not indices:
        raise ValueError(f"No samples selected for split={split!r} under {root_dir}")

    # 构建分割统计信息对象
    stats = SplitStats(
        split=split,
        frames=len(indices),  # 当前分割的总样本数
        scenes=len(split_map),  # 涉及场景数
        shards={scene: sorted(shards) for scene, shards in split_map.items()},  # 各场景分片详情
    )
    
    # 返回子集数据集（仅包含选定索引）和统计信息
    return Subset(base_dataset, indices), stats


def discover_split_shards(root_dir: Path, split: SplitName, scenes: list[str] | None) -> dict[str, set[str]]:
    """发现并返回指定分割对应的分片映射。
    
    根据metadata.json中的分片信息，按照规则划分训练/验证/测试集:
    - train: 除最后两个分片外的所有分片
    - val: 倒数第二个分片  
    - test: 最后一个分片
    
    Args:
        root_dir: H5数据根目录路径
        split: 分割类型 ('train', 'val', 'test')
        scenes: 要处理的场景列表，None表示所有场景
        
    Returns:
        dict: {scene_name: set(shard_filenames)} 映射，表示每个场景在该分割中使用的分片文件集合
        
    Raises:
        ValueError: 当场景分片数不足3个或分割类型不支持时抛出异常
    """
    # 解析场景目录列表，支持单场景或多场景模式
    scene_dirs = resolve_scene_dirs(root_dir, scenes)
    selected: dict[str, set[str]] = {}  # 存储最终选定的分片映射
    
    for scene_dir in scene_dirs:
        # 读取场景元数据，包含分片列表及其全局起始索引等信息
        metadata = read_json(scene_dir / "metadata.json")
        
        # 按全局起始索引和分片ID排序分片，确保时间顺序正确
        shards = sorted(
            metadata["shards"],
            key=lambda item: (int(item.get("global_start_idx", 0)), int(item.get("shard_id", 0))),
        )
        
        # 至少需要3个分片才能进行train/val/test分割（前N-2为train，第N-1为val，第N为test）
        if len(shards) < 3:
            raise ValueError(f"{scene_dir} needs at least 3 shards for train/val/test split.")
            
        # 根据分割类型选择对应分片
        if split == "train":
            split_shards = shards[:-2]  # 训练集：除最后两个外的所有分片
        elif split == "val":
            split_shards = [shards[-2]]  # 验证集：倒数第二个分片
        elif split == "test":
            split_shards = [shards[-1]]  # 测试集：最后一个分片
        else:
            raise ValueError(f"Unsupported split: {split}")
            
        # 将选中分片的文件名存入集合，以场景名为键保存
        selected[scene_dir.name] = {str(item["file"]) for item in split_shards}
        
    return selected


def resolve_scene_dirs(root_dir: Path, scenes: list[str] | None) -> list[Path]:
    """解析并返回场景目录列表。
    
    支持两种模式:
    1. root_dir直接指向某个scene_x目录（该目录下有metadata.json）
    2. root_dir指向H5根目录，自动扫描所有含metadata.json的子目录作为场景
    
    Args:
        root_dir: 根目录路径，可以是H5根目录或单个scene目录
        scenes: 指定的场景名称列表，None表示使用所有找到的场景
        
    Returns:
        list[Path]: 排序后的场景目录路径列表
        
    Raises:
        ValueError: 当指定的场景不存在于目录下时抛出异常
    """
    root_dir = Path(root_dir)
    
    # 判断root_dir是否为单个场景目录（直接包含metadata.json）
    if (root_dir / "metadata.json").exists():
        scene_dirs = [root_dir]  # 单场景模式
    else:
        # 多场景模式：扫描所有包含metadata.json的子目录并按编号排序
        scene_dirs = sorted(
            [path for path in root_dir.iterdir() if path.is_dir() and (path / "metadata.json").exists()],
            key=scene_sort_key,  # 使用scene_sort_key确保scene_2排在scene_10前面
        )
        
    # 如果未指定特定场景，返回所有找到的场景目录
    if scenes is None:
        return scene_dirs
        
    # 否则过滤出指定的场景目录
    target_names = {normalize_scene_name(scene) for scene in scenes}  # 标准化场景名称格式
    selected = [scene_dir for scene_dir in scene_dirs if scene_dir.name in target_names]
    
    # 检查是否有请求的场景未找到
    missing = target_names.difference({scene_dir.name for scene_dir in selected})
    if missing:
        raise ValueError(f"Requested scenes not found under {root_dir}: {sorted(missing)}")
        
    return selected


def build_loader(
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    sample_weights: Sequence[float] | None = None,
    sampler_num_samples: int | None = None,
    sampler_replacement: bool = True,
    drop_last: bool | None = None,
) -> DataLoader[Any]:
    """构建PyTorch DataLoader。
    
    配置了适当的随机种子、内存锁定和持久化工作进程以优化性能。
    
    Args:
        dataset: PyTorch Dataset实例
        batch_size: 批次大小
        num_workers: 数据加载工作进程数，0表示在主进程中加载
        shuffle: 是否打乱数据顺序（通常仅训练集设为True）
        seed: 随机种子，用于可重复的数据打乱和工作进程初始化
        
    Returns:
        DataLoader: 配置好的PyTorch数据加载器
        
    Note:
        - pin_memory=True时启用CUDA固定内存传输加速（如果可用）
        - persistent_workers=True保持工作进程存活以减少进程创建开销
        - worker_init_fn确保每个工作进程有独立但可重现的随机状态
        - drop_last=True在shuffle时丢弃不完整的最后一个batch
    """
    # 创建随机数生成器并设置种子，保证数据打乱的可重复性
    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = None
    if sample_weights is not None:
        if len(sample_weights) != len(dataset):
            raise ValueError(f"sample_weights length {len(sample_weights)} does not match dataset length {len(dataset)}")
        weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
        if not torch.all(torch.isfinite(weights_tensor)):
            raise ValueError("sample_weights must be finite")
        if torch.any(weights_tensor < 0):
            raise ValueError("sample_weights must be non-negative")
        if float(weights_tensor.sum()) <= 0.0:
            raise ValueError("sample_weights must contain at least one positive value")
        num_samples = int(sampler_num_samples or len(weights_tensor))
        if num_samples <= 0:
            raise ValueError("sampler_num_samples must be > 0 when provided")
        sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=num_samples,
            replacement=sampler_replacement,
            generator=generator,
        )
        shuffle = False

    if drop_last is None:
        drop_last = shuffle or sampler is not None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # CUDA可用时启用固定内存加速GPU传输
        persistent_workers=num_workers > 0,     # 工作进程>0时保持进程持久化
        worker_init_fn=seed_worker,             # 工作进程初始化函数，设置各进程随机种子
        generator=generator,                    # 用于shuffle的随机数生成器
        drop_last=drop_last,                    # 训练或带sampler时丢弃末尾不完整batch
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader工作进程的随机种子初始化函数。
    
    确保每个工作进程拥有独立但可重现的随机状态，这对数据增强和shuffle至关重要。
    
    Args:
        worker_id: 工作进程ID（从0开始）
        
    Note:
        1. 首先基于torch.initial_seed()派生worker_seed，保证不同worker有不同种子但整体可重现
        2. 分别设置random、numpy的随机种子以覆盖不同类型的随机操作
        3. 如果数据集有_rng属性（如H5FusionDataset），也为其设置独立种子以保证crop等操作可重现
    """
    # 从PyTorch初始种子派生出工作进程专用种子（取模2^32避免溢出）
    worker_seed = torch.initial_seed() % 2**32
    
    # 设置Python内置random模块和NumPy的随机种子，确保各种随机操作可重现
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    # 获取当前工作进程的信息对象（仅在worker进程中有效）
    worker_info = get_worker_info()
    if worker_info is None:
        return  # 如果在主进程中调用则直接返回
        
    # 获取工作进程负责的数据集实例（可能是Subset包装的）
    dataset = worker_info.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset  # 解包Subset获取底层真实数据集
        
    # 如果数据集有_rng属性（如H5FusionDataset中的numpy随机数生成器），为其设置种子
    if hasattr(dataset, "_rng"):
        dataset._rng = np.random.default_rng(worker_seed)  # type: ignore[attr-defined]


def close_dataset(dataset: Dataset[Any]) -> None:
    """安全关闭数据集资源，特别是H5文件句柄。
    
    Args:
        dataset: PyTorch Dataset实例，可能是Subset包装的
        
    Note:
        - 如果传入的是Subset，会递归获取其底层数据集
        - 调用数据集的close方法（如果存在）来释放H5文件句柄等资源
        - 用于训练结束后清理资源，防止文件句柄泄漏
    """
    # 如果是Subset包装的数据集，获取其底层真实数据集
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
        
    # 安全调用数据集的close方法（如果存在）
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def scene_sort_key(path: Path) -> tuple[int, str]:
    """场景目录排序键函数，确保数值顺序而非字典序。
    
    例如：scene_2应该排在scene_10前面，而不是按字符串比较排在后面。
    
    Args:
        path: 场景目录路径（如/path/to/scene_5）
        
    Returns:
        tuple[int, str]: 排序键，(场景编号, 目录名)
        - 如果目录名末尾是数字，则提取该数字作为第一排序键
        - 如果不是数字，则使用极大值10^9使其排在最后，再按名称排序
        
    Example:
        scene_1 -> (1, 'scene_1')
        scene_2 -> (2, 'scene_2')  
        scene_10 -> (10, 'scene_10')
        other_dir -> (1000000000, 'other_dir')
    """
    suffix = path.name.split("_")[-1]  # 提取下划线后的部分（期望是数字）
    return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)


def normalize_scene_name(scene: str) -> str:
    """标准化场景名称为统一的'scene_x'格式。
    
    Args:
        scene: 原始场景标识，可以是数字、字符串如'1'、'scene_1'等
        
    Returns:
        str: 标准化后的场景名称，格式为'scene_x'
        
    Example:
        '1' -> 'scene_1'
        'scene_1' -> 'scene_1'
        5 -> 'scene_5' (如果传入int)
    """
    scene_name = str(scene)
    return scene_name if scene_name.startswith("scene_") else f"scene_{scene_name}"


def load_hard_sampling_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load build_hard_sampling_index.py output and index records by sample_id."""
    payload = read_json(path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not look like a hard-sampling index: missing list field 'samples'.")

    records: dict[str, dict[str, Any]] = {}
    for record in samples:
        if not isinstance(record, dict):
            continue
        sample_id = record.get("sample_id")
        if sample_id is None:
            scene_name = record.get("scene_name")
            scene_frame_index = record.get("scene_frame_index")
            if scene_name is None or scene_frame_index is None:
                continue
            sample_id = f"{scene_name}:{int(scene_frame_index)}"
        records[str(sample_id)] = record

    if not records:
        raise ValueError(f"{path} did not contain any usable hard-sampling records.")
    return records


def build_hard_crop_map(
    records: dict[str, dict[str, Any]],
    min_sample_weight: float,
    min_gap_weight: float,
    top_k: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Filter hard-index crop candidates for dataset-level hard crop."""
    hard_crop_map: dict[str, list[dict[str, Any]]] = {}
    for sample_id, record in records.items():
        sample_weight = float(record.get("sample_weight", 1.0))
        gap_weight = float(record.get("psnr_gap_weight", 1.0))
        if sample_weight < min_sample_weight or gap_weight < min_gap_weight:
            continue

        candidates = record.get("crop_candidates", [])
        if not isinstance(candidates, list):
            continue
        selected = candidates if top_k is None or top_k <= 0 else candidates[:top_k]

        cleaned: list[dict[str, Any]] = []
        for candidate in selected:
            if not isinstance(candidate, dict):
                continue
            xyxy = candidate.get("xyxy")
            if not isinstance(xyxy, Sequence) or len(xyxy) != 4:
                continue
            cleaned.append(
                {
                    "xyxy": [int(value) for value in xyxy],
                    "score": float(candidate.get("score", 0.0)),
                }
            )
        if cleaned:
            hard_crop_map[sample_id] = cleaned
    return hard_crop_map


def summarize_hard_crop_map(hard_crop_map: dict[str, Sequence[dict[str, Any]]] | None) -> dict[str, Any]:
    if not hard_crop_map:
        return {"samples": 0, "crop_candidates": 0}
    return {
        "samples": len(hard_crop_map),
        "crop_candidates": int(sum(len(candidates) for candidates in hard_crop_map.values())),
    }


def build_hard_sample_weights(
    dataset: Dataset[Any],
    records: dict[str, dict[str, Any]],
    weight_key: str,
    default_weight: float,
) -> tuple[list[float], dict[str, Any]]:
    """Build sampler weights aligned to a Dataset or Subset."""
    if default_weight <= 0.0:
        raise ValueError("default hard sample weight must be > 0")

    weights: list[float] = []
    matched = 0
    missing = 0
    for sample_id in iter_dataset_sample_ids(dataset):
        record = records.get(sample_id)
        if record is None:
            weight = default_weight
            missing += 1
        else:
            raw_weight = record.get(weight_key, default_weight)
            weight = float(raw_weight)
            matched += 1
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Invalid hard sample weight for {sample_id}: {weight!r}")
        weights.append(weight)

    if not weights:
        raise ValueError("Cannot build hard sampler weights for an empty dataset.")

    weights_array = np.asarray(weights, dtype=np.float64)
    return weights, {
        "records": len(records),
        "matched_samples": matched,
        "missing_samples": missing,
        "weight_key": weight_key,
        "weight_min": float(np.min(weights_array)),
        "weight_mean": float(np.mean(weights_array)),
        "weight_max": float(np.max(weights_array)),
        "weight_sum": float(np.sum(weights_array)),
    }


def iter_dataset_sample_ids(dataset: Dataset[Any]) -> list[str]:
    base_dataset: Dataset[Any]
    indices: Sequence[int]
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = range(len(dataset))

    sample_index = getattr(base_dataset, "_index", None)
    if sample_index is None:
        raise RuntimeError("Dataset no longer exposes _index; update hard-sampling alignment code.")

    sample_ids: list[str] = []
    for index in indices:
        item = sample_index[int(index)]
        sample_ids.append(f"{item.scene_name}:{int(item.scene_frame_index)}")
    return sample_ids


def read_json(path: Path) -> dict[str, Any]:
    """读取JSON文件并返回解析后的字典。
    
    Args:
        path: JSON文件路径
        
    Returns:
        dict: 解析后的JSON内容字典
        
    Note:
        使用UTF-8编码读取，适用于metadata.json等配置文件
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

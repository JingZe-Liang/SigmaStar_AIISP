"""Independent guidance, detail-biased crops, and reuse of sparse teacher values."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from company_exposure.core import guidance_masks, signed_signal
from company_exposure.data import read_packed

INPUT_OFFSETS = (-2, -1, 0, 1, 2)
GUIDE_OFFSETS = (-5, -3, 3, 5, 7, 9)
TARGET_OFFSET = 6


def temporal_context(stream, role, frame):
    if stream.get('role') != 'development' or role not in ('train', 'validation'):
        raise ValueError('Only development train/validation contexts may be prepared')
    lo, hi = stream['splits'][role]
    inputs = [frame + i for i in INPUT_OFFSETS]
    guides = [frame + i for i in GUIDE_OFFSETS]
    target = frame + TARGET_OFFSET
    if frame % 4 or not all(lo <= i <= hi for i in [*inputs, *guides, target]):
        raise ValueError('Temporal context crosses split or has invalid center')
    assert set(guides).isdisjoint([*inputs, target]) and target not in inputs
    return inputs, guides, target


def independent_masks(observed_guides, stream):
    static, motion, stats = guidance_masks(observed_guides, stream['shot_gain_dn12_per_e'],
        stream['read_variance_dn12_sq'], stream['black_dn12'], stream.get('pooling_variance_factor'))
    average = signed_signal(observed_guides, stream['black_dn12']).mean(0).mean(0)
    smooth = F.avg_pool2d(F.pad(torch.from_numpy(average)[None, None], (1,1,1,1), mode='reflect'), 3, 1)[0, 0].numpy()
    score = np.abs(np.diff(smooth, axis=0, prepend=smooth[:1])) + np.abs(np.diff(smooth, axis=1, prepend=smooth[:, :1]))
    available = score[static[0]]
    threshold = max(1., float(np.quantile(available, .75))) if available.size else float('inf')
    # These are texture candidates, not text recognition or a clean edge map.
    # Crop selection depends on guides only, never on the input/target noise.
    detail = static & (score[None] > threshold)
    stats.update(detail_fraction=float(detail.mean()),detail_threshold_dn12=threshold if np.isfinite(threshold) else None)
    return static, motion, detail, stats


def sample(root, cache, stream, record, rng, size):
    root, cache = Path(root), Path(cache)
    inputs, _, target = temporal_context(stream, record['role'], record['frame'])
    with np.load(cache/record['mask_path'], allow_pickle=False) as masks:
        static, motion, detail = [masks[k].copy() for k in ('static', 'motion', 'detail')]
    height, width = static.shape[1:]
    if size > min(height, width):
        raise ValueError('Crop exceeds stream dimensions')
    choice = rng.random()
    desired = detail if choice < .5 else static if choice < .75 else motion
    if not desired.any():
        desired = static | motion
    sites = np.argwhere(desired[0])
    if not len(sites):
        raise ValueError('No supervised crop sites')
    y, x = sites[int(rng.integers(len(sites)))]
    y = int(np.clip(y-int(rng.integers(size)), 0, height-size))
    x = int(np.clip(x-int(rng.integers(size)), 0, width-size))
    observed = read_packed(root, stream, [*inputs, target], (y, x, size))
    signal = signed_signal(observed, stream['black_dn12'])
    teacher = torch.load(root/record['teacher_path'], map_location='cpu', weights_only=True)
    dense = torch.zeros(teacher['shape'])
    dense[:, teacher['motion'][0]] = teacher['motion_signal_dn12'].T
    return dict(signal=torch.from_numpy(signal[:5]), target=torch.from_numpy(signal[5]),
                teacher=dense[:, y:y+size, x:x+size],
                static=torch.from_numpy(static[:, y:y+size, x:x+size]),
                motion=torch.from_numpy(motion[:, y:y+size, x:x+size]),
                span=torch.tensor(4095-np.asarray(stream['black_dn12']), dtype=torch.float32),
                shot=torch.tensor(stream['shot_gain_dn12_per_e'], dtype=torch.float32))

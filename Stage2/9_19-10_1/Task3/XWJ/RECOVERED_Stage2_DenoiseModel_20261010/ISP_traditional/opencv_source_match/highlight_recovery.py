import numpy as np


def reconstruct_camera_highlights(camera_rgb, neutral, start=0.90, end=0.995):
    neutral = np.asarray(neutral[:3], dtype=np.float32)
    if neutral.size != 3 or np.any(neutral <= 0):
        return camera_rgb

    camera = np.clip(camera_rgb, 0.0, None).astype(np.float32, copy=False)
    normalized = camera / neutral.reshape(1, 1, 3)

    t = np.clip((camera - start) / max(end - start, 1e-6), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    available = 1.0 - weight
    available_sum = available.sum(axis=2, keepdims=True)
    estimate = (normalized * available).sum(axis=2, keepdims=True) / np.maximum(
        available_sum, 1e-6
    )
    fallback = normalized.max(axis=2, keepdims=True)
    estimate = np.where(available_sum > 0.15, estimate, fallback)

    recovered = normalized * (1.0 - weight) + estimate * weight
    return recovered * neutral.reshape(1, 1, 3)

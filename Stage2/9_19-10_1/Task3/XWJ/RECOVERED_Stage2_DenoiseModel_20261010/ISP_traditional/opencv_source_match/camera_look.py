import numpy as np


CAMERA_TINT_GAINS = {
    "iPhone18,1 back camera": np.asarray(
        [1.0092089, 1.0010222, 0.9627573], dtype=np.float32
    ),
    "iPhone18,1 back telephoto camera": np.asarray(
        [0.9917224, 1.0089744, 0.9354746], dtype=np.float32
    ),
}


def apply_camera_tint(linear_srgb, camera_model):
    gain = CAMERA_TINT_GAINS.get(camera_model)
    if gain is None:
        return linear_srgb
    return linear_srgb * gain.reshape(1, 1, 3)

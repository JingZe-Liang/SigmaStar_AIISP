import numpy as np


D50_XY = np.asarray([0.34567, 0.35850], dtype=np.float64)

XYZ_D50_TO_D65 = np.asarray(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float64,
)

XYZ_D65_TO_SRGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

BRADFORD = np.asarray(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)

# EXIF LightSource values used by DNG CalibrationIlluminant tags.
ILLUMINANT_CCT = {
    17: 2856.0,
    18: 4874.0,
    19: 6774.0,
    20: 5503.0,
    21: 6504.0,
    22: 7504.0,
    23: 5003.0,
    24: 3200.0,
}


def _matrix(values, shape=(3, 3), default=None):
    values = np.asarray(values or [], dtype=np.float64)
    if values.size != shape[0] * shape[1]:
        return np.eye(shape[0], shape[1], dtype=np.float64) if default is None else default
    return values.reshape(shape)


def xy_to_xyz(xy):
    x, y = np.asarray(xy, dtype=np.float64)
    y = max(float(y), 1e-8)
    return np.asarray([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def xyz_to_xy(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    total = float(np.sum(xyz))
    if not np.isfinite(total) or total <= 1e-8:
        return D50_XY.copy()
    return xyz[:2] / total


def xy_to_cct(xy):
    x, y = np.asarray(xy, dtype=np.float64)
    denom = y - 0.1858
    if abs(denom) < 1e-8:
        return 5000.0
    n = (x - 0.3320) / denom
    cct = -449.0 * n**3 + 3525.0 * n**2 - 6823.3 * n + 5520.33
    return float(np.clip(cct, 2000.0, 50000.0))


def reciprocal_temperature_weight(cct, cct1, cct2):
    inv1 = 1.0 / float(cct1)
    inv2 = 1.0 / float(cct2)
    denom = inv2 - inv1
    if abs(denom) < 1e-12:
        return 1.0
    return float(np.clip((1.0 / float(cct) - inv1) / denom, 0.0, 1.0))


def chromatic_adaptation(src_xy, dst_xy=D50_XY):
    src_lms = BRADFORD @ xy_to_xyz(src_xy)
    dst_lms = BRADFORD @ xy_to_xyz(dst_xy)
    scale = np.diag(dst_lms / np.maximum(src_lms, 1e-8))
    return np.linalg.inv(BRADFORD) @ scale @ BRADFORD


def resolve_dng_transform(meta):
    cm1 = _matrix(meta.get("color_matrix_1"), default=None)
    cm2_values = meta.get("color_matrix_2") or []
    cm2 = _matrix(cm2_values, default=cm1)
    cc1 = _matrix(meta.get("camera_calibration_1"), default=np.eye(3))
    cc2 = _matrix(meta.get("camera_calibration_2"), default=cc1)

    analog = np.asarray(meta.get("analog_balance") or [1.0, 1.0, 1.0], dtype=np.float64)
    if analog.size != 3 or np.any(analog <= 0):
        analog = np.ones(3, dtype=np.float64)
    ab = np.diag(analog)

    neutral = np.asarray(meta.get("neutral") or [1.0, 1.0, 1.0], dtype=np.float64)
    if neutral.size != 3 or np.any(neutral <= 0):
        neutral = np.ones(3, dtype=np.float64)

    illum1 = int(meta.get("calibration_illuminant_1") or 17)
    illum2 = int(meta.get("calibration_illuminant_2") or illum1)
    cct1 = ILLUMINANT_CCT.get(illum1, 2856.0)
    cct2 = ILLUMINANT_CCT.get(illum2, cct1)
    has_dual = len(cm2_values) == 9 and abs(cct2 - cct1) > 1e-6

    weight = 1.0 if has_dual else 0.0
    white_xy = D50_XY.copy()
    for _ in range(12):
        cm = (1.0 - weight) * cm1 + weight * cm2
        cc = (1.0 - weight) * cc1 + weight * cc2
        xyz_to_camera = ab @ cc @ cm
        white_xy = xyz_to_xy(np.linalg.pinv(xyz_to_camera) @ neutral)
        if not has_dual:
            break
        new_weight = reciprocal_temperature_weight(xy_to_cct(white_xy), cct1, cct2)
        if abs(new_weight - weight) < 1e-7:
            weight = new_weight
            break
        weight = new_weight

    cm = (1.0 - weight) * cm1 + weight * cm2
    cc = (1.0 - weight) * cc1 + weight * cc2
    xyz_to_camera = ab @ cc @ cm
    camera_to_xyz = np.linalg.pinv(xyz_to_camera)
    camera_to_xyz_d50 = chromatic_adaptation(white_xy, D50_XY) @ camera_to_xyz
    camera_to_srgb = XYZ_D65_TO_SRGB @ XYZ_D50_TO_D65 @ camera_to_xyz_d50
    return camera_to_srgb.astype(np.float32), white_xy.astype(np.float32), weight


def camera_to_linear_srgb(camera_rgb, meta):
    matrix, white_xy, weight = resolve_dng_transform(meta)
    linear = camera_rgb @ matrix.T
    return np.clip(linear, 0.0, None), white_xy, weight


def apply_exposure(linear_rgb, ev):
    return linear_rgb * np.float32(2.0 ** float(ev))

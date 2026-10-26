import json
from pathlib import Path

import numpy as np

from compare_company_rvi import ROOT, write_json
from company_exposure.core import sha256
from company_exposure.data import read_packed

LOCAL = Path('E:/Dataset/SigmaStar/Sigmastar_7_30')
OUTPUT = ROOT / 'work_v3/holdout10_5000_20261024'


def bins(stream, start, pairs, black):
    count = np.zeros((4, 128), dtype=np.int64)
    variance = np.zeros((4, 128), dtype=np.float64)
    for pair in range(start, start + pairs):
        values = read_packed(ROOT, stream, [2 * pair, 2 * pair + 1])[:, :, ::4, ::4]
        mean = values.mean(0) - black[:, None, None]
        delta = values[0] - values[1]
        delta -= np.median(delta, axis=(1, 2), keepdims=True)
        for channel in range(4):
            valid = (mean[channel] >= 64) & (mean[channel] < 3000)
            indices = (mean[channel][valid] / 32).astype(int)
            count[channel] += np.bincount(indices, minlength=128)
            variance[channel] += np.bincount(indices, weights=.5 * delta[channel][valid] ** 2, minlength=128)
    return count, np.divide(variance, count, out=np.zeros_like(variance), where=count > 0)


def main():
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT / 'selection.json').exists():
        raise RuntimeError('Existing selection protected')
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    target = next(dict(stream) for stream in manifest['streams'] if stream['id'] == 'mis20s1_114900bf5559')
    calibration = [dict(stream) for stream in manifest['streams'] if stream['role'] == 'calibration'
                   and '/12800/' in stream['source_relative']]
    if len(calibration) != 2 or target['role'] != 'test':
        raise RuntimeError('Expected independent color/dark calibration and held-out target')
    for stream in [target, *calibration]:
        source = LOCAL / stream['source_relative'].split('/', 1)[1]
        if source.stat().st_size != stream['bytes'] or sha256(source) != stream['sha256']:
            raise RuntimeError('Source byte/hash mismatch')
        stream['path'] = str(source)
    dark = next(stream for stream in calibration if '/black/' in stream['source_relative'])
    color = next(stream for stream in calibration if '/color/' in stream['source_relative'])
    analog = target['metadata']['sensor_gain_code']
    if any(stream['metadata']['sensor_gain_code'] != analog for stream in calibration):
        raise RuntimeError('Analog gain mismatch')
    digital_ratio = color['metadata']['isp_gain_code'] / target['metadata']['isp_gain_code']
    if dark['metadata']['isp_gain_code'] != target['metadata']['isp_gain_code']:
        raise RuntimeError('Independent dark digital gain mismatch')
    black = np.asarray(target['black_dn12'], dtype=np.float32)
    dark_variances = []
    for pair in range(20):
        values = read_packed(ROOT, dark, [2 * pair, 2 * pair + 1])[:, :, ::4, ::4]
        delta = values[0] - values[1]
        delta -= np.median(delta, axis=(1, 2), keepdims=True)
        dark_variances.append((.5 * delta ** 2).mean(axis=(1, 2)))
    read_variance = np.median(dark_variances, axis=0)
    fit_count, fit_var = bins(color, 0, 15, black)
    check_count, check_var = bins(color, 20, 5, black)
    centers = (np.arange(128) + .5) * 32
    channels = []
    slopes = []
    for channel in range(4):
        use = (fit_count[channel] >= 100) & (centers >= 64) & (centers < 3000) & (fit_var[channel] > 0)
        if use.sum() < 12:
            raise RuntimeError('Insufficient independent calibration brightness coverage')
        weights = fit_count[channel, use] / np.maximum(fit_var[channel, use] ** 2, 1)
        slope, intercept = np.polyfit(centers[use], fit_var[channel, use], 1, w=np.sqrt(weights / weights.max()))
        check = (check_count[channel] >= 30) & (centers >= 64) & (centers < 3000) & (check_var[channel] > 0)
        prediction = slope * centers[check] + intercept
        error = float(np.median(np.abs(prediction - check_var[channel, check]) / check_var[channel, check]))
        if not np.isfinite(slope + intercept + error) or slope <= 0 or check.sum() < 12 or error > .25:
            raise RuntimeError('Independent calibration validation gate failed')
        slopes.append(float(slope / digital_ratio))
        channels.append(dict(channel=channel, fitted_color_slope=float(slope), free_intercept=float(intercept),
                             fitted_bins=int(use.sum()), validation_bins=int(check.sum()),
                             independent_validation_median_relative_error=error))
    conditions = dict(shot_gain_dn12_per_e=slopes, read_variance_dn12_sq=read_variance.tolist(),
                      black_dn12=target['black_dn12'], channels=channels, digital_ratio=digital_ratio,
                      sources=[dict(id=stream['id'], path=stream['path'], sha256=stream['sha256'],
                                    role=stream['role'], metadata=stream['metadata']) for stream in calibration],
                      method='independent calibration only; temporal pair photon-transfer weighted free-intercept fit; slope rescaled by digital-gain ratio; dark pair variance',
                      fit_color_frames=list(range(30)), validation_color_frames=list(range(40, 50)),
                      dark_frames=list(range(40)), heldout_images_used_for_fit=False,
                      status='approximate_operational_conditions_not_factory_calibration',
                      limitations=['Dark exposure 83317 us differs from test exposure 54992 us; exposure-dependent dark current is not separated',
                                   'Color digital gain 1071 rescaled to target 1024 assuming linear gain and unchanged black pedestal',
                                   'Marginal scalar-channel variance omits fixed-pattern, row/column and temporal correlations',
                                   'Original training conditions and manifest remain unchanged; these conditions are evaluation-only'])
    write_json(OUTPUT / 'independent_conditions.json', conditions)
    frames = list(range(8, 189, 20))
    if len(frames) != 10 or frames[-1] + 4 >= target['frame_count']:
        raise RuntimeError('Invalid fixed ten-frame selection')
    write_json(OUTPUT / 'selection.json', dict(stream=target, frames=frames,
               selection_reason='128x independent calibration has matching analog gain and matching dark digital gain; fixed uniformly spaced frames, not image-quality selection',
               final_checkpoint=str(ROOT / 'work_v3/runs/residual_v3_001/final.pt'), required_checkpoint_step=5000,
               conditions_sha256=sha256(OUTPUT / 'independent_conditions.json'),
               panels=['Noisy center', 'Official RViDeformer-L CRVD / 8 frames', 'V3 final5000 / 5 frames'],
               training_or_selection_on_heldout=False))
    print(json.dumps(dict(output=str(OUTPUT), frames=frames, shot=slopes, read_variance=read_variance.tolist(),
                          validation_errors=[channel['independent_validation_median_relative_error'] for channel in channels]), indent=2))


if __name__ == '__main__':
    main()

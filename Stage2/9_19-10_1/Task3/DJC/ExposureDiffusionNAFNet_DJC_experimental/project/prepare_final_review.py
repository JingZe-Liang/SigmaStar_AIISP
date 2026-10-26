import json
from pathlib import Path

import numpy as np

from compare_company_rvi import ROOT, write_json
from company_exposure.burst_detail import pilot
from company_exposure.core import sha256
from company_exposure.data import read_packed
from prepare_holdout10 import bins

OUTPUT = ROOT / 'work_v3/final_review_5000_20261024_v2'
LOCAL = Path('E:/Dataset/SigmaStar/Sigmastar_7_30')


def main():
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT / 'selection.json').exists():
        raise RuntimeError('Frozen review selection protected')
    manifest, development, labels, groups = pilot.inventory(ROOT)
    gain_evidence = [dict(stream=stream['id'], analog_gain=stream['metadata']['sensor_gain_code'],
                         isp_gain=stream['metadata']['isp_gain_code'], shot=stream['shot_gain_dn12_per_e'])
                     for stream in development.values() if stream['sensor'] == 'MIS20S1']
    measured = np.asarray([row['shot'] for row in gain_evidence])
    isp_ratio = max(row['isp_gain'] for row in gain_evidence) / min(row['isp_gain'] for row in gain_evidence)
    shot_ratios = measured.max(axis=0) / measured.min(axis=0)
    if len({row['analog_gain'] for row in gain_evidence}) != 1 or isp_ratio < 3 or shot_ratios.max() > 1.5:
        raise RuntimeError('Development evidence does not support omitting extra ISP digital scaling')
    profiles = []
    for analog in [8192, 16384, 65536, 131072]:
        entries = [dict(stream) for stream in manifest['streams'] if stream['role'] == 'calibration'
                   and stream['metadata']['sensor_gain_code'] == analog]
        if len(entries) != 2:
            raise RuntimeError('Independent calibration pair missing')
        for stream in entries:
            source = LOCAL / stream['source_relative'].split('/', 1)[1]
            if source.stat().st_size != stream['bytes'] or sha256(source) != stream['sha256']:
                raise RuntimeError('Calibration source checksum failed')
            stream['path'] = str(source)
        dark = next(stream for stream in entries if '/black/' in stream['source_relative'])
        color = next(stream for stream in entries if '/color/' in stream['source_relative'])
        if dark['metadata']['isp_gain_code'] != 1024:
            raise RuntimeError('Unexpected dark digital gain')
        black = np.full(4, 256.5, dtype=np.float32)
        variances = []
        for pair in range(20):
            value = read_packed(ROOT, dark, [2 * pair, 2 * pair + 1])[:, :, ::4, ::4]
            delta = value[0] - value[1]
            delta -= np.median(delta, axis=(1, 2), keepdims=True)
            variances.append((.5 * delta ** 2).mean(axis=(1, 2)))
        fit_count, fit_variance = bins(color, 0, 15, black)
        check_count, check_variance = bins(color, 20, 5, black)
        centers = (np.arange(128) + .5) * 32
        slopes, diagnostics = [], []
        digital = 1.
        for channel in range(4):
            use = (fit_count[channel] >= 100) & (centers >= 64) & (centers < 3000) & (fit_variance[channel] > 0)
            if use.sum() < 12:
                raise RuntimeError('Calibration brightness coverage insufficient')
            weights = fit_count[channel, use] / np.maximum(fit_variance[channel, use] ** 2, 1)
            slope, intercept = np.polyfit(centers[use], fit_variance[channel, use], 1, w=np.sqrt(weights / weights.max()))
            check = (check_count[channel] >= 30) & (centers >= 64) & (centers < 3000) & (check_variance[channel] > 0)
            error = float(np.median(np.abs(slope * centers[check] + intercept - check_variance[channel, check]) / check_variance[channel, check]))
            if not np.isfinite(slope + intercept + error) or slope <= 0 or check.sum() < 12 or error > .25:
                raise RuntimeError('Independent calibration validation failed')
            slopes.append(float(slope / digital))
            diagnostics.append(dict(channel=channel, validation_median_relative_error=error,
                                    free_intercept=float(intercept), bins=int(use.sum())))
        profiles.append(dict(analog_gain=analog, shot_at_digital1=slopes,
                             read_variance_at_digital1=np.median(variances, axis=0).tolist(),
                             diagnostics=diagnostics, sources=[dict(id=stream['id'], path=stream['path'],
                             sha256=stream['sha256'], role='calibration', metadata=stream['metadata']) for stream in entries]))
        print(json.dumps(dict(calibrated_analog_gain=analog, slopes=slopes)), flush=True)
    write_json(OUTPUT / 'independent_noise_profiles.json', dict(profiles=profiles,
               development_gain_policy_evidence=gain_evidence, isp_metadata_ratio=isp_ratio,
               measured_raw_shot_ratios=shot_ratios.tolist(),
               gain_policy='Do not treat ISP-gain filename codes as RAW multiplication; working assumption supported only by development noise fits and independent analog calibration, not a claim of factory hardware identification',
               fit_color_frames=list(range(30)), check_color_frames=list(range(40, 50)), dark_frames=list(range(40)),
               heldout_images_used_for_fit=False, training_conditions_modified=False,
               method='independent frame-pair photon-transfer fit; source RawSpec/read_frames; per-channel free intercept',
               limitations=['Operational approximation, not factory-calibrated sensor likelihood',
                            'Dark exposure 83317 us differs from test exposures; dark current is not separated',
                            'ISP digital-gain metadata is not applied again to RAW noise conditions; development fits support near-constant RAW shot slope across that metadata range',
                            'Intermediate analog gains use linear shot-gain and analog-gain-squared read-variance interpolation',
                            'Fixed-pattern and temporal correlations omitted']))
    scenes = []
    for key, stream in sorted(development.items()):
        lower, upper = stream['splits']['validation']
        frames = [frame for frame in range(lower + 3, upper - 3) if frame % 4 == 0]
        selected = [frames[0], frames[len(frames) // 2], frames[-1]]
        if len(set(selected)) != 3:
            raise RuntimeError('Three distinct legal validation contexts required')
        scenes.append(dict(stream=stream, role='development_validation', frames=selected,
                           conditions=dict(shot_gain_dn12_per_e=stream['shot_gain_dn12_per_e'],
                                           read_variance_dn12_sq=stream['read_variance_dn12_sq'], basis='unchanged training manifest')))
    analogs = np.asarray([profile['analog_gain'] for profile in profiles], dtype=np.float64)
    for original in sorted([stream for stream in manifest['streams'] if stream['role'] == 'test'], key=lambda stream: stream['source_relative']):
        stream = dict(original)
        source = LOCAL / stream['source_relative'].split('/', 1)[1]
        if sha256(source) != stream['sha256']:
            raise RuntimeError('Heldout source identity failed')
        stream['path'] = str(source)
        analog = stream['metadata']['sensor_gain_code']
        digital = 1.
        if not analogs[0] <= analog <= analogs[-1]:
            raise RuntimeError('Do not extrapolate analog noise conditions')
        shot = [float(np.interp(analog, analogs, [profile['shot_at_digital1'][channel] for profile in profiles]) * digital) for channel in range(4)]
        read = [float(np.interp(analog ** 2, analogs ** 2, [profile['read_variance_at_digital1'][channel] for profile in profiles]) * digital ** 2) for channel in range(4)]
        selected = [int(frame) for frame in np.linspace(8, stream['frame_count'] - 9, 15)]
        if len(set(selected)) != 15:
            raise RuntimeError('Fifteen distinct holdout frames required')
        scenes.append(dict(stream=stream, role='heldout_test', frames=selected,
                           conditions=dict(shot_gain_dn12_per_e=shot, read_variance_dn12_sq=read,
                                           basis='independent analog-gain transfer, no extra ISP-digital scaling; development-supported working assumption, no test-image fit')))
    if len(scenes) != 27 or sum(len(scene['frames']) for scene in scenes) != 165:
        raise RuntimeError('Expected 20x3 plus 7x15 review cases')
    write_json(OUTPUT / 'selection.json', dict(scenes=scenes, cases=165, development_frames=60, heldout_frames=105,
               frame_index_base=0, required_checkpoint_step=5000, use_final_not_test_selected_best=True,
               selection='each development validation interval first/middle/last legal full-frame RVi center, no sparse-label restriction; each holdout uniformly spaced interior frames; fixed before final inference',
               noise_profiles_sha256=sha256(OUTPUT / 'independent_noise_profiles.json'),
               request='每个未参与训练的场景15帧，其余各3帧；Noisy/RVi/冻结主干/加分支；通俗报告；不再询问'))
    print(json.dumps(dict(scenes=27, cases=165, output=str(OUTPUT))), flush=True)


if __name__ == '__main__':
    main()

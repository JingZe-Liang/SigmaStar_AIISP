"""Audited joint camera likelihood. Never assume seven independent observations.

Separable temporal/spatial Gaussian approximation with signal-dependent variance.
ADC approximation and covariance separability MUST pass real-data checks externally.
This is a project observation adapter, not an EMDiffusion paper reproduction.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.diffusion_gates import sha256


class SevenFrameCameraLikelihood(nn.Module):
    required_checks = ("raw_dn12", "training_data_only", "mean_bias", "signal_variance",
        "spatial_covariance", "temporal_covariance", "covariance_separability",
        "quantization_approximation", "dc_and_fixed_pattern", "static_roi_coverage")

    def __init__(self, archive, audit, *, diagnostic_only=False):
        super().__init__()
        self.archive, self.audit_path = str(Path(archive).resolve()), str(Path(audit).resolve())
        report = json.loads(Path(audit).read_text())
        if report.get("model_sha256") != sha256(archive) or (not report.get("acceptance_passed") and not diagnostic_only):
            raise ValueError("STOP: joint camera model has no matching passed real-data audit")
        if not diagnostic_only and any(report.get("checks", {}).get(key) is not True for key in self.required_checks):
            raise ValueError("STOP: missing joint noise/ADC/DC/data checks")
        if report.get("profile") not in ("mis20s1_outdoor_128x", "mis20s1_outdoor_512x",
                                        "mis20s1_outdoor_1000x", "mis20s1_outdoor_1500x"):
            raise ValueError("held-out/unsupported company profile cannot train this branch")
        self.diagnostic_only = diagnostic_only
        self.profile = report["profile"]
        self.report = report
        with np.load(archive, allow_pickle=False) as values:
            for key in ("gain_dn", "read_variance_dn2", "black_dn", "temporal_covariance", "spatial_power"):
                value = torch.from_numpy(values[key].astype(np.float32))
                if not torch.isfinite(value).all():
                    raise ValueError(f"nonfinite camera coefficient: {key}")
                self.register_buffer(key, value)
            fpn = values["fixed_pattern_variance_dn2"] if "fixed_pattern_variance_dn2" in values else np.zeros(4)
            self.register_buffer("fixed_pattern_variance_dn2", torch.as_tensor(fpn, dtype=torch.float32))
            reference = values["variance_reference_signal_dn"] if "variance_reference_signal_dn" in values else np.empty(0)
            self.register_buffer("variance_reference_signal_dn", torch.as_tensor(reference, dtype=torch.float32))
        if self.fixed_pattern_variance_dn2.shape != (4,) or not torch.isfinite(self.fixed_pattern_variance_dn2).all() or (self.fixed_pattern_variance_dn2 < 0).any():
            raise ValueError("invalid independent fixed-pattern variance")
        if self.gain_dn.shape != (4,) or self.read_variance_dn2.shape != (4,) or self.black_dn.shape != (4,):
            raise ValueError("camera gain/variance/black must have four canonical Bayer channels")
        if (self.gain_dn < 0).any() or (self.read_variance_dn2 <= 0).any():
            raise ValueError("invalid noise variance")
        if self.temporal_covariance.shape != (4, 7, 7):
            raise ValueError("requires a seven-frame covariance for each Bayer channel")
        torch.testing.assert_close(self.temporal_covariance, self.temporal_covariance.transpose(-1, -2))
        torch.testing.assert_close(self.temporal_covariance.diagonal(dim1=-2, dim2=-1), torch.ones(4, 7))
        cholesky = torch.linalg.cholesky(self.temporal_covariance)
        self.register_buffer("precision", torch.cholesky_inverse(cholesky))
        self.register_buffer("temporal_logdet", 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        if self.spatial_power.ndim != 3 or self.spatial_power.shape[0] != 4 or (self.spatial_power <= 0).any():
            raise ValueError("requires positive four-channel spatial spectral power, including DC")
        if self.variance_reference_signal_dn.numel():
            if self.variance_reference_signal_dn.shape != self.spatial_power.shape or not torch.isfinite(self.variance_reference_signal_dn).all():
                raise ValueError("invalid independently calibrated variance reference")
        elif not diagnostic_only and self.gain_dn.any():
            raise ValueError("signal-dependent DPS needs independent variance calibration; noisy inverse weights are biased")
        torch.testing.assert_close(self.spatial_power.mean((-2, -1)), torch.ones(4), atol=.01, rtol=.01)
        # A real stationary Gaussian field must have conjugate-symmetric spectrum.
        h, w = self.spatial_power.shape[-2:]
        mirror = self.spatial_power[:, (-torch.arange(h)) % h][:, :, (-torch.arange(w)) % w]
        torch.testing.assert_close(self.spatial_power, mirror, atol=1e-5, rtol=1e-4)

    def log_prob(self, observation_dn, signal_dn):
        if self.fixed_pattern_variance_dn2.any():
            raise NotImplementedError("FPN marginal logdet is not implemented; use the DPS covariance distance")
        if observation_dn.shape != signal_dn.shape or signal_dn.ndim != 5 or signal_dn.shape[1:3] != (7, 4):
            raise ValueError("joint likelihood expects matching [B,7,4,H,W] DN12")
        if not torch.isfinite(observation_dn).all() or not torch.isfinite(signal_dn).all():
            raise ValueError("nonfinite observation/signal")
        # CPU likelihood supports FFT + covariance autograd on MPS without a silent
        # device fallback; transfers remain differentiable back to diffusion state.
        signal = signal_dn.to(self.gain_dn.device)
        observed = observation_dn.to(signal.device)
        h, w = signal.shape[-2:]
        if (h, w) != tuple(self.spatial_power.shape[-2:]):
            raise ValueError("STOP: unvalidated spatial resolution for joint camera spectrum")
        variance = self.gain_dn[None, None, :, None, None] * signal.clamp_min(0)
        variance = variance + self.read_variance_dn2[None, None, :, None, None]
        residual = (observed - self.black_dn[None, None, :, None, None] - signal) / variance.sqrt()
        # Do NOT subtract residual means: the DC/color error must stay constrained.
        spectrum = torch.fft.fft2(residual, norm="ortho")
        quadratic = torch.einsum("btcij,ctu,bucij->bcij", spectrum.conj(),
                                self.precision.to(spectrum.dtype), spectrum).real
        quadratic = (quadratic / self.spatial_power).sum((1, 2, 3))
        logdet = variance.log().sum((1, 2, 3, 4))
        logdet = logdet + h * w * self.temporal_logdet.sum() + 7 * self.spatial_power.log().sum()
        result = -.5 * (quadratic + logdet + signal[0].numel() * np.log(2 * np.pi))
        return result.to(signal_dn.device)


    @torch.no_grad()
    def conditional_clean_mean(self, observation_dn, mean_dn, prior_variance_dn2, prior_dc_variance_dn2=None):
        """Type-II Gaussian conditioning with PGDM scalar prior covariance.

        ICML 2024 optimal-posterior-covariance official condition.py, Type II.
        We adapt its identity observation solve to our fixed joint camera C_y:
        m + r² (C_y + r² I)^-1 (y-black-m). Not learned optimal covariance.
        """
        if not self.variance_reference_signal_dn.numel():
            raise ValueError("Type-II requires independent variance calibration")
        if observation_dn.shape != mean_dn.shape or mean_dn.shape[1:3] != (7,4):
            raise ValueError("requires matching seven-frame DN12 tensors")
        if tuple(mean_dn.shape[-2:]) != tuple(self.spatial_power.shape[-2:]):
            raise ValueError("unvalidated spatial resolution")
        r2 = torch.as_tensor(prior_variance_dn2, device=mean_dn.device)
        if r2.numel() not in (1,4) or not torch.isfinite(r2).all() or (r2<=0).any():
            raise ValueError("positive finite scalar/four-channel prior variance required")
        if r2.numel()==4:r2=r2.view(1,1,4,1,1)
        variance = self.gain_dn[None,None,:,None,None]*self.variance_reference_signal_dn[None,None].clamp_min(0)
        variance = variance+self.read_variance_dn2[None,None,:,None,None]
        sigma=variance.sqrt()
        fpn=self.fixed_pattern_variance_dn2[None,None,:,None,None]
        def apply(value):
            spectrum=torch.fft.fft2(sigma*value,norm="ortho")
            spectrum=torch.einsum("ctu,bucij->btcij",self.temporal_covariance.to(spectrum.dtype),spectrum)
            correlated=torch.fft.ifft2(spectrum*self.spatial_power,norm="ortho").real
            return sigma*correlated+fpn*value.sum(1,keepdim=True)+r2*value
        rhs=observation_dn-self.black_dn[None,None,:,None,None]-mean_dn
        axes=(1,2,3,4); diagonal=variance+fpn+r2
        def solve(right):
            solution=torch.zeros_like(right); residual=right.clone()
            z=residual/diagonal; direction=z.clone(); rz=(residual*z).sum(axes,keepdim=True)
            initial=right.square().sum(axes,keepdim=True).clamp_min(1e-30)
            for _ in range(128):
                applied=apply(direction)
                alpha=rz/(direction*applied).sum(axes,keepdim=True).clamp_min(1e-30)
                solution+=alpha*direction; residual-=alpha*applied
                if (residual.square().sum(axes,keepdim=True)<=initial*1e-10).all():break
                z=residual/diagonal; new_rz=(residual*z).sum(axes,keepdim=True)
                direction=z+(new_rz/rz.clamp_min(1e-30))*direction; rz=new_rz
            relative=((apply(solution)-right).square().sum(axes,keepdim=True)/initial).sqrt().max()
            if not torch.isfinite(relative) or relative>1e-4:
                raise FloatingPointError("Type-II covariance solve failed residual check")
            return solution,float(relative)
        solution,relative=solve(rhs)
        result=mean_dn+r2*solution
        if prior_dc_variance_dn2 is not None:
            dc=torch.as_tensor(prior_dc_variance_dn2,device=mean_dn.device)
            if dc.shape!=(4,) or not torch.isfinite(dc).all() or (dc<0).any():
                raise ValueError('invalid shared channel-DC prior covariance')
            dc=dc.view(1,1,4,1,1)
            inverse_constant,error=solve(torch.ones_like(rhs));relative=max(relative,error)
            # Woodbury: independent white error plus coherent channel offset.
            # Stable conditional mean, without subtracting nearly equal DC sums.
            channel_axes=(1,3,4)
            offset=dc*solution.sum(channel_axes,keepdim=True)/(1+dc*inverse_constant.sum(channel_axes,keepdim=True))
            result=result+(1-r2*inverse_constant)*offset
        return result,relative

    def observation_distance(self, observation_dn, signal_dn):
        """DPS norm with observation-fixed GLS weights, in normalized RAW units.

        Fixing heteroscedastic weights at an independent calibration reference
        avoids both variance inflation and noisy-dependent inverse-weight bias. This is an explicit camera adapter
        of official EMDiffusion's DPS residual norm, not exact posterior sampling.
        """
        signal = signal_dn.to(self.gain_dn.device)
        observed = observation_dn.to(signal.device)
        black = self.black_dn[None, None, :, None, None]
        reference = (self.variance_reference_signal_dn[None,None]
                     if self.variance_reference_signal_dn.numel() else observed-black)
        variance = self.gain_dn[None, None, :, None, None] * reference.clamp_min(0)
        variance = variance.expand_as(observed)
        variance = variance + self.read_variance_dn2[None, None, :, None, None]
        if tuple(signal.shape[-2:]) != tuple(self.spatial_power.shape[-2:]):
            raise ValueError("unvalidated observation resolution")
        residual = (observed-black-signal) / variance.sqrt()
        if self.fixed_pattern_variance_dn2.any():
            # Marginalize an unknown per-pixel sensor bias shared by all frames.
            # Its independently calibrated variance is retained, not subtracted
            # using a possibly misregistered dark map.
            fixed = (self.fixed_pattern_variance_dn2[None,None,:,None,None]/variance).sqrt()
            def covariance_apply(value):
                spectrum = torch.fft.fft2(value, norm="ortho")
                spectrum = torch.einsum("ctu,bucij->btcij", self.temporal_covariance.to(spectrum.dtype), spectrum)
                correlated = torch.fft.ifft2(spectrum*self.spatial_power, norm="ortho").real
                return correlated+fixed*(fixed*value).sum(1,keepdim=True)
            # Covariance is observation-fixed: implicit quadratic gradient is
            # 2*C^-1*r. No differentiation through iterative solver is needed.
            with torch.no_grad():
                solution = torch.zeros_like(residual)
                remainder = residual.detach().clone(); direction = remainder.clone()
                axes=(1,2,3,4)
                rr = remainder.square().sum(axes,keepdim=True)
                initial_rr=rr.clone()
                for _ in range(64):
                    applied=covariance_apply(direction)
                    alpha=rr/(direction*applied).sum(axes,keepdim=True).clamp_min(1e-30)
                    solution+=alpha*direction;remainder-=alpha*applied
                    new_rr=remainder.square().sum(axes,keepdim=True)
                    if (new_rr <= initial_rr*1e-10).all():break
                    direction=remainder+(new_rr/rr.clamp_min(1e-30))*direction;rr=new_rr
                relative=((covariance_apply(solution)-residual.detach()).square().sum(axes)/initial_rr.flatten().clamp_min(1e-30)).sqrt()
                if relative.max()>1e-4:
                    raise FloatingPointError("camera covariance solver failed its residual check")
            energy=((2*residual-residual.detach())*solution).sum((1,2,3,4))
        else:
            spectrum = torch.fft.fft2(residual, norm="ortho")
            energy = torch.einsum("btcij,ctu,bucij->bcij", spectrum.conj(),
                                 self.precision.to(spectrum.dtype), spectrum).real
            energy = (energy/self.spatial_power).sum((1,2,3))
        # The observation operator whitens by the calibrated covariance.
        # Return its Euclidean norm in noise-standard-deviation units; multiplying
        # back by a reference sigma would erase the camera's absolute noise scale.
        return energy.clamp_min(1e-20).sqrt().to(signal_dn.device)


def camera_posterior_sample(model, observed_dn, likelihood, schedule, guidance_scale=1., trajectory=None, progress=None):
    """Official DPS residual-norm update with an audited camera GLS operator.

    EMDiffusoin/guided_diffusion/condition_methods.py: x_next -= scale*grad(norm).
    All seven states share each timestep and are updated; no beta multiplier.
    Caller freezes network weights. Candidates still require real quality review.
    """
    from models.schedule import predict_x0_from_epsilon
    from models.seven_frame_diffusion import ddpm_step
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("E-step requires a frozen network")
    if not np.isfinite(guidance_scale) or guidance_scale <= 0:
        raise ValueError("guidance scale must be finite and positive")
    likelihood = likelihood.to(observed_dn.device)
    state = torch.randn_like(observed_dn)
    for t in range(len(schedule["betas"])-1, -1, -1):
        with torch.enable_grad():
            state = state.detach().requires_grad_(True)
            times = torch.full((state.shape[0],), t, device=state.device, dtype=torch.long)
            epsilon = model(state, times)
            clean = predict_x0_from_epsilon(state, epsilon, times, schedule["alpha_bars"])
            # Same physical-support prediction as ddpm_step; official mean
            # processor also supplies the processed pred_xstart to DPS.
            clean = clean.clamp(-1, 1)
            distance = likelihood.observation_distance(observed_dn, (clean+1)*(4095/2))
            gradient = torch.autograd.grad(distance.sum(), state)[0]
        with torch.no_grad():
            state = ddpm_step(state, epsilon, t, schedule) - guidance_scale*gradient
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"nonfinite posterior trajectory at timestep {t}")
            if progress is not None and (t % 50 == 0 or t == 999):
                progress({"t":t,"distance":float(distance.mean()),"state_rms":float(state.square().mean().sqrt())})
            if trajectory is not None and t in (999,750,500,250,100,15,4,0):
                trajectory.append({"t":t,"x0":clean.detach().cpu(),
                                   "observation_distance":distance.detach().cpu()})
    return state.detach()


@torch.no_grad()
def camera_type_ii_sample(model, observed_dn, likelihood, schedule, trajectory=None, progress=None, prior_covariance=None):
    """Published Type-II clean mean + PGDM covariance, adapted to adjacent DDPM.

    This is a different E-step approximation from EMDiffusion's DPS. Physical
    support projection is part of the conditional denoiser (as in official code),
    and its actual frequency is recorded, never reported as unchanged clean GT.
    """
    from models.schedule import predict_x0_from_epsilon
    from models.seven_frame_diffusion import ddpm_step
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("E-step requires frozen weights")
    likelihood=likelihood.to(observed_dn.device); state=torch.randn_like(observed_dn)
    for t in range(len(schedule['betas'])-1,-1,-1):
        times=torch.full((state.shape[0],),t,device=state.device,dtype=torch.long)
        epsilon=model(state,times)
        mean=predict_x0_from_epsilon(state,epsilon,times,schedule['alpha_bars']).clamp(-1,1)
        bar=schedule['alpha_bars'][t]
        # VE sigma²=(1-bar)/bar; PGDM r²=sigma²/(1+sigma²)=1-bar.
        prior_variance=(1-bar)
        prior_dc=None
        if prior_covariance is not None and ((1-bar)/bar).sqrt()<.2:
            rows=prior_covariance['rows']
            prior_variance=bar.new_tensor(np.interp(t,[r['t'] for r in rows],[r['mse_normalized'] for r in rows])).clamp_min(1e-6)
            if 'residual_mse_normalized_per_channel' in rows[0]:
                prior_variance=bar.new_tensor([np.interp(t,[r['t'] for r in rows],[r['residual_mse_normalized_per_channel'][c] for r in rows]) for c in range(4)])
            if 'dc_error_variance_normalized_per_channel' in rows[0]:
                prior_dc=bar.new_tensor([np.interp(t,[r['t'] for r in rows],[r['dc_error_variance_normalized_per_channel'][c] for r in rows]) for c in range(4)])*(4095/2)**2
        clean_dn, residual=likelihood.conditional_clean_mean(observed_dn,(mean+1)*(4095/2),prior_variance*(4095/2)**2,prior_dc)
        unbounded=clean_dn/(4095/2)-1
        projection=float(((unbounded < -1)|(unbounded > 1)).float().mean())
        clean=unbounded.clamp(-1,1)
        conditioned_epsilon=(state-bar.sqrt()*clean)/(1-bar).sqrt()
        state=ddpm_step(state,conditioned_epsilon,t,schedule)
        if not torch.isfinite(state).all():raise FloatingPointError(f"nonfinite Type-II state at {t}")
        row=dict(t=t,state_rms=float(state.square().mean().sqrt()),solver_relative_residual=residual,support_projection_ratio=projection,prior_variance_normalized=prior_variance.tolist(),prior_dc_variance_dn2=prior_dc.tolist() if prior_dc is not None else None)
        if progress is not None and (t%50==0 or t==999):progress(row)
        if trajectory is not None and t in (999,750,500,250,100,15,4,0):
            trajectory.append(dict(**row,x0=clean.cpu(),unconditional_x0=mean.cpu()))
    return state.detach()

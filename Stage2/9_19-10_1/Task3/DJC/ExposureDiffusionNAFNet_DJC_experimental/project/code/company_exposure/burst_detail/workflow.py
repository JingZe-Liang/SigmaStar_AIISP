"""CPU proof, strict branch loading and center-reference training loss interface."""
import hashlib
import json
from pathlib import Path

import torch

from company_exposure.core import sha256, mixed_loss
from company_exposure.data import write_json
from company_exposure.workflow import code_signature
from company_adaptation_v2 import candidate_loss, restoration_loss
from .model import ARCHITECTURE, BASE_SHA, load_model
from .references import ROOT, SOURCES


def signature():
    files = (sorted(Path(__file__).parent.glob('*.py')) + [ROOT/'company_adaptation_v2.py']
             + [ROOT/p for p, _ in SOURCES.values()])
    return hashlib.sha256((code_signature() + ''.join(sha256(p) for p in files)).encode()).hexdigest()


def config(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg['architecture'] != ARCHITECTURE or cfg['base_sha256'] != BASE_SHA
            or cfg['precision'] != 'float32' or not cfg['freeze_backbone']
            or cfg['input_offsets'] != [-2,-1,0,1,2] or cfg['target_offset'] != 0
            or cfg['teacher_window_offsets'] != list(range(-4,5)) or cfg['exposure_steps'] != 3
            or cfg['width'] != 16 or cfg.get('injection') != 'raw_output_residual_only'
            or cfg.get('loss_steps') != [3]
            or cfg['patch_packed'] != 192 or not cfg['read_variance_required']
            or cfg['automatic_training']):
        raise ValueError('Unsupported paper-branch pilot configuration')
    for key in ('batch_size','accumulate','optimizer_steps','validate_every','seed'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError('Positive integer required: '+key)
    if (not 0 < cfg['learning_rate'] <= 1e-3 or cfg['weight_decay'] != 0
            or not 0 < cfg['gradient_clip'] <= 1):
        raise ValueError('Invalid branch-only optimization settings')
    for key in ('base_checkpoint',):
        if Path(cfg[key]).is_absolute() or '..' in Path(cfg[key]).parts:
            raise ValueError('Project-relative configuration path required: '+key)
    return cfg


def cpu_check(checkpoint, cfg, output):
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    model = load_model(checkpoint, cfg['base_sha256'], width=cfg['width']).train()
    base_state = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    signal = torch.randn(1,5,4,32,48)*10+150
    with torch.no_grad():
        torch.manual_seed(81); expected = model(signal,[24],[3838.5],[4],enabled=False)
        torch.manual_seed(81); actual = model(signal,[24],[3838.5],[4])
    equal = all(torch.equal(a,b) for a,b in zip(actual,expected))
    if not equal:
        raise ValueError('Zero branch changes pretrained three-step inference')
    sum(x.square().mean() for x in model(signal,[24],[3838.5],[4])).backward()
    if not all(head.weight.grad is not None and head.weight.grad.abs().sum()>0
               for head in (model.to_residual,)):
        raise ValueError('Residual head has no gradient')
    if (any(p.grad is not None for p in model.backbone.parameters())
            or not all(torch.equal(base_state[k],v) for k,v in model.backbone.state_dict().items())
            or sha256(checkpoint) != cfg['base_sha256']):
        raise ValueError('Frozen company backbone changed')
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    report = dict(passed=True,architecture=ARCHITECTURE,signature=signature(),
        config=cfg,base_sha256=cfg['base_sha256'],base_step=payload.get('step'),
        zero_residual_bitwise_equal=True,final_step_head_gradient=True,
        backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
        branch_parameters=sum(p.numel() for p in model.adapter_parameters()),
        backbone_unchanged=True,optimizer_updates=0,cuda_checked=False,
        real_quality_validated=False)
    write_json(output,report)
    return report


def training_loss(outputs, sample):
    """Final-step pixel loss on an independently checked center reference.

    Upstream center_pseudo_sample() verifies noise evidence BEFORE teacher
    inference. Independent clean-center data may also be supplied with source
    evidence, so the architecture is not tied to one teacher's texture limit.
    This interface refuses legacy t+6 labels. It introduces no
    unverified edge/gradient loss and makes no clean-GT or quality claim.
    Both a single unbatched sample and a consistently batched sample work.
    The existing v2 whole-image review and bounded runner remain mandatory
    for any future actual pilot; this module does not launch that pilot.
    """
    if len(outputs) != 3:
        raise ValueError('Exactly three ExposureDiffusion outputs required')
    clean = sample.get('target_kind') == 'independent_clean_center'
    if clean and (sample.get('target_offset') != 0 or sample.get('signal_unit') != 'signed_dn12'
                  or not sample.get('reference_evidence_sha256')):
        raise ValueError('Independent clean reference requires center geometry, DN12 and source evidence')
    if clean:
        span = torch.as_tensor(sample['span_dn12'], device=outputs[0].device)
        if not torch.isfinite(span).all() or (span<=0).any() or (span>4095).any():
            raise ValueError('Invalid clean-reference DN12 span')
    losses = []
    for output in outputs[-1:]:
        if output.ndim != 4:
            raise ValueError('Expected batched four-channel RAW predictions')
        value = output
        if sample['target'].ndim == 3:
            if len(output) != 1:
                raise ValueError('Single reference requires batch size one')
            value = output[0]
        losses.append(restoration_loss(value,sample['target']/span,sample['valid']) if clean
                      else candidate_loss(value, sample))
    return torch.stack(losses).mean()


def final_mixed_loss(outputs, *args):
    """Apply the existing regional loss only to the adapted final prediction.

    Repeating a tensor reuses the original three-term mean exactly; it performs
    no additional inference and does not supervise the two frozen-only outputs.
    """
    if len(outputs) != 3:
        raise ValueError('Exactly three ExposureDiffusion outputs required')
    return mixed_loss([outputs[-1]] * 3, *args)


def load_trained(base_checkpoint, adapter_checkpoint, *, device='cpu'):
    saved = torch.load(adapter_checkpoint,map_location='cpu',weights_only=True)
    if (saved['architecture'] != ARCHITECTURE or saved['signature'] != signature()
            or saved['base_sha256'] != BASE_SHA or saved['config']['base_sha256'] != BASE_SHA):
        raise ValueError('Adapter/base/source provenance mismatch')
    model = load_model(base_checkpoint,saved['base_sha256'],device=device,width=saved['config']['width'])
    model.load_adapter(saved['adapter'])
    return model.eval()

"""Official pretrained RViDeformer teacher on company RAW, no training or flow."""
import argparse,ast,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from noise_model.raw_io import RawSpec,read_frames,pack_bayer,unpack_bayer,encode_raw16,decode_raw16
from noise_model.profile_observation import ProfileObservationBank
from data.signal_mapping import CompanySceneMapping
from utils.rgb_metrics import packed_to_linear_rgb

VENDOR=ROOT/'vendor/RViDeformer'
OUT=ROOT/'runs/rvideformer_motion_teacher_20261018'

def to_teacher(signal_dn):
    """RGGB canonical DN -> vertical-flipped GBRG official R,Gh,B,Gv order."""
    mosaic=unpack_bayer(signal_dn,'RGGB')[:,::-1,:].copy()
    return pack_bayer(mosaic,'GBRG')[:,[0,2,3,1]]

def from_teacher(value):
    canonical=value[:,[0,3,1,2]]
    mosaic=unpack_bayer(canonical,'GBRG')[:,::-1,:].copy()
    return pack_bayer(mosaic,'RGGB')

def self_check():
    signal=np.arange(2*4*8*8,dtype=np.float32).reshape(2,4,8,8)
    assert np.array_equal(from_teacher(to_teacher(signal)),signal)
    # Compare to the actual official pack function without importing obsolete metric APIs.
    tree=ast.parse((VENDOR/'utils.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pack_gbrg_raw')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<official pack>','exec'),ns)
    flipped=unpack_bayer(signal,'RGGB')[:,::-1,:].copy()
    expected=np.stack([ns['pack_gbrg_raw'](m+240).transpose(2,0,1) for m in flipped])
    assert np.allclose(to_teacher(signal)/(4095-240),expected,atol=1e-7)
    spec=RawSpec(16,16,cfa='RGGB')
    raw=unpack_bayer(signal[:1],'RGGB')[0]
    assert np.array_equal(decode_raw16(encode_raw16(raw,spec),spec),raw)

def load_teacher(checkpoint,device):
    import models
    spec=importlib.util.spec_from_file_location('models.RViDeformer_arch',VENDOR/'models/RViDeformer_arch.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    setattr(models,'RViDeformer_arch',module)
    stored=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=module.RViDeformer(upscale=1,img_size=[8,64,64],window_size=[6,8,8],depths=[4]*7+[2]*4,
        indep_reconsts=[11,12],embed_dims=[84]*7+[108]*4,num_heads=[6]*11,
        spynet_path=None,pa_frames=2,deformable_groups=12)
    model.load_state_dict(stored.state_dict(),strict=True);del stored
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return model.eval().requires_grad_(False).to(device)

def positions(n,tile=64,stride=56):
    return sorted(set(list(range(0,n-tile+1,stride))+[n-tile]))

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frames',nargs='+',type=int,default=[20,60,100,180])
    parser.add_argument('--device',choices=['mps','cpu'],default='mps');parser.add_argument('--self-check-only',action='store_true')
    parser.add_argument('--checkpoint',type=Path,default=VENDOR/'checkpoints/RViDeformer-L_CRVD.pth')
    args=parser.parse_args();self_check()
    if args.self_check_only:print('RAW container, Bayer roundtrip, official channel normalization checks passed.');return
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=='mps':torch.mps.set_per_process_memory_fraction(.55)
    checkpoint=args.checkpoint.resolve()
    weight_sha=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    assert weight_sha=='0870c25a4eede7fd77a8da2131c5d5d357c5e9a94e9bbc7d24651aa07d7f2e37'
    model=load_teacher(checkpoint,device)
    manifest=json.loads((ROOT/'data_manifest/company_scene_profiles.json').read_text());profile='shdarkroom_128x'
    source=(ROOT/manifest['company_root']/manifest['profiles'][profile]['path']).resolve();spec=RawSpec(**manifest['raw_spec'])
    config=yaml.safe_load((ROOT/'configs/r0_recovery_local_20261009.yaml').read_text())
    mapping=CompanySceneMapping.from_config(config['data']['signal_mapping'])
    bank=ProfileObservationBank(config['noise']['calibration_json'],mapping,spatial_model_manifest_json=config['noise']['spatial_model_manifest_json'],device='cpu')
    black=float(manifest['profiles'][profile]['black_level_dn']);span=4095-black
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',30)
    reports=[]
    for frame in args.frames:
        out=OUT/f'frame_{frame:03d}';out.mkdir(exist_ok=True)
        if (out/'report.json').exists():raise ValueError(f'Completed output exists: {out}')
        baseline=ROOT/f'runs/shdarkroom128_N2N_multiframe_20261017/frame_{frame:03d}'
        display=json.loads((baseline/'report.json').read_text())
        raw=read_frames(source,spec,max_frames=8,start_frame=frame-3)
        assert raw.shape==(8,1080,1920) and raw.min()>=0 and raw.max()<=4095
        assert np.array_equal(decode_raw16(encode_raw16(raw[3,:8,:8],spec),spec),raw[3,:8,:8])
        obs=torch.from_numpy(pack_bayer(raw,'RGGB'))
        signal=bank.observation_to_signal(obs,0,0,(profile,)*8).clamp(0,span)
        official=to_teacher(signal.numpy())/span
        assert official.shape==(8,4,540,960) and np.isfinite(official).all()
        assert np.allclose(from_teacher(official)*span,signal.numpy(),atol=.001)
        coords=[(y,x) for y in positions(540) for x in positions(960)]
        result=torch.zeros(1,4,540,960);norm=torch.zeros(1,1,540,960)
        start=time.time();outside=0.;samples=0
        for i,(y,x) in enumerate(coords):
            patch=torch.from_numpy(official[:,:,y:y+64,x:x+64].copy())[None].to(device)
            prediction=model(patch)[0,3:4].float().cpu()
            assert prediction.shape==(1,4,64,64) and torch.isfinite(prediction).all()
            outside+=int(((prediction<0)|(prediction>1)).sum());samples+=prediction.numel()
            # Same 64/8 spatial support as official testing; normalized overlap fusion.
            weight=torch.ones(1,1,64,64);ramp=torch.linspace(0,1,9)[1:]
            if y>0:weight[:,:,:8]*=ramp[None,None,:,None]
            if y+64<540:weight[:,:,-8:]*=ramp.flip(0)[None,None,:,None]
            if x>0:weight[:,:,:,:8]*=ramp[None,None,None,:]
            if x+64<960:weight[:,:,:,-8:]*=ramp.flip(0)[None,None,None,:]
            result[:,:,y:y+64,x:x+64]+=prediction.clamp(0,1)*weight
            norm[:,:,y:y+64,x:x+64]+=weight
            del patch,prediction
            if device.type=='mps':torch.mps.empty_cache()
            if i==0 or (i+1)%10==0 or i+1==len(coords):
                status=dict(frame=frame,tile=i+1,tiles=len(coords),seconds=round(time.time()-start,2),state='inferencing')
                (OUT/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
        assert norm.min()>0
        teacher=from_teacher((result/norm).numpy())*span
        assert teacher.shape==(1,4,540,960) and np.isfinite(teacher).all() and teacher.min()>=0 and teacher.max()<=4095
        value=torch.from_numpy(teacher)/4095
        rgb=packed_to_linear_rgb(value,'GBRG',wb_gains=tuple(display['wb_red_blue']))
        image=Image.fromarray((np.clip(rgb[0].permute(1,2,0).numpy()/display['shared_display_scale'],0,1)*255).astype(np.uint8))
        assert image.size==(1920,1080);image.save(out/'RViDeformer_teacher.png')
        torch.save({'signal_dn12':torch.from_numpy(teacher),'profile':profile,'frame':frame,'input_frames':list(range(frame-3,frame+5)),
            'teacher_sha256':weight_sha,'is_clean_gt':False,'usage':'motion pseudo-GT candidate; no mask or student training in this run'},out/'teacher_signal_dn12.pt')
        report=dict(frame=frame,input_frames=list(range(frame-3,frame+5)),source=str(source),checkpoint=str(checkpoint),checkpoint_sha256=weight_sha,
            model='RViDeformer-L_CRVD',training='official CRVD supervised pretrained, no company training',parameters=sum(p.numel() for p in model.parameters()),
            official_commit='4b3f2744f5d19707e718e735c12f2e531fa64dda',device=str(device),seconds=time.time()-start,
            external_optical_flow=False,pseudo_gt_candidate=True,is_clean_gt=False,student_training_started=False,motion_mask_generated=False,
            black_reference_dn12=black,normalization_span_dn12=span,channel_adapter='lossless RGGB vertical flip to GBRG; official R,Gh,B,Gv; inverse flip for output',
            tiling=dict(packed_size=64,stride=56,tiles=len(coords),batch=1,fusion='normalized overlap, 8-pixel ramps; may differ slightly from sequential official stitch'),
            preclip_outside_ratio=outside/samples,wb_red_blue=display['wb_red_blue'],shared_display_scale=display['shared_display_scale'],
            notes='Official unmerged model, strict complete state load, one device-only zeros_like CUDA removal. Shared existing ISP. Full frame reference, not accepted motion labels.')
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));reports.append(report)
        labels=[('Noisy',baseline/'Noisy.png'),('Company 2DNR',baseline/'Company_2DNR.png'),('Company 3DNR',baseline/'Company_3DNR.png'),('Company-trained N2N',baseline/'N2N.png'),('RViDeformer teacher',out/'RViDeformer_teacher.png')]
        canvas=Image.new('RGB',(5760,2256),'black');draw=ImageDraw.Draw(canvas)
        for i,(label,path) in enumerate(labels):
            x=i%3*1920;y=i//3*1128;im=Image.open(path);assert im.size==(1920,1080)
            canvas.paste(im,(x,y+48));draw.text((x+12,y+6),f'{label} | frame {frame}',font=font,fill='white')
        draw.text((3850,1220),'Teacher output = motion pseudo-GT candidate',font=font,fill='white')
        canvas.save(out/'full_comparison.png')
        print(json.dumps({'completed_frame':frame,'seconds':report['seconds']}),flush=True)
    (OUT/'batch_report.json').write_text(json.dumps({'frames':reports,'self_checks_passed':True},ensure_ascii=False,indent=2))
    (OUT/'status.json').write_text(json.dumps({'state':'completed','frames':args.frames}))
if __name__=='__main__':main()

from torch.nn.init import normal_

from Liang.src.denoise.Data_Adjust.Back_Norm import back_norm
from Liang.src.denoise.Data_Adjust.Before_Processing import befor_processing
from Liang.src.denoise.Data_Adjust.Norm import norm
from Liang.src.denoise.Data_Adjust.Processed import processed
from Liang.src.denoise.MD.MD_MotionAdaptiveGating import MD
from Liang.src.denoise.Pre_Denoise.PreDenoising_MeanFilter import PreDenoising
from Liang.src.denoise.Three_DNR.Three_DNR_TemporalAverage import denoise_3d
from Liang.src.denoise.Two_DNR.Two_DNR_bilateralFilter import denoise_2d
from Liang.src.denoise.Fusion.Fusion_Sigmoid import fusion


class Denoising1:
    def __init__(self):
        pass

    def __call__(self, x,bitdepth):
        return self.forward(x,bitdepth)

    def forward(self, x,bitdepth):
        x,T = befor_processing(x)

        x_norm = norm(x,bitdepth)

        # x_Pre = PreDenoising(x_norm,T)

        x_2d = denoise_2d(x_norm,T)

        map = MD(x_norm,T)

        x_3d = denoise_3d(x_norm,map,T)

        x_fusion = fusion(x_2d,x_3d,map)

        x_mid_back_norm = back_norm(x_2d,bitdepth)

        x_out = processed(x_mid_back_norm,T)

        return x_out







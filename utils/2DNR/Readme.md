算子文件夹下的README.md按此格式填写，尽量人工完成，避免一些很明显的代码错误产生。  
#算子名称：

##1.功能定位：

一句话说明该算子做了什么。  
Eg:该算子基于帧间差分做出运动检测，生成了用于2dnr和3dnr融合所需的Motion map。

##2.输入输出格式：

输入格式：  
  元素类型：（tiff图片、npy数组、Tensors张量？）  
  Shape:（如果是图片，请给出图片规模，1080p、无限制，如果是张量，请给出形状，C*H*W、H*W*C，以此类推）  
  Dtype:float32、uint16？  
  是否需要归一化：  
  是否需要metadata.json：  
  一个简短的说明：Eg:已减blacklevel并归一化的RAW视频帧序列。

  输出格式：
    Shape:  
    Dtype:  
    是否已归一化：  
    Effective bits:  
    Container bits:  
    如果有指标产生，请说明指标的作用。Eg:PSNR越高表明去噪效果越好。

其余额外信息根据具体的算子进行补充，宁滥勿缺。  

##3。方法简述：

用3-8行说明核心逻辑，不要无关背景，给出核心公式。

##4.运行示例：  
至少给出一种运行示例，包括超参设置，dir路径设置等等。  
Eg:  
python D:\pipline\myPipeline\scripts\dng_to_tiff_pipeline.py `
  --input_dir D:\project\new_data\outdoor\scene1\A001_05122130_C013 `  
  --output_dir D:\project\new_output\outdoor\scene1\A001_05122130_C013_tiff  
  --overwrite

##5.参数解析：  
对核心超参给出其作用解释。  
Eg:  
--overwrite 覆盖输出路径中已含有的tiff图片  
--history 背景建模时初始化所用帧数

##6算子特点：  
Eg:  
该2dnr算子有明显的视觉效果。  
该3dnr算子能明显产生拖影问题。

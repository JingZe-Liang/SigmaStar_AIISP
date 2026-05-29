算子名称： Boost_iso

1.功能定位：  
对输入图像的有效信号进行指定倍数的线性增益。  
2.输入输出格式：  
输入格式：  
元素类型：单通道Bayer RAW TIFF序列  
Shape：1080p  
Dtype:float32  
是否需要归一化：不需要  
是否需要metadata.json：需要给出metadata.json文件地址，里面包含black_level字段  

输出格式：  
元素类型：单通道Bayer RAW TIFF序列  
Shape:1080p  
Dtype:float32  
是否已归一化：否  
Effective bits:12bits  
Container bits：16bits  

3.方法简述：  
核心是公式--boosted = (raw - black_level)*scale + black_level  \
其中scale是增益倍数。

4.运行示例：  
```powershell
python D:\pipline\myPipeline\boost_iso_tiff.py `
  --input_dir D:\project\new_data\outdoor\scene1\tiff `
  --output_dir D:\project\new_output\outdoor\scene1\boosted '
  --metadata D:\project\new_data\outdoor\scene1\tiff\metadata.json ‘
  --scale 40
```  
5.参数解析：  
1. `--metadata` 元数据的地址  
2. `--scale` 线性增益倍数

6.算子特点：  
该算子使用了一种极简单明了的有效信号增益逻辑，实际上不能很好的模拟相机内部的增益逻辑
增益后的等效ISO也并非与原ISO呈scale倍数。在较暗场景下效果良好，在极低照度下效果
极差。


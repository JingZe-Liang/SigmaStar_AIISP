一：命名规范
所有人按照以下要求在相应的算子文件夹中上传自己相应的算子。

以2dnr算子举例：
将你的2dnr算子代码以及README.md文件打包进一个新的文件夹中，将该文件夹命名为Method name_负责人名字_方法状态（MOG2_wsj_usable)。
同时该文件夹中的README.md文档应按照相应的2DNR下的Readme.md中的规范进行填写，尽量自己填，避免使用ai后产生过多废话，增加其他同学使用和学习成本。

然后将打包好的Method name_负责人名字_方法状态的文件夹传入相应的2DNR文件夹中。

二：Git commit 规范
每次commit时也应按照如下规范命名
格式限定<type>(<scope>):summary

##Type有如下:
feat: 新增功能、算子、实验脚本
fix: 修复 bug
docs: 修改 README、说明文档
data: 新增或修改小规模样例数据、数据说明
exp: 新增实验结果或实验配置
refactor: 重构代码但不改变功能
test: 新增或修改测试
chore: 环境、路径、格式等杂项

##Scope:写上算子或模块名

##Summary:使用简单的英文短句指明做了什么，最好一次只做一件事

Eg:  feat(2dnr):upload 2dnr baseline
  	 Fix(dng_to_tiff):fix the incorrect rearrangement of Bayer array

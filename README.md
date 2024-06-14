# image-matting
## 1.介绍  
In-context Matting:https://github.com/tiny-smart/in-context-matting  
Robust Video Matting:https://github.com/PeterL1n/RobustVideoMatting  
## 2.准备数据  
数据地址：链接：https://pan.baidu.com/s/1Tbgm2LYRPZ8gFPIM1JFpxA?pwd=apii 
提取码：apii  
将数据集放入项目文件夹中，使其具有以下格式
````
项目
├──── MyData
│    ├──── alpha 
│    ├──── image
│    └──── video
├──── RobustVideoMatting-master
├──── in-context-matting-main
````
## 运行指南
### 运行环境
* python 3.8
* torch 2.3.0
* torchvision 0.15.1
* scikit-image 0.21.0
* diffusers 0.28.2

其他库版本详见In-context Matting与Robust Video Matting的github仓库
### 程序运行
首先下载权重文件到in-context-matting-main下，权重链接：    
链接：https://pan.baidu.com/s/1JktLYB1tL72leWu5wFO46g?pwd=r636 
提取码：r636  

在In-context Matting与Robust Video Matting中都是首先运行**MyEval.py**文件生成output_frames与true_alpha两个文件夹之后再运行**MyEvaluate.py**文件输出评价指标的值。  

如果不想运行模型，可以在下面的链接中下载已经生成的output_frames与true_alpha两个文件夹。  

链接：https://pan.baidu.com/s/1v_1vhe-dHRl2oDQDc5HEJQ?pwd=bneh 
提取码：bneh  

将相应模型的output_frames与true_alpha两个文件夹放在相应模型的文件夹下，直接运行MyEvaluate.py文件即可得到模型的评价指标值。

# yolo_tensorrt_inference
yolov5的TensorRT推理，支持单张推理和多图推理。  

## 代码运行环境
RTX 4070 12GB (40系显卡需要cuda11.8以上)   
TensorRT 8.5.3  
CUDA 11.8   
CUDNN 8.6.0  
OpenCV4.5.5  
配置步骤：https://www.cnblogs.com/Fish0403/p/15781888.html  

## 模型转换
使用Yolov5代码中的export.py导出engine模型，更改batchsize  
```python
parser.add_argument('--batch-size', type=int, default=8, help='batch size')
```
下面是转换好的模型链接  
链接：https://pan.baidu.com/s/1RlGhxdWV4Zo_U5vDdKDQVQ  
提取码：zj6l  

## 代码
yolov5代码：https://github.com/ultralytics/yolov5 （使用的是7.0版本）  
yolov7和v8的差别主要在后处理，相关后处理代码在dnn推理的项目中写过，故此处不再重复。  
yolov5、v7和v8的dnn推理项目：https://github.com/Fish0403/yolo_dnn_inference  


### 效果展示
| 姓名   | 单图推理时间 | 单图显存 | 多图时间 |
|--------|------|------|------|
| dnn   | 8ms   | 2.1GB   |    |
| tensorrt   | 4ms   | 1.3GB   | 16ms   |  
<img src="yolo_trt/results/bus.jpg" alt="yolov5">


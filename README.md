# yolo_tensorrt_inference
yolov5的TensorRT推理，支持单张推理和多图推理。  
## 本机环境配置
RTX 4070 12GB
TensorRT 8.5.3
OpenCV4.5.5
CUDA 11.8 
CUDNN 8.6.0

## 模型转换
使用Yolov5代码中的export导出engine模型，下面是转换好的模型链接
链接：https://pan.baidu.com/s/1mxQNyENiDANNrDi4EqiqJg   
提取码：iqkf  

## 代码
yolov5代码：https://github.com/ultralytics/yolov5 （使用的是7.0版本）  
yolov7和v8的后处理已在dnn推理的项目中写过，故此处不再重复

### 效果展示
dnn推理onnx效果  
<img src="yolo_trt/results/bus.jpg" alt="yolov5" width="500">


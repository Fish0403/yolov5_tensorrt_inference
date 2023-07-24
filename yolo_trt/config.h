#pragma once
#include "cuda_utils.h"
#include "logging.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <NvInfer.h>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv::dnn;
using namespace cv;
using namespace std;
using namespace nvinfer1;

/* --------------------------------------------------------
 * These configs are related to tensorrt model, if these are changed,
 * please re-compile and re-serialize the tensorrt model.
 * --------------------------------------------------------*/

 // For INT8, you need prepare the calibration dataset, please refer to
 // https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5#int8-quantization
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

// These are used to define input/output tensor names,
// you can set them to whatever you want.
const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output0"; //  yolov5 7.0
//const static char* kOutputTensorName = "output"; //  yolov5 6.0

// Detection model and Segmentation model' number of classes
constexpr static int kNumClass = 80; // for coco

// BatchSize refers to the number of samples input during a reasoning process
constexpr static int kBatchSize = 1;

// Yolo's input width and height must by divisible by 32
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

constexpr static int kNumOutputBbox = 25200;



/* --------------------------------------------------------
 * These configs are NOT related to tensorrt model, if these are changed,
 * please re-compile, but no need to re-serialize the tensorrt model.
 * --------------------------------------------------------*/

 // NMS overlapping thresh and final detection confidence thresh
const static float kConfThresh = 0.25f;
const static float kNmsThresh = 0.45f;

const static int kGpuId = 0;
constexpr static int kChannel = 3;

// This struct is used to store information about detected objects, 
// including their class, confidence, and location within the image.
struct Detection
{
    int class_id{ 0 };
    float confidence{ 0.0 };
    cv::Rect box{};
};

// Class names corresponding to kNumClass
vector<string> classes = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush" };
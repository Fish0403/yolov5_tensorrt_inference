#include "config.h"

static Logger gLogger;
IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
IExecutionContext* context = nullptr;
cudaStream_t stream;


void* gpu_buffers[2];
const static int kOutputSize = kNumOutputBbox * (kNumClass + 5);
static float mydata[kBatchSize * kChannel * kInputH * kInputW];
static float output[kBatchSize * kOutputSize];

Mat formatToSquare(const Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

bool readEngine(string& engine_name, int device_id) {
    try {
        // 设置GPU
        cudaSetDevice(device_id);

        // 从本地读取engine模型文件
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << engine_name << " error!" << std::endl;
            assert(false);
        }
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        char* serialized_engine = new char[size];
        assert(serialized_engine);
        file.read(serialized_engine, size);
        file.close();

        // 创建推理运行环境实例
        runtime = createInferRuntime(gLogger);
        assert(runtime);
        // 反序列化模型
        engine = runtime->deserializeCudaEngine(serialized_engine, size);
        assert(engine);
        // 创建推理上下文
        context = engine->createExecutionContext();
        assert(context);
        delete[] serialized_engine;

        // Create stream
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine->getNbBindings() == 2);

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(kInputTensorName);
        const int outputIndex = engine->getBindingIndex(kOutputTensorName);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        Dims dims= context->getBindingDimensions(1);
        int d = dims.d[1];
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc(&gpu_buffers[inputIndex], kBatchSize * kChannel * kInputH * kInputW * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_buffers[outputIndex], kBatchSize * kOutputSize * sizeof(float)));
        return true;
    }
    catch(Exception ex) {
        return false;
    }
}

void cudaInit() {
    float* tmp = new float[kBatchSize * kChannel * kInputH * kInputW]();

    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[0], tmp, kBatchSize * kChannel * kInputH * kInputW * sizeof(float), cudaMemcpyHostToDevice, stream)); // CPU->GPU

    context->enqueueV2(gpu_buffers, stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream)); // GPU->CPU

    cudaStreamSynchronize(stream); // Block the CPU until all operations in the CUDA stream are completed

    delete[] tmp;
}

void inference(const Mat img, vector<Detection>& detections) {
    // Step1: Preprocess
    Mat tmp = formatToSquare(img);
    float x_factor = (float)tmp.cols / kInputW;
    float y_factor = (float)tmp.rows / kInputH;
    
    //Mat blob = cv::dnn::blobFromImage(dst, 1.0 / 255, Size(640, 640), Scalar(0, 0, 0), true);

    resize(tmp, tmp, Size(640, 640));
    tmp.convertTo(tmp, CV_32FC3, 1.0 / 255);
    for (int r = 0; r < kInputH; r++)
    {
        const float* rowData = tmp.ptr<float>(r);
        for (int c = 0; c < kInputW; c++)
        {
            mydata[0 * kInputH * kInputW + r * kInputW + c] = rowData[3 * c + 2]; // R
            mydata[1 * kInputH * kInputW + r * kInputW + c] = rowData[3 * c + 1]; // G
            mydata[2 * kInputH * kInputW + r * kInputW + c] = rowData[3 * c]; // B
        }
    }

    // Step2: Infer
    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[0], mydata, kBatchSize * kChannel * kInputH * kInputW * sizeof(float), cudaMemcpyHostToDevice, stream)); // CPU->GPU

    context->enqueueV2(gpu_buffers, stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream)); // GPU->CPU

    cudaStreamSynchronize(stream); // Block the CPU until all operations in the CUDA stream are completed

    // Step3: Postprocess
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    for (int i = 0; i < kNumOutputBbox; i++) {
        float confidence = output[(kNumClass + 5) * i + 4];
        if (confidence <= 0.25) continue;

        Mat scores(1, kNumClass, CV_32FC1, &output[(kNumClass + 5) * i + 5]);
        Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        float x = output[(kNumClass + 5) * i + 0];
        float y = output[(kNumClass + 5) * i + 1];
        float w = output[(kNumClass + 5) * i + 2];
        float h = output[(kNumClass + 5) * i + 3];
        int left = int((x - 0.5 * w) * x_factor);
        int top = int((y - 0.5 * h) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);

        confidences.push_back(confidence);
        class_ids.push_back(class_id.x);
        boxes.push_back(Rect(left, top, width, height));
    }

    vector<int> nms_result;
    NMSBoxes(boxes, confidences, kConfThresh, kNmsThresh, nms_result); // NMS
    for (unsigned long i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        detections.push_back(result);
    }    
}

void inference_batch(vector<Mat> imgs, vector<vector<Detection>>& detections) {
    // Step1: Preprocess
    float x_factor = 1;
    float y_factor = 1;

    //Mat blob = cv::dnn::blobFromImages(imgs, 1.0 / 255, Size(0, 0), Scalar(0, 0, 0));
    for (int b = 0; b < kBatchSize; b++)
    {
        Mat img = imgs[b];
        img.convertTo(img, CV_32FC1, 1.0 / 255);
        for (int r = 0; r < kInputH; r++)
        {
            const float* rowData = img.ptr<float>(r);
            for (int c = 0; c < kInputW; c++)
            {
                mydata[b * kChannel * kInputH * kInputW + 0 * kInputH * kInputW + r * kInputW + c] = rowData[3 * c + 2]; // R
                mydata[b * kChannel * kInputH * kInputW + 1 * kInputH * kInputW + r * kInputW + c] = rowData[3 * c + 1]; // G
                mydata[b * kChannel * kInputH * kInputW + 2 * kInputH * kInputW + r * kInputW + c] = rowData[3 * c]; // B
            }
        }
    }


    // Step2: Infer
    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[0], mydata, kBatchSize * kChannel * kInputH * kInputW * sizeof(float), cudaMemcpyHostToDevice, stream)); // CPU->GPU

    context->enqueueV2(gpu_buffers, stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream)); // GPU->CPU

    cudaStreamSynchronize(stream); // Block the CPU until all operations in the CUDA stream are completed

    // Step3: Postprocess
    detections.clear();
    float* pdata = &output[0];
    for (int b = 0; b < kBatchSize; b++) {
        vector<Detection> detection;
        vector<int> class_ids;
        vector<float> confidences;
        vector<Rect> boxes;
        Mat tmp(25200, 85, CV_32FC1, pdata);
        for (int i = 0; i < kNumOutputBbox; i++) {
            float confidence = pdata[(kNumClass + 5) * i + 4];
            if (confidence <= 0.25) continue;
            Mat scores(1, kNumClass, CV_32FC1, pdata + (kNumClass + 5) * i + 5);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            float x = pdata[(kNumClass + 5) * i + 0];
            float y = pdata[(kNumClass + 5) * i + 1];
            float w = pdata[(kNumClass + 5) * i + 2];
            float h = pdata[(kNumClass + 5) * i + 3];
            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            confidences.push_back(confidence);
            class_ids.push_back(class_id.x);
            boxes.push_back(Rect(left, top, width, height));
        }

        vector<int> nms_result;
        NMSBoxes(boxes, confidences, kConfThresh, kNmsThresh, nms_result); // NMS
        for (unsigned long i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];
            Detection result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            detection.push_back(result);
        }
        pdata += kNumOutputBbox * (kNumClass + 5);
        detections.push_back(detection);
    }
    
}

void drawPred(Mat& img, vector<Detection> result, vector<Scalar> colors) {

    for (int i = 0; i < result.size(); ++i)
    {
        Detection detection = result[i];

        Rect box = detection.box;
        Scalar color = colors[detection.class_id];

        // Detection box
        rectangle(img, box, color, 2);

        // Detection box text
        string classString = classes[detection.class_id] + ' ' + to_string(detection.confidence).substr(0, 4);
        Size textSize = cv::getTextSize(classString, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        rectangle(img, textBox, color, FILLED);
        putText(img, classString, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}

void gpuMemoryRelease()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

int main() {
    //生成随机颜色
    vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < kNumClass; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }

   // 单图推理
    string engine_name = "models/yolov5s.engine";
    bool isOK = readEngine(engine_name, 0);
    Mat img = imread("images/bus.jpg");
    vector<Detection> detections{}; 
    inference(img, detections);
    drawPred(img, detections, color);
    imwrite("results/yolov5.jpg", img);

    // 批量推理 
   /* string engine_name = "models/yolov5s_batch8.engine";
    bool isOK = readEngine(engine_name, 0); 
    if (!isOK) {
        cout << "read " << engine_name << " error!" << :endl;
        return -1;
    }
    cudaInit();
    vector<string> filenames;
    glob("images", filenames);
    for (int i = 0; i < filenames.size(); ) {
        Mat tmp(kInputH, kInputW, CV_8UC3, Scalar(0, 0, 0));
        vector<Mat> images;
        vector<vector<Detection>> detections;
        int index = 0;
        
        for (int j = 0; j < kBatchSize; j++) {
            index = i + j;
            if(index < filenames.size()){
                Mat image = imread(filenames[index]);
                resize(image, image, Size(kInputW, kInputH));
                images.push_back(image);
            }
            else {
                images.push_back(tmp);
            }
        }
        auto start = chrono::high_resolution_clock::now();
        inference_batch(images, detections);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "执行时间: " << duration.count() << "ms" << endl;
        for (int j = 0; j < kBatchSize; j++) {
            index = i + j;
            if (index < filenames.size()) {
                drawPred(images[j], detections[j], color);
                imwrite("results/" + to_string(index) + ".jpg", images[j]);
            }
        }
        i += kBatchSize;
    }*/

    gpuMemoryRelease();
    return 0;
}

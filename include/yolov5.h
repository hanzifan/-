#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5         //zhi xin du
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8);

static int get_depth(int x, float gd);

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name);

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name);

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name);

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize);

void initMat(cv::Mat &mat, float(*p)[3]);

void transform(cv::Mat &H,cv::cuda::GpuMat &ir, cv::Mat &result);

int yolo(cv::Mat &vis_frame, cv::Mat &thermel_frame);

class Yolov5{
public:
    Yolov5();
    Yolov5(std::string engine_path);
    ~Yolov5();
    cv::Mat yolo(cv::Mat &frame, std::vector<cv::Rect> &debug, int* count);
    cv::Mat vis_yolo(cv::Mat &vis_frame, int count);
    cv::Mat ir_yolo(cv::Mat &thermel_frame, int count);
    int yolo_test();
    int kcftrack(cv::Mat fusion_result, int xMin, int yMin, int width, int height);

private:

private:
    //have no idea about it now 2022.03.31
    float prob[BATCH_SIZE * OUTPUT_SIZE];

    //engine
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    
    //cuda memory buffers
    float* buffers[2];
    int inputIndex;
    int outputIndex;

    //cuda memory stream
    cudaStream_t stream;

    //host/device pointer
    uint8_t* img_host;
    uint8_t* img_device;

    //homo buffer
    std::vector<cv::Mat> homo_buffer;

    //tracking start flag
    bool start_tracking;

    //tracking usage
    int topleft_x;
    int topleft_y;
    int width;
    int height;
};





/*
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
*/
#include <yolov5.h>
#include "camera_v4l2_cuda.h"
#include <runtracker.h>
#include <mutex>

std::mutex mtx_imshow;

/*
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
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
*/

static int get_width(int x, float gw, int divisor) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    assert(conv0);
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");
    /* ------ yolov5 head ------ */
    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto c3_10 = C3(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.10");
    auto sppf11 = SPPF(network, weightMap, *c3_10->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.11");

    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *sppf11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
    auto upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
    ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    auto upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
    ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
    auto upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
    ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
    IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
    IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
    IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
        engine = build_engine_p6(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


void initMat(cv::Mat &mat, float(*p)[3]){
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            mat.at<float>(i,j) = *(*(p+i)+j);
        }
    }
}

void transform(cv::Mat &H,cv::cuda::GpuMat &ir, cv::Mat &result){
    cv::cuda::warpPerspective(ir, ir, H, result.size());
}

void load_homo(std::string txt_path, std::vector<cv::Mat> &homo_buffer){
    //读取数据，每行存9个float数据作为一个homo矩阵
    std::ifstream myfile(txt_path);
    while(!(myfile.eof())){
        if (!myfile.is_open()){
            std::cout << "can not open this file" << std::endl;
        }
        float test[3][3];
        for(int i = 0; i < 3; i++){
           for(int j = 0; j < 3; j++){
                myfile >> test[i][j];
            }
        }

        std::vector<cv::Mat> h;
        cv::Mat test_m(3, 3, CV_32F);
        initMat(test_m, test);
        homo_buffer.push_back(test_m);
    }

    myfile.close();
}

Yolov5::Yolov5(){
    //load homo matrix
    std::string txt_path = "../homography.txt";
    load_homo(txt_path, homo_buffer);

    //set tracking start flag
    start_tracking = 1;

    cudaSetDevice(DEVICE);

    //load engine
    std::string wts_name = "";
    std::string engine_name = "./best.engine";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        // return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    prob[BATCH_SIZE * OUTPUT_SIZE];
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
    img_host = nullptr;
    img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
}

Yolov5::Yolov5(std::string engine_path){
    //load homo matrix
    std::string txt_path = "../homography.txt";
    load_homo(txt_path, homo_buffer);

    //set tracking start flag
    start_tracking = 1;

    cudaSetDevice(DEVICE);

    //load engine
    std::string wts_name = "";
    std::string engine_name = "./" + engine_path;
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        // return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    prob[BATCH_SIZE * OUTPUT_SIZE];
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
    img_host = nullptr;
    img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
}

Yolov5::~Yolov5(){
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

cv::Mat Yolov5::yolo(cv::Mat &frame, std::vector<cv::Rect> &debug, int* count){
    int fcount = 0;
    KCFTracker tracker(1, 0, 1, 0);
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    // for (;;) {
        // auto start_debug = std::chrono::system_clock::now();
        fcount++;
        float* buffer_idx = (float*)buffers[inputIndex];
        cv::Mat img = frame;
        auto end_f = std::chrono::system_clock::now();
        // std::cout << "fusion time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_f - start_f).count() << "ms" << std::endl;	    

        imgs_buffer[0] = img;
        auto start_pre = std::chrono::system_clock::now();
        size_t  size_image = img.cols * img.rows * 3;
        size_t  size_image_dst = INPUT_H * INPUT_W * 3;
        //copy data to pinned memory
        memcpy(img_host,img.data,size_image);
        //copy data to device memory
        CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);       
        buffer_idx += size_image_dst;
        auto end_pre = std::chrono::system_clock::now();
        // std::cout << "pre-process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count() << "ms" << std::endl;
        // auto end_debug1 = std::chrono::system_clock::now();
        // std::cout << "debug1 time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_debug1 - start_debug1).count() << "ms" << std::endl;
        // // }
        // auto end_debug = std::chrono::system_clock::now();
        // std::cout << "debug time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_debug - start_debug).count() << "ms" << std::endl;
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        // std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        
        auto start_nms = std::chrono::system_clock::now();
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            count[0] = res.size();
            // std::cout << res.size() << std::endl;
            cv::Mat detect = imgs_buffer[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(detect, res[j].bbox);
                // std::cout << "test coordinate:" << r.tl().x << " " << r.tl().y << std::endl;
                cv::rectangle(detect, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                std::string label;
                if((int)res[j].class_id == 0)
                    label = "human";
                else if((int)res[j].class_id == 1)
                    label = "bicycle";
                else if((int)res[j].class_id == 2)
                    label = "car";
                cv::putText(detect, label, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                cv::putText(detect, std::to_string(j), cv::Point(r.x + r.width, r.y + 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            imgs_buffer[b] = detect;
            
            // cv::imwrite("../detec/" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
        debug.clear();
        auto& res = batch_res[0];
        if(res.size() != 0){
            cv::Rect r = get_rect(imgs_buffer[0], res[0].bbox);
            // debug.push_back(r);
        }

        auto end_nms = std::chrono::system_clock::now();
        // std::cout << "nms time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_nms - start_nms).count() << "ms" << std::endl;


    return imgs_buffer[0];
}

cv::Mat Yolov5::ir_yolo(cv::Mat &thermel_frame, int count){
    int fcount = 0;
    KCFTracker tracker(1, 0, 1, 0);
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    // for (;;) {
        // auto start_debug = std::chrono::system_clock::now();
        fcount++;
        float* buffer_idx = (float*)buffers[inputIndex];
        cv::Mat ir = thermel_frame;	//image in	    

        imgs_buffer[0] = ir;
        auto start_pre = std::chrono::system_clock::now();
        size_t  size_image = ir.cols * ir.rows * 3;
        size_t  size_image_dst = INPUT_H * INPUT_W * 3;
        //copy data to pinned memory
        memcpy(img_host,ir.data,size_image);
        //copy data to device memory
        CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
        preprocess_kernel_img(img_device, ir.cols, ir.rows, buffer_idx, INPUT_W, INPUT_H, stream);       
        buffer_idx += size_image_dst;
        auto end_pre = std::chrono::system_clock::now();
        // std::cout << "pre-process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count() << "ms" << std::endl;
        // auto end_debug1 = std::chrono::system_clock::now();
        // std::cout << "debug1 time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_debug1 - start_debug1).count() << "ms" << std::endl;
        // // }
        // auto end_debug = std::chrono::system_clock::now();
        // std::cout << "debug time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_debug - start_debug).count() << "ms" << std::endl;
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        
        auto start_nms = std::chrono::system_clock::now();
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            std::cout << res.size() << std::endl;
            cv::Mat detect = imgs_buffer[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(detect, res[j].bbox);
                // std::cout << "test coordinate:" << r.tl().x << " " << r.tl().y << std::endl;
                cv::rectangle(detect, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                std::string label;
                if((int)res[j].class_id == 0)
                    label = "human";
                else if((int)res[j].class_id == 1)
                    label = "bicycle";
                else if((int)res[j].class_id == 2)
                    label = "car";
                cv::putText(detect, label, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                cv::putText(detect, std::to_string(j), cv::Point(r.x + r.width, r.y + 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            imgs_buffer[b] = detect;
            
            // cv::imwrite("../detec/" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
        // if(count < 3000){
        //     std::string name = std::to_string(count) + ".jpg";
        //     name = "../detec/" + name;
        //     cv::imwrite(name, imgs_buffer[0]);
        // }
        // cv::waitKey(5);
        auto end_nms = std::chrono::system_clock::now();
        std::cout << "nms time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_nms - start_nms).count() << "ms" << std::endl;

        // fcount = 0;
        // if(start_tracking == 1){
        //     auto& res = batch_res[0];
        //     if(res.size() == 0)
        //         break;
        //     std::cout << res.size() << std::endl;
        //     cv::Rect r = get_rect(imgs_buffer[0], res[0].bbox);
        //     topleft_x = r.tl().x;
        //     topleft_y = r.tl().y;
        //     width = r.width;
        //     height = r.height;
        //     // std::cout << topleft_x << " " << topleft_y << " " << width << " " << height << std::endl;
        //     runtracking(tracker, imgs_buffer[0], start_tracking, topleft_x, topleft_y, width, height);
        //     start_tracking = 0;
        // }
        // else{
        //     auto start_track = std::chrono::system_clock::now();
        //     runtracking(tracker, imgs_buffer[0], start_tracking, topleft_x, topleft_y, width, height);
        //     auto end_track = std::chrono::system_clock::now();
        //     std::cout << "tracking time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_track - start_track).count() << "ms" << std::endl;
        // }
    // }


    return imgs_buffer[0];
}

cv::Mat Yolov5::vis_yolo(cv::Mat &vis_frame, int count){
    int fcount = 0;
    KCFTracker tracker(1, 0, 1, 0);
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    // for (;;) {
        // auto start_debug = std::chrono::system_clock::now();
        fcount++;
        float* buffer_idx = (float*)buffers[inputIndex];

        cv::Mat img = vis_frame;

        cv::Mat bgr(1080, 1920, CV_8UC3);	   
        // std::cout << "debug" << std::endl;
        // std::cout << "vis_frame channels: " << vis_frame.channels() << std::endl;
        cv::cvtColor(img, bgr, cv::COLOR_YUV2BGR_YUYV); 

        imgs_buffer[0] = bgr;
        auto start_pre = std::chrono::system_clock::now();
        size_t  size_image = bgr.cols * bgr.rows * 3;
        size_t  size_image_dst = INPUT_H * INPUT_W * 3;
        //copy data to pinned memory
        memcpy(img_host,bgr.data,size_image);
        //copy data to device memory
        CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
        preprocess_kernel_img(img_device, bgr.cols, bgr.rows, buffer_idx, INPUT_W, INPUT_H, stream);       
        buffer_idx += size_image_dst;

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        
        auto start_nms = std::chrono::system_clock::now();
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            std::cout << "res size: " << res.size() << std::endl;
            cv::Mat detect = imgs_buffer[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(detect, res[j].bbox);
                // std::cout << "test coordinate:" << r.tl().x << " " << r.tl().y << std::endl;
                cv::rectangle(detect, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                std::string label;
                if((int)res[j].class_id == 0)
                    label = "human";
                else if((int)res[j].class_id == 1)
                    label = "bicycle";
                else if((int)res[j].class_id == 2)
                    label = "car";
                cv::putText(detect, label, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                cv::putText(detect, std::to_string(j), cv::Point(r.x + r.width, r.y + 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            imgs_buffer[b] = detect;
            
            // cv::imwrite("../detec/" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
        // if(count < 3000){
        //     std::string name = std::to_string(count) + ".jpg";
        //     name = "../detec/" + name;
        //     cv::imwrite(name, imgs_buffer[0]);
        // }
        // cv::waitKey(5);
        auto end_nms = std::chrono::system_clock::now();
        std::cout << "nms time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_nms - start_nms).count() << "ms" << std::endl;

        // fcount = 0;
        // if(start_tracking == 1){
        //     auto& res = batch_res[0];
        //     if(res.size() == 0)
        //         break;
        //     std::cout << res.size() << std::endl;
        //     cv::Rect r = get_rect(imgs_buffer[0], res[0].bbox);
        //     topleft_x = r.tl().x;
        //     topleft_y = r.tl().y;
        //     width = r.width;
        //     height = r.height;
        //     // std::cout << topleft_x << " " << topleft_y << " " << width << " " << height << std::endl;
        //     runtracking(tracker, imgs_buffer[0], start_tracking, topleft_x, topleft_y, width, height);
        //     start_tracking = 0;
        // }
        // else{
        //     auto start_track = std::chrono::system_clock::now();
        //     runtracking(tracker, imgs_buffer[0], start_tracking, topleft_x, topleft_y, width, height);
        //     auto end_track = std::chrono::system_clock::now();
        //     std::cout << "tracking time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_track - start_track).count() << "ms" << std::endl;
        // }
    // }


    return imgs_buffer[0];
}

int Yolov5::yolo_test() {
    std::string img_dir = "../samples";
    std::vector<std::string> file_names;
    // std::string img_dir_file = img_dir + "/vis(old)";
    std::string img_dir_file = img_dir + "/vis(old)";
    //if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    if(read_files_in_dir(img_dir_file.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

   
    int fcount = 0;
    std::cout << (int)file_names.size() << std::endl;
	KCFTracker tracker(1, 0, 1, 0);
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    for (int f = 0; f < (int)file_names.size(); f++) {
        auto start_whole = std::chrono::system_clock::now();
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        //auto start = std::chrono::system_clock::now();
        float* buffer_idx = (float*)buffers[inputIndex];
        for (int b = 0; b < fcount; b++) {
            // auto start_h = std::chrono::system_clock::now();
            cv::Mat img = cv::imread(img_dir + "/vis(old)/" + file_names[f - fcount + 1 + b]);
            cv::Mat img_ir = cv::imread(img_dir + "/ir(old)/" + file_names[f - fcount + 1 + b]); cv::Mat img_irgray;		//image in
            auto start_h = std::chrono::system_clock::now();
            cv::Mat H(3, 3, CV_32F);
            H = homo_buffer[1];
            cv::cuda::GpuMat ir_cu(img_ir);;
            transform(H, ir_cu, img);
            // auto end_h = std::chrono::system_clock::now();
            cv::Mat ir;
            ir_cu.download(ir);
            auto end_h = std::chrono::system_clock::now();
            std::cout << "homo time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_h - start_h).count() << "ms" << std::endl;
            //std::cout << img.size() << " " << img_ir.size() << std::endl;

            auto start_f = std::chrono::system_clock::now();
            fusion(img, ir, img_device, stream);
            auto end_f = std::chrono::system_clock::now();
            std::cout << "fusion time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_f - start_f).count() << "ms" << std::endl;	  
            // cv::Mat img =  cv::imread(img_dir + "/result/" + file_names[f - fcount + 1 + b]);

            if (img.empty()) continue;
            imgs_buffer[b] = img;
	        auto start_pre = std::chrono::system_clock::now();
            size_t  size_image = img.cols * img.rows * 3;
            size_t  size_image_dst = INPUT_H * INPUT_W * 3;

            // //copy data to pinned memory
            memcpy(img_host,img.data,size_image);
            // //copy data to device memory
            CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);       
            buffer_idx += size_image_dst;
            auto end_pre = std::chrono::system_clock::now();
            std::cout << "pre-process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count() << "ms" << std::endl;
        }
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);

        //use nms to fuzhi batchres
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            std::cout << res.size() << std::endl;
            cv::Mat img = imgs_buffer[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                // std::cout << "test coordinate:" << r.tl().x << " " << r.tl().y << std::endl;
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                std::string label;
                if((int)res[j].class_id == 0)
                    label = "human";
                else if((int)res[j].class_id == 1)
                    label = "bicycle";
                else if((int)res[j].class_id == 2)
                    label = "car";
                cv::putText(img, label, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                cv::putText(img, std::to_string(j), cv::Point(r.x + r.width, r.y + 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            // cv::imwrite("../detec/" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;

		cv::imshow("detection", imgs_buffer[0]);
		cv::waitKey(1);

        // if(start_tracking == 1){

		//     std::cout << "debug1" << std::endl;
        //     auto& res = batch_res[0];
        //     std::cout << res.size() << std::endl;
        //     if(res.size() == 0 )
        //         continue;
        //     cv::Rect r = get_rect(imgs_buffer[0], res[3].bbox);
        //     topleft_x = r.tl().x;
        //     topleft_y = r.tl().y;
        //     width = r.width;
        //     height = r.height;
        //     // std::cout << topleft_x << " " << topleft_y << " " << width << " " << height << std::endl;
        //     runtracking(tracker, imgs_buffer[0], start_tracking, topleft_x, topleft_y, width, height);
        //     start_tracking = 0;
        // }
        // else{
		//     std::cout << "debug2" << std::endl;
        //     auto start_track = std::chrono::system_clock::now();
        //     runtracking(tracker, imgs_buffer[0], start_tracking, topleft_x, topleft_y, width, height);
        //     auto end_track = std::chrono::system_clock::now();
        //     std::cout << "tracking time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_track - start_track).count() << "ms" << std::endl;
        // }
        // auto end_whole = std::chrono::system_clock::now();
        // std::cout << "whole time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_whole - start_whole).count() << "ms" << std::endl;
    }

    return 0;
}


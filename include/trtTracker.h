#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "ConvertBBoxPlugin.h"
#include "DepthwiseCorrelationPlugin.h"
#include "backbone.hpp"
#include "cuda_utils.h"

//input
static constexpr int TEMPLATE_SIZE = 127;
static constexpr int TEMPLATE_FEAT_SIZE = 6;
static constexpr int ANCHOR_STRIDE = 8;
static constexpr int INSTANCE_SIZE = 287;
int SCORE_SIZE = (INSTANCE_SIZE - TEMPLATE_SIZE) / ANCHOR_STRIDE + 1;
int INSTANCE_SIZE_LONG = 487;
int SCORE_SIZE_LONG = (INSTANCE_SIZE_LONG - TEMPLATE_SIZE) / ANCHOR_STRIDE + 1;

// rpn
// Anchor ratios
float cfg_ANCHOR_RATIOS[5] = {0.33, 0.5, 1, 2, 3};
// Anchor scales
int cfg_ANCHOR_SCALES[1] = {8};
// Anchor number
static constexpr int ANCHOR_NUM = sizeof(cfg_ANCHOR_RATIOS) / sizeof(cfg_ANCHOR_RATIOS[0]) * 
                                sizeof(cfg_ANCHOR_SCALES) / sizeof(cfg_ANCHOR_SCALES[0]);
// anchor
static const std::vector<float> ANCHOR_RATIOS = {0.33, 0.5, 1, 2, 3};
static const std::vector<float> ANCHOR_SIZES = {8.0};

static const std::vector<std::string> OUTPUT_NAMES = {"scores", "boxes"};


// TrtTracker, could satify shortterm and longterm, just use different engine
class TrtTracker{
public:
    TrtTracker();
    bool putEngine(std::string engine_file_path);
    ~TrtTracker();
    bool track(float* img, int size, std::vector<float> &my_template, std::vector<float> &score, std::vector<float> &bbox);

private:
    //engine
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    
    //cuda memory buffers
    std::vector<void *> buffers = std::vector<void *>(4);
    int input0Index;
    int input1Index;
    int output0Index;
    int output1Index;

    //cuda memory stream
    cudaStream_t stream;

    //host/device pointer
    uint8_t* img_host;
    uint8_t* img_device;  

    //input
    std::vector<float> data0;
    std::vector<float> data1;

    //output
    std::vector<float> scores_h = std::vector<float>(SCORE_SIZE * SCORE_SIZE * ANCHOR_NUM);
    std::vector<float> boxes_h = std::vector<float>(SCORE_SIZE * SCORE_SIZE * ANCHOR_NUM * 4);
};

class longTracker{
public:
    longTracker();
    bool putEngine(std::string engine_file_path);
    ~longTracker();
    bool track(float* img, int size, std::vector<float> &my_template, std::vector<float> &score, std::vector<float> &bbox);

private:
    //engine
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    
    //cuda memory buffers
    std::vector<void *> buffers = std::vector<void *>(4);
    int input0Index;
    int input1Index;
    int output0Index;
    int output1Index;

    //cuda memory stream
    cudaStream_t stream;

    //host/device pointer
    uint8_t* img_host;
    uint8_t* img_device;  

    //input
    std::vector<float> data0;
    std::vector<float> data1;

    //output
    std::vector<float> scores_h = std::vector<float>(SCORE_SIZE_LONG * SCORE_SIZE_LONG * ANCHOR_NUM);
    std::vector<float> boxes_h = std::vector<float>(SCORE_SIZE_LONG * SCORE_SIZE_LONG * ANCHOR_NUM * 4);
};

// TrackerInit, use to initial
class TrackerInit{
public:
    TrackerInit();
    bool putEngine(std::string engine_file_path);
    ~TrackerInit();
    bool init(float* img, int size, std::vector<float> &score);

private:
    //engine
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    
    //cuda memory buffers
    std::vector<void *> buffers = std::vector<void *>(2);
    int inputIndex;
    int outputIndex;

    //cuda memory stream
    cudaStream_t stream;

    //host/device pointer
    uint8_t* img_host;
    uint8_t* img_device;  

    //input
    std::vector<float> data0;
    // std::vector<float *> inputs;

    //output
    std::vector<float> scores_h = std::vector<float>(TEMPLATE_FEAT_SIZE * TEMPLATE_FEAT_SIZE * 256);
    // std::vector<float *> outputs;
};
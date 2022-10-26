#include "trtTracker.h"
#include "logging.h"
#include <vector>

static Logger gLogger;

void SiamRPN_doInference(IExecutionContext &context, cudaStream_t &stream,
                 std::vector<void *> &buffers, std::vector<float *> &inputs,
                 std::vector<float *> &outputs, int INSTANCE_SIZE) {
  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to host
  int SCORE_SIZE = (INSTANCE_SIZE - TEMPLATE_SIZE) / ANCHOR_STRIDE + 1;
  CUDA_CHECK(
      cudaMemcpyAsync(buffers[0], inputs[0],
                      3 * INSTANCE_SIZE * INSTANCE_SIZE * sizeof(float),
                      cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(buffers[1], inputs[1],
                             256 * TEMPLATE_FEAT_SIZE * TEMPLATE_FEAT_SIZE *
                                 sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  context.enqueueV2(buffers.data(), stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(outputs[0], buffers[2],
                             SCORE_SIZE * SCORE_SIZE * ANCHOR_NUM * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(outputs[1], buffers[3],
                             SCORE_SIZE * SCORE_SIZE * ANCHOR_NUM * 4 * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void Alex_doInference(IExecutionContext &context, cudaStream_t &stream,
                 std::vector<void *> &buffers, std::vector<float *> &inputs,
                 std::vector<float *> &outputs) {
  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to hosts
  CUDA_CHECK(
      cudaMemcpyAsync(buffers[0], inputs[0],
                      3 * TEMPLATE_SIZE * TEMPLATE_SIZE * sizeof(float),
                      cudaMemcpyHostToDevice, stream));
  context.enqueueV2(buffers.data(), stream, nullptr);

  CUDA_CHECK(cudaMemcpyAsync(outputs[0], buffers[1],
                             256 * TEMPLATE_FEAT_SIZE * TEMPLATE_FEAT_SIZE *
                                 sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

// debug
void printVector(std::vector<float *> vec){
    for(auto val:vec)
        std::cout << val << " ";
    std::cout << std::endl;
}

// TrtTracker: SiamRPN
TrtTracker::TrtTracker(){
    
}

bool TrtTracker::putEngine(std::string engine_file_path){
// deserialize the .engine and run inference
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_file_path << " error!" << std::endl;
        return -1;
    }

    std::string trtModelStream;
    size_t modelSize{0};
    file.seekg(0, file.end);
    modelSize = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream.resize(modelSize);
    assert(!trtModelStream.empty());
    file.read(const_cast<char *>(trtModelStream.c_str()), modelSize);
    file.close();

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream.c_str(), modelSize);
    assert(engine != nullptr);context = engine->createExecutionContext();
    assert(context != nullptr);

    CUDA_CHECK(cudaStreamCreate(&stream));
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine->getNbBindings() == 4);
    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than
    // IEngine::getNbBindings()
    input0Index = engine->getBindingIndex("search");
    input1Index = engine->getBindingIndex("template");
    output0Index = engine->getBindingIndex("scores");
    output1Index = engine->getBindingIndex("boxes");
    context->setBindingDimensions(input0Index,
                                Dims4(1, 3, INSTANCE_SIZE, INSTANCE_SIZE));
    context->setBindingDimensions(
            input1Index, Dims4(1, 256, TEMPLATE_FEAT_SIZE, TEMPLATE_FEAT_SIZE));
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[input0Index],
                        3 * INSTANCE_SIZE * INSTANCE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[input1Index], 256 * TEMPLATE_FEAT_SIZE *
                                                   TEMPLATE_FEAT_SIZE *
                                                   sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output0Index], SCORE_SIZE * SCORE_SIZE * ANCHOR_NUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output1Index], SCORE_SIZE * SCORE_SIZE * ANCHOR_NUM * 4 * sizeof(float)));

    return 1;
}

bool TrtTracker::track(float* img, int size, std::vector<float> &my_template, std::vector<float> &score, std::vector<float> &bbox){
    std::vector<float> temp(img, img+size);
    data0 = temp;
    // int iter = 0;
    // for(auto i : data0){
    //     iter++;
    //     std::cout << i << " ";
    //     if(iter > 100)
    //         break;
    // }
    data1 = my_template;
    std::vector<float *> inputs = {data0.data(), data1.data()};
    std::vector<float *> outputs = {scores_h.data(), boxes_h.data()};
    // Run inference
    auto start = std::chrono::system_clock::now();
    SiamRPN_doInference(*context, stream, buffers, inputs, outputs, INSTANCE_SIZE);
    score = scores_h;
    bbox = boxes_h;
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    return 1;
}

TrtTracker::~TrtTracker(){
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[input0Index]));
    CUDA_CHECK(cudaFree(buffers[input1Index]));
    CUDA_CHECK(cudaFree(buffers[output0Index]));
    CUDA_CHECK(cudaFree(buffers[output1Index]));

    context->destroy();
    engine->destroy();
    runtime->destroy();
}

// longTracker: SiamRPN
longTracker::longTracker(){
    
}

bool longTracker::putEngine(std::string engine_file_path){
// deserialize the .engine and run inference
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_file_path << " error!" << std::endl;
        return -1;
    }

    std::string trtModelStream;
    size_t modelSize{0};
    file.seekg(0, file.end);
    modelSize = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream.resize(modelSize);
    assert(!trtModelStream.empty());
    file.read(const_cast<char *>(trtModelStream.c_str()), modelSize);
    file.close();

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream.c_str(), modelSize);
    assert(engine != nullptr);context = engine->createExecutionContext();
    assert(context != nullptr);

    CUDA_CHECK(cudaStreamCreate(&stream));
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine->getNbBindings() == 4);
    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than
    // IEngine::getNbBindings()
    input0Index = engine->getBindingIndex("search");
    input1Index = engine->getBindingIndex("template");
    output0Index = engine->getBindingIndex("scores");
    output1Index = engine->getBindingIndex("boxes");
    context->setBindingDimensions(input0Index,
                                Dims4(1, 3, INSTANCE_SIZE_LONG, INSTANCE_SIZE_LONG));
    context->setBindingDimensions(
            input1Index, Dims4(1, 256, TEMPLATE_FEAT_SIZE, TEMPLATE_FEAT_SIZE));
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[input0Index],
                        3 * INSTANCE_SIZE_LONG * INSTANCE_SIZE_LONG * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[input1Index], 256 * TEMPLATE_FEAT_SIZE *
                                                   TEMPLATE_FEAT_SIZE *
                                                   sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output0Index], SCORE_SIZE_LONG * SCORE_SIZE_LONG * ANCHOR_NUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output1Index], SCORE_SIZE_LONG * SCORE_SIZE_LONG * ANCHOR_NUM * 4 * sizeof(float)));

    return 1;
}

bool longTracker::track(float* img, int size, std::vector<float> &my_template, std::vector<float> &score, std::vector<float> &bbox){
    std::vector<float> temp(img, img+size);
    data0 = temp;
    // for(auto i : data0)
    //     std::cout << i << " ";
    data1 = my_template;
    std::vector<float *> inputs = {data0.data(), data1.data()};
    std::vector<float *> outputs = {scores_h.data(), boxes_h.data()};
    // Run inference
    auto start = std::chrono::system_clock::now();
    SiamRPN_doInference(*context, stream, buffers, inputs, outputs, INSTANCE_SIZE_LONG);
    score = scores_h;
    bbox = boxes_h;
    // std::cout << "output size:" << boxes_h.size() << std::endl;
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    return 1;
}

longTracker::~longTracker(){
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[input0Index]));
    CUDA_CHECK(cudaFree(buffers[input1Index]));
    CUDA_CHECK(cudaFree(buffers[output0Index]));
    CUDA_CHECK(cudaFree(buffers[output1Index]));

    context->destroy();
    engine->destroy();
    runtime->destroy();
}

// TrackerInit: Alexnet
TrackerInit::TrackerInit(){
}

bool TrackerInit::putEngine(std::string engine_file_path){
    // deserialize the .engine and run inference
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_file_path << " error!" << std::endl;
        return -1;
    }

    std::string trtModelStream;
    size_t modelSize{0};
    file.seekg(0, file.end);
    modelSize = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream.resize(modelSize);
    assert(!trtModelStream.empty());
    file.read(const_cast<char *>(trtModelStream.c_str()), modelSize);
    file.close();

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream.c_str(), modelSize);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    CUDA_CHECK(cudaStreamCreate(&stream));
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than
    // IEngine::getNbBindings()
    
    context->setBindingDimensions(0, Dims4(1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE));
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[0],
                            3 * TEMPLATE_SIZE * TEMPLATE_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&buffers[1], 256 * TEMPLATE_FEAT_SIZE *
                                                    TEMPLATE_FEAT_SIZE *
                                                    sizeof(float)));

    return 1;
}

bool TrackerInit::init(float* img, int size, std::vector<float> &score){
    std::vector<float> temp(img, img+size);
    data0 = temp;
    // for(auto i : data0)
    //     std::cout << i << " ";
    std::vector<float *> inputs = {data0.data()};
    std::vector<float *> outputs = {scores_h.data()};
    // Run inference
    auto start = std::chrono::system_clock::now();
    Alex_doInference(*context, stream, buffers, inputs, outputs);
    score = scores_h;
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    return 1;
}

TrackerInit::~TrackerInit(){
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));

    context->destroy();
    engine->destroy();
    runtime->destroy();
}

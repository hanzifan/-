#include <cstdint>
#include "DepthwiseCorrelationPlugin.h"
#include <algorithm>


namespace nvinfer1 {
DimsExprs DepthwiseCorrelationPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs *inputs, int nbInputs,
    IExprBuilder &exprBuilder) {
  assert(nbInputs == 2);
  assert(outputIndex == 0); 
  DimsExprs ret(inputs[0]);
  auto sub1 = exprBuilder.operation(DimensionOperation::kSUB, *inputs[0].d[2], *inputs[1].d[2]);
  auto sub2 = exprBuilder.operation(DimensionOperation::kSUB, *inputs[0].d[3], *inputs[1].d[3]);
  ret.d[2] = exprBuilder.operation(DimensionOperation::kSUM, *sub1, *exprBuilder.constant(1));
  ret.d[3] = exprBuilder.operation(DimensionOperation::kSUM, *sub2, *exprBuilder.constant(1));
  return ret;
}
bool DepthwiseCorrelationPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  // 2 inputs, 1 outputs, so 3 input/output in total
  assert(0 <= pos && pos < 3);
  const auto *in = inOut;
  const auto *out = inOut + nbInputs;
  const bool consistentFloatPrecision = (in[0].type == in[pos].type);
  switch (pos) {
  case 0:
    return in[0].type == DataType::kFLOAT &&
           in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
  case 1:
    return in[1].type == DataType::kFLOAT &&
           in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
  case 2:
    return out[0].type == DataType::kFLOAT &&
           out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
  }
  return false;
}
void DepthwiseCorrelationPlugin::attachToContext(
    cudnnContext *cudnnContext, cublasContext *cublasContext,
    IGpuAllocator *gpuAllocator) {
  mCudnnHandle = cudnnContext;
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnCreateFilterDescriptor(&filt_desc);
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);
}

// Detach the plugin object from its execution context.
void DepthwiseCorrelationPlugin::detachFromContext() {

  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyFilterDescriptor(filt_desc);
}

cudnnStatus_t convertTrt2cudnnDtype(nvinfer1::DataType trt_dtype,
                                    cudnnDataType_t *cudnn_dtype) {
  switch (trt_dtype) {
  case nvinfer1::DataType::kFLOAT:
    *cudnn_dtype = CUDNN_DATA_FLOAT;
    break;
  case nvinfer1::DataType::kHALF:
    *cudnn_dtype = CUDNN_DATA_HALF;
    break;
  default:
    return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

int DepthwiseCorrelationPlugin::enqueue(const PluginTensorDesc *inputDesc,
                                        const PluginTensorDesc *outputDesc,
                                        const void *const *inputs,
                                        void *const *outputs, void *workspace,
                                        cudaStream_t stream) {
  // inputs 0: x 1: kernel
  cudnnDataType_t cudnn_dtype;
  CUDNN_CALL(convertTrt2cudnnDtype(inputDesc[0].type, &cudnn_dtype));
  Dims input_dims = inputDesc[0].dims;
  int32_t c = input_dims.d[1];
  int32_t xh = input_dims.d[2];
  int32_t xw = input_dims.nbDims > 3 ? input_dims.d[3] : 1;

  CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, cudnn_dtype,
                                        1, c, xh, xw));
  Dims inputk_dims = inputDesc[1].dims;
  int32_t out_ch = inputk_dims.d[0];
  int32_t kh = inputk_dims.d[2];
  int32_t kw = inputk_dims.nbDims > 3 ? inputk_dims.d[3] : 1;
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filt_desc, cudnn_dtype,
                                        CUDNN_TENSOR_NCHW, out_ch, 1, kh, kw));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1,
                                             CUDNN_CONVOLUTION, cudnn_dtype));

  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, c));
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                        cudnn_dtype, out_n, out_c, out_h, out_w));
  float alpha = 1;
  float beta = 0;
  CUDNN_CALL(cudnnSetStream(mCudnnHandle, stream));
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  // const int algo_count = 8;
  // int  res_algo_count;
  // cudnnConvolutionFwdAlgoPerf_t algos[algo_count];
  // auto err = cudnnFindConvolutionForwardAlgorithm(mCudnnHandle, in_desc,
  // filt_desc, conv_desc, out_desc,
  //                                                       algo_count,
  //                                                       &res_algo_count,
  //                                                       algos);

  //       if (err == CUDNN_STATUS_ALLOC_FAILED)
  //       {
  //           res_algo_count  = 1;
  //           algos[0].algo   = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  //           algos[0].status = CUDNN_STATUS_SUCCESS;
  //           algos[0].memory = 0;
  //           algos[0].time   = -1;
  //       }

  //       assert(res_algo_count > 0);
  //       assert(algos[0].status == CUDNN_STATUS_SUCCESS);

  //       // Best algo is the first.
  //       best_algo_       = algos[0].algo;
  //       workspace_bytes_ = algos[0].memory;
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      mCudnnHandle, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
  CUDNN_CALL(cudnnConvolutionForward(
      mCudnnHandle, &alpha, in_desc, inputs[0], filt_desc, inputs[1], conv_desc,
      algo, workspace, ws_size, &beta, out_desc, outputs[0]));

  return 0;
}
} // namespace nvinfer1

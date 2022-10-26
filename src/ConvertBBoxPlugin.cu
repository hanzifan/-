#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>


#include <algorithm>
#include <cstdint>

#include "./cuda_utils.h"
#include "ConvertBBoxPlugin.h"

namespace nvinfer1 {
DimsExprs ConvertBBoxPlugin::getOutputDimensions(int outputIndex,
                                                 const DimsExprs *inputs,
                                                 int nbInputs,
                                                 IExprBuilder &exprBuilder) {
  assert(nbInputs == 1);
  DimsExprs ret;
  ret.nbDims = 2;
  auto mul1 = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
  ret.d[0] = exprBuilder.constant(4);
  ret.d[1] = exprBuilder.operation(DimensionOperation::kPROD, *mul1, *exprBuilder.constant(_anchors.size() / 4));
  return ret;
  // return inputs[0];
}
bool ConvertBBoxPlugin::supportsFormatCombination(int pos,
                                                  const PluginTensorDesc *inOut,
                                                  int nbInputs, int nbOutputs) {
  // 1 inputs, 1 outputs, so 2 input/output in total
  assert(0 <= pos && pos < 2);
  const auto *in = inOut;
  const auto *out = inOut + nbInputs;
  const bool consistentFloatPrecision = (in[0].type == in[pos].type);
  switch (pos) {
  case 0:
    return in[0].type == DataType::kFLOAT &&
           in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
  case 1:
    return out[0].type == DataType::kFLOAT &&
           out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
  }
  return false;
}

size_t ConvertBBoxPlugin::getWorkspaceSize(const PluginTensorDesc *inputs,
                                           int nbInputs,
                                           const PluginTensorDesc *outputs,
                                           int nbOutputs) const {
  // Return required scratch space size cub style
  size_t workspace_size = get_size_aligned<float>(_anchors.size()); // anchors
  return workspace_size;
}
__global__ void DecodeDelta(int score_size, const float *in_boxes, float *output, float *anchors_d, int width, int height, int num_anchors, uint8_t stride)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < score_size) {
    // get index of inputs
    int x = idx % width;
    int y = (idx / width) % height;
    int a = (idx / height / width) % num_anchors;
    //
    float4 box = float4{in_boxes[((0 * num_anchors + a) * height + y) * width + x],
                        in_boxes[((1 * num_anchors + a) * height + y) * width + x],
                        in_boxes[((2 * num_anchors + a) * height + y) * width + x],
                        in_boxes[((3 * num_anchors + a) * height + y) * width + x]};
    // Add anchors offsets to deltas
    int half_w = width/2;
    int half_h = height/2;
    // printf("$ %d ss: %d", stride, half_w);
    int xx = (x - half_w) * stride;
    int yy = (y - half_h) * stride;
    float *d = anchors_d + 4 * a;
    float w = d[2] - d[0];
    float h = d[3] - d[1];
    // if(idx==0) {
    //   printf("# %d xx: %d", xx, yy);
    //   printf("# %f wh: %f", w, h);
    //   printf("#d %f %f %f %f $\n", box.x, box.y, box.z, box.w);
    // }
    float pred_ctr_x = box.x * w + xx;
    float pred_ctr_y = box.y * h + yy;
    float pred_w = exp(box.z) * w;
    float pred_h = exp(box.w) * h;
    output[((0 * num_anchors + a) * height + y) * width + x] = pred_ctr_x;
    output[((1 * num_anchors + a) * height + y) * width + x] = pred_ctr_y;
    output[((2 * num_anchors + a) * height + y) * width + x] = pred_w;
    output[((3 * num_anchors + a) * height + y) * width + x] = pred_h;
    }
   
}

int ConvertBBoxPlugin::enqueue(const PluginTensorDesc *inputDesc,
                               const PluginTensorDesc *outputDesc,
                               const void *const *inputs, void *const *outputs,
                               void *workspace, cudaStream_t stream) {
  Dims input_dims = inputDesc[0].dims;
  int32_t height = input_dims.d[2];
  int32_t width = input_dims.d[3];
  size_t workspace_size = getWorkspaceSize(inputDesc, 1, outputDesc, 1);
  size_t num_anchors = _anchors.size() / 4;
  int scores_size = num_anchors * width * height;
  auto anchors_d =
      get_next_ptr<float>(_anchors.size(), workspace, workspace_size);
  cudaMemcpyAsync(anchors_d, _anchors.data(),
                  _anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice,
                  stream);
  auto in_boxes = static_cast<const float *>(inputs[0]);
  auto out_boxes = static_cast<float *>(outputs[0]); 
  DecodeDelta <<< (scores_size + 255) / 256, 256, 0, stream >>>
      (scores_size, in_boxes, out_boxes, anchors_d, width, height, num_anchors, _stride);

  return 0;
}
PluginFieldCollection ConvertBBoxPluginCreator::mFC{};
std::vector<PluginField> ConvertBBoxPluginCreator::mPluginAttributes;
} // namespace nvinfer1

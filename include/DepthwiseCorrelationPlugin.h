#pragma once

#include <NvInfer.h>
#include <cassert>
#include <cuda.h>
#include <cudnn.h>
#include <string>
#include <vector>
#include <iostream>

using namespace nvinfer1;

#define PLUGIN_NAME "DepthwiseCorrelation"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

#define CUDNN_CALL(f)                                                          \
  {                                                                            \
    cudnnStatus_t err = (f);                                                   \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      std::cout << "Error occurred: " << err << std::endl;                 \
      std::exit(1);                                                            \
    }                                                                          \
  }

namespace nvinfer1 {

class DepthwiseCorrelationPlugin final : public IPluginV2DynamicExt {

public:
  explicit DepthwiseCorrelationPlugin() {}
  DepthwiseCorrelationPlugin(void const *data, size_t length) {}
  ~DepthwiseCorrelationPlugin() {};
  int getNbOutputs() const override { return 1; }
  DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs,
                                int nbInputs,
                                IExprBuilder &exprBuilder) override;
  int initialize() override { return 0; }
  void terminate() override {}
  size_t getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs,
                          const PluginTensorDesc *outputs, int nbOutputs) const
      noexcept override {
    return 0; 
  }
  int enqueue(const PluginTensorDesc *inputDesc,
              const PluginTensorDesc *outputDesc, const void *const *inputs,
              void *const *outputs, void *workspace,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override { return 0; }

  void serialize(void *buffer) const override {}
  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) override;
  const char *getPluginType() const override { return PLUGIN_NAME; }
  const char *getPluginVersion() const override { return PLUGIN_VERSION; }
  void destroy() override { delete this; }
  IPluginV2DynamicExt *clone() const override {
    try {
      auto *plugin = new DepthwiseCorrelationPlugin();
      plugin->setPluginNamespace(mNamespace.c_str());
      return plugin;
    } catch (const std::exception &e) {
      std::cout << "Error occurred: " << e.what() << std::endl;
    }
    return nullptr;
  }
  void setPluginNamespace(const char *libNamespace) override {
    mNamespace = libNamespace;
  }
  const char *getPluginNamespace() const override { return mNamespace.c_str(); }

  DataType getOutputDataType(int index, const DataType *inputTypes,
                             int nbInputs) const override {
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
  }
  void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                       IGpuAllocator *gpuAllocator)  override;
  void detachFromContext()  override;

  void configurePlugin(const DynamicPluginTensorDesc *inputs, int nbInputs,
                       const DynamicPluginTensorDesc *out,
                       int nbOutputs) override {

    assert(nbInputs == 2);
    assert(nbOutputs == 1);
  }

private:
  std::string mNamespace;
  cudnnHandle_t mCudnnHandle{nullptr};
  cudnnTensorDescriptor_t in_desc;
  cudnnFilterDescriptor_t filt_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnTensorDescriptor_t out_desc;
};

class DepthwiseCorrelationPluginCreator : public IPluginCreator {
public:
  DepthwiseCorrelationPluginCreator() {}

  ~DepthwiseCorrelationPluginCreator() override = default;

  const char *getPluginName() const override { return PLUGIN_NAME; }

  const char *getPluginVersion() const override { return PLUGIN_VERSION; }

  const PluginFieldCollection *getFieldNames() override { return nullptr; }
  IPluginV2DynamicExt *createPlugin(const char *name,
                          const PluginFieldCollection *fc) override {
    try {
      auto *plugin = new DepthwiseCorrelationPlugin();
      return plugin;
    } catch (const std::exception &e) {
      std::cout << "Error occurred: " << e.what() << std::endl;
    }
    return nullptr;
  }
  IPluginV2DynamicExt *deserializePlugin(const char *name,
                                         const void *serialData,
                                         size_t serialLength) override{
    try {
      return new DepthwiseCorrelationPlugin(serialData, serialLength);
    } catch (const std::exception &e) {
      std::cout << "Error occurred: " << e.what() << std::endl;
    }
    return nullptr;
  }

  void setPluginNamespace(const char *N) override {}
  const char *getPluginNamespace() const override { return PLUGIN_NAMESPACE; }
};


REGISTER_TENSORRT_PLUGIN(DepthwiseCorrelationPluginCreator);

} // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE

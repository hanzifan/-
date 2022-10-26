#pragma once

#include <NvInfer.h>
#include <cassert>
#include <string.h>
#include <iostream>
#include <string>
#include <vector>

using namespace nvinfer1;

#define PLUGIN_NAME "ConvertBBoxPlugin"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {

class ConvertBBoxPlugin final : public IPluginV2DynamicExt {

public:
  explicit ConvertBBoxPlugin(std::vector<float> const &anchors, int stride)
      : _anchors(anchors), _stride(stride) {}
  ConvertBBoxPlugin(void const *data, size_t length) {
    const char *d = static_cast<const char *>(data);
    size_t anchors_size;
    read(d, anchors_size);
    while (anchors_size--) {
      float val;
      read(d, val);
      _anchors.push_back(val);
    }
    read(d, _stride);
  }
  ConvertBBoxPlugin() = delete;
  ~ConvertBBoxPlugin() {}
  int getNbOutputs() const override { return 1; }
  DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs,
                                int nbInputs,
                                IExprBuilder &exprBuilder) override;
  int initialize() override { return 0; }
  virtual void terminate() override{};
  size_t getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs,
                          const PluginTensorDesc *outputs,
                          int nbOutputs) const override;
  int enqueue(const PluginTensorDesc *inputDesc,
              const PluginTensorDesc *outputDesc, const void *const *inputs,
              void *const *outputs, void *workspace,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override {
    return  sizeof(size_t) + sizeof(float) * _anchors.size() + sizeof(_stride);
  }

  void serialize(void *buffer) const override {
    char *d = static_cast<char *>(buffer);
    write(d, _anchors.size());
    for (auto &val : _anchors) {
      write(d, val);
    }
    write(d, _stride);
  }
  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) override;
  const char *getPluginType() const override { return PLUGIN_NAME; }
  const char *getPluginVersion() const override { return PLUGIN_VERSION; }
  void destroy() override { delete this; }
  IPluginV2DynamicExt *clone() const override {
    try {
      auto *plugin = new ConvertBBoxPlugin(_anchors, _stride);
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
    assert(inputTypes && nbInputs > 0);
    return inputTypes[0];
  }
  void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                       IGpuAllocator *gpuAllocator) override {}
  void detachFromContext() override {}

  void configurePlugin(const DynamicPluginTensorDesc *inputs, int nbInputs,
                       const DynamicPluginTensorDesc *out,
                       int nbOutputs) override {

    assert(nbInputs == 1);
    assert(nbOutputs == 1);
  }

private:
  template <typename T> void write(char *&buffer, const T &val) const {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
  }

  template <typename T> void read(const char *&buffer, T &val) {
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
  }

private:
  std::string mNamespace;
  std::vector<float> _anchors;
  int _stride;
};

class ConvertBBoxPluginCreator : public IPluginCreator {
public:
  ConvertBBoxPluginCreator() {
    mPluginAttributes.emplace_back(
        PluginField("anchors", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(
        PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }

  ~ConvertBBoxPluginCreator() override = default;

  const char *getPluginName() const override { return PLUGIN_NAME; }

  const char *getPluginVersion() const override { return PLUGIN_VERSION; }

  const PluginFieldCollection *getFieldNames() override { return &mFC; }
  IPluginV2DynamicExt *createPlugin(const char *name,
                                    const PluginFieldCollection *fc) override {
    try {
      int stride = 8;
      std::vector<float> anchors;
      const PluginField *fields = fc->fields;
      for (int i = 0; i < fc->nbFields; ++i) {
        const char *attrName = fields[i].name;
        if (!strcmp(attrName, "anchors")) {
          assert(fields[i].type == PluginFieldType::kFLOAT32);
          int size = fields[i].length;
          anchors.reserve(size);
          const auto *a = static_cast<const float *>(fields[i].data);
          for (int j = 0; j < size; j++) {
            anchors.push_back(*a);
            a++;
          }
        } else if (!strcmp(attrName, "stride")) {
          assert(fields[i].type == PluginFieldType::kINT32);
          stride = *(static_cast<const int *>(fields[i].data));
          
        }
      }
      auto *plugin = new ConvertBBoxPlugin(anchors, stride);
      return plugin;
    } catch (const std::exception &e) {
      std::cout << "Error occurred: " << e.what() << std::endl;
    }
    return nullptr;
  }
  IPluginV2DynamicExt *deserializePlugin(const char *name,
                                         const void *serialData,
                                         size_t serialLength) override {
    try {
      return new ConvertBBoxPlugin(serialData, serialLength);
    } catch (const std::exception &e) {
      std::cout << "Error occurred: " << e.what() << std::endl;
    }
    return nullptr;
  }

  void setPluginNamespace(const char *libNamespace) override {
    mNamespace = libNamespace;
  }
  const char *getPluginNamespace() const override { return mNamespace.c_str(); }

private:
  std::string mNamespace;
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(ConvertBBoxPluginCreator);

} // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE

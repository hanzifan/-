#pragma once
#include "common.hpp"
#include <map>
#include <string>
#include <vector>

// IScaleLayer *addBatchNorm2d(INetworkDefinition *network,
//                             std::map<std::string, Weights> &weightMap,
//                             ITensor &input, std::string lname, float eps) {
//   float *gamma = (float *)weightMap[lname + ".weight"].values;
//   float *beta = (float *)weightMap[lname + ".bias"].values;
//   float *mean = (float *)weightMap[lname + ".running_mean"].values;
//   float *var = (float *)weightMap[lname + ".running_var"].values;
//   int len = weightMap[lname + ".running_var"].count;
//   float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
//   for (int i = 0; i < len; i++) {
//     scval[i] = gamma[i] / sqrt(var[i] + eps);
//   }
//   Weights scale{DataType::kFLOAT, scval, len};

//   float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
//   for (int i = 0; i < len; i++) {
//     shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
//   }
//   Weights shift{DataType::kFLOAT, shval, len};

//   float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
//   for (int i = 0; i < len; i++) {
//     pval[i] = 1.0;
//   }
//   Weights power{DataType::kFLOAT, pval, len};

//   weightMap[lname + ".scale"] = scale;
//   weightMap[lname + ".shift"] = shift;
//   weightMap[lname + ".power"] = power;
//   IScaleLayer *scale_1 =
//       network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
//   assert(scale_1);
//   return scale_1;
// }

IActivationLayer *basicConv2d(INetworkDefinition *network,
                              std::map<std::string, Weights> &weightMap,
                              ITensor &input, int outch, int ksize, int s,
                              int p, std::string lname, int layer_ord) {

  IConvolutionLayer *conv1 = network->addConvolutionNd(
      input, outch, DimsHW{ksize, ksize},
      weightMap[lname + std::to_string(layer_ord) + ".weight"],
      weightMap[lname + std::to_string(layer_ord) + ".bias"]);
  assert(conv1);
  conv1->setStrideNd(DimsHW{s, s});
  conv1->setPaddingNd(DimsHW{p, p});
  layer_ord += 1;
  IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0),
                                    lname + std::to_string(layer_ord), 1e-5);
  layer_ord += 1;
  IActivationLayer *relu1 =
      network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
  assert(relu1);
  return relu1;
}

ITensor *BuildAlexNet(INetworkDefinition *network,
                      std::map<std::string, Weights> &weightMap,
                      ITensor &input) {
  ITensor *out = nullptr;
  //========================
  IConvolutionLayer *conv1 = network->addConvolutionNd(
      input, 96, DimsHW{11, 11}, weightMap["backbone.features.0.weight"],
      weightMap["backbone.features.0.bias"]);
  assert(conv1);
  conv1->setStrideNd(DimsHW{2, 2});

  IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0),
                                    "backbone.features.1", 1e-5);
  // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
  IPoolingLayer *pool1 = network->addPoolingNd(*bn1->getOutput(0),
                                               PoolingType::kMAX, DimsHW{3, 3});
  assert(pool1);
  pool1->setStrideNd(DimsHW{2, 2});
  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu1 =
      network->addActivation(*pool1->getOutput(0), ActivationType::kRELU);
  assert(relu1);
  //========================
  IConvolutionLayer *conv2 =
      network->addConvolutionNd(*relu1->getOutput(0), 256, DimsHW{5, 5},
                                weightMap["backbone.features.4.weight"],
                                weightMap["backbone.features.4.bias"]);
  assert(conv2);
  IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0),
                                    "backbone.features.5", 1e-5);
  // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
  IPoolingLayer *pool2 = network->addPoolingNd(*bn2->getOutput(0),
                                               PoolingType::kMAX, DimsHW{3, 3});
  assert(pool2);
  pool2->setStrideNd(DimsHW{2, 2});
  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu2 =
      network->addActivation(*pool2->getOutput(0), ActivationType::kRELU);
  assert(relu2);
  //=====================
  IActivationLayer *relu3 =
      basicConv2d(network, weightMap, *relu2->getOutput(0), 384, 3, 1, 0, "backbone.features.",
                  8); // outchannel, kernel_size, stride, padding
  assert(relu3);
  IActivationLayer *relu4 = basicConv2d(network, weightMap, *relu3->getOutput(0), 384, 3, 1, 0,
                                        "backbone.features.", 11);
  assert(relu4);
  IConvolutionLayer *conv5 =
      network->addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3, 3},
                                weightMap["backbone.features.14.weight"],
                                weightMap["backbone.features.14.bias"]);

  assert(conv5);
  IScaleLayer *bn5 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0),
                                    "backbone.features.15", 1e-5);
  out = bn5->getOutput(0);
  return out;
}

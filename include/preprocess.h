#ifndef __PREPROCESS_H
#define __PREPROCESS_H

#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>


struct AffineMatrix{
    float value[6];
};


void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           cudaStream_t stream);

void fusion(cv::Mat &vis, cv::Mat &ir, uint8_t* img_device, cudaStream_t &stream);
#endif  // __PREPROCESS_H

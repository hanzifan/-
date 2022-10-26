#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include<iostream>
#include<fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

using namespace cv;

__global__ void Plus(uint8_t A[], uint8_t B[], int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    A[i] = 0.3*A[i] + 0.7*B[i];
}

__global__ void BGR2YUV(uint8_t r[], uint8_t g[], uint8_t b[], uint8_t y[], uint8_t u[], uint8_t v[],int n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    y[i] = 0.299*r[i] + 0.587*g[i] + 0.114*b[i];
	u[i] = -0.147*r[i]- 0.289*g[i] + 0.436*b[i];
	v[i] = 0.615*r[i] - 0.515*g[i] - 0.100*b[i];
}
__global__ void YUV2BGR(uint8_t r[], uint8_t g[], uint8_t b[], uint8_t y[], uint8_t u[], uint8_t v[],int n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
	r[i] = y[i] + 1.14 * v[i];
	g[i] = y[i] - 0.39 * u[i] - 0.58 * v[i];
	b[i] = y[i] + 2.03 * u[i];
}

void fusion(Mat &vis, Mat &ir) {
    Mat b;
    Mat g;
    Mat r;
    extractChannel(vis, b, 0);
    extractChannel(vis, g, 1);
    extractChannel(vis, r, 2);
    Mat y;
    Mat u;
    Mat v;
    uint8_t* ir_cu = nullptr;
    uint8_t* b_cu =nullptr;
    uint8_t* g_cu =nullptr;
    uint8_t* r_cu =nullptr;
    uint8_t* y_cu =nullptr;
    uint8_t* u_cu =nullptr;
    uint8_t* v_cu =nullptr;
    size_t size = vis.cols * vis.rows * 1 * sizeof(uint8_t);
    cudaMalloc((void**)&ir_cu, size);
    cudaMalloc((void**)&b_cu, size);
    cudaMalloc((void**)&g_cu, size);
    cudaMalloc((void**)&r_cu, size);
    cudaMalloc((void**)&y_cu, size);
    cudaMalloc((void**)&u_cu, size);
    cudaMalloc((void**)&v_cu, size);
    cudaMemcpy(ir_cu, ir.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cu, b.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_cu, g.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(r_cu, r.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_cu, y.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(u_cu, u.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_cu, v.data, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(256);
    dim3 dimGrid(vis.cols * vis.rows * 1 / 256);

    auto start_ff = std::chrono::system_clock::now();
    BGR2YUV<<<dimGrid, dimBlock>>>(r_cu, g_cu, b_cu, y_cu, u_cu, v_cu, vis.cols * vis.rows * 1);
    Plus<<<dimGrid, dimBlock>>>(y_cu, ir_cu, vis.cols * vis.rows * 1);
    YUV2BGR<<<dimGrid, dimBlock>>>(r_cu, g_cu, b_cu, y_cu, u_cu, v_cu, vis.cols * vis.rows * 1);
    auto end_ff = std::chrono::system_clock::now();
    std::cout << "cuda fusion time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_ff - start_ff).count() << "ns" << std::endl;

    cudaMemcpy(r.data, r_cu, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data, g_cu, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data, b_cu, size, cudaMemcpyDeviceToHost);
    cudaFree(ir_cu);cudaFree(r_cu); cudaFree(g_cu); cudaFree(b_cu); cudaFree(y_cu); cudaFree(u_cu); cudaFree(v_cu);

//     cvtColor(vis, vis, COLOR_BGR2YUV);
//     BGR2YUV(vis,vis);
//
//     Mat y_channel;
//     extractChannel(vis, y_channel, 0);
//
//     addWeighted(y_channel,0.3, ir, 0.7, 0, y_channel);

    insertChannel(b, vis, 0);
    insertChannel(g, vis, 1);
    insertChannel(r, vis, 2);


}

void initMat(cv::Mat &mat, float(*p)[3]){
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            mat.at<float>(i,j) = *(*(p+i)+j);
        }
    }
}

void transform(cv::Mat &H,cv::Mat &ir, cv::Mat result){
    cv::warpPerspective(ir, result, H, result.size(), cv::INTER_LINEAR);
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


int main() {

    std::string txt_path = "../homography.txt";
    std::vector<cv::Mat> homo_buffer;
    load_homo(txt_path, homo_buffer);
    
    Mat vis;
    Mat ir; cv::Mat ir_gray;

    auto start_read = std::chrono::system_clock::now();
    vis = imread("./visi1.jpeg");
    ir = imread("./infra1.jpeg");
    auto end_read = std::chrono::system_clock::now();
    std::cout << "read time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count() << "ms" << std::endl;

    cv::Mat H(3, 3, CV_32F);
    H = homo_buffer[0].t();

    cv::Mat result(vis);
    auto start_t = std::chrono::system_clock::now();
    transform(H, ir, result);
    auto end_t = std::chrono::system_clock::now();
    std::cout << "transform time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << "ms" << std::endl;

    // std::cout << ir.channels() << vis.channels() << std::endl;
    auto start_f = std::chrono::system_clock::now();
    cvtColor(result, ir_gray, COLOR_BGR2GRAY);
    fusion(vis, ir_gray);
    auto end_f = std::chrono::system_clock::now();
    std::cout << "fusion time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_f - start_f).count() << "ms" << std::endl;

    imshow("fusion", vis);
    imwrite("./1.jpg",vis);
//    waitKey();
//    destroyAllWindows();
/*
    auto start_read1 = std::chrono::system_clock::now();
	unsigned char* b = new unsigned char[1800*1600];
	unsigned char* g = new unsigned char[1800*1600];
	unsigned char* r = new unsigned char[1800*1600];
	unsigned char* vis_ = new unsigned char[1800*1600*3];
	unsigned char* ir_ = new unsigned char[1800*1600];
	FILE* VIS = fopen("../vis/FLIR_08871.jpg", "rb");
	fread(vis_, sizeof(unsigned char), 1800*1600*3, VIS);
	fclose(VIS);

	FILE* IR = fopen("../ir/FLIR_08871.jpg", "rb");
	fread(ir_, sizeof(unsigned char), 1800*1600*3, IR);
	fclose(IR);

    auto end_read2 = std::chrono::system_clock::now();
    std::cout << "read2 time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_read2 - start_read1).count() << "ms" << std::endl;

	for (int i = 0;i < 1800*1600; i++)//rgb文件按每个像素BGR分量以此存储，所以读取方式为b，g，r，b,g,r........
	{
		b[i] = vis_[3*i];
		g[i] = vis_[3*i+1];
		r[i] = vis_[3*i+2];
	}

	unsigned char* y = new unsigned char[1800*1600];
	unsigned char* u = new unsigned char[1800*1600];
	unsigned char* v = new unsigned char[1800*1600];
 	unsigned char* yuv = new unsigned char[1800*1600*3];
	for (int i = 0;i < 1800*1600; i++)//rgb文件按每个像素BGR分量以此存储，所以读取方式为b，g，r，b,g,r........
	{
		y[i] = 0.299*r[i] + 0.587*g[i] + 0.114*b[i];
		u[i] = -0.147*r[i]- 0.289*g[i] + 0.436*b[i];
		v[i] = 0.615*r[i] - 0.515*g[i] - 0.100*b[i];
		y[i] = 0.3 * y[i] + 0.7 * ir_[i];
		r[i] = y[i] + 1.14 * v[i];
		g[i] = y[i] - 0.39 * u[i] - 0.58 * v[i];
		b[i] = y[i] + 2.03 * u[i];
	}
	y = 0.299*r + 0.587*g + 0.114*b;
    u = -0.147*r- 0.289*g + 0.436*b;
    v = 0.615*r - 0.515*g - 0.100*b;
    y[i] = 0.3 * y[i] + 0.7 * ir_[i];

    r = y + 1.14 * v;
    g = y - 0.39 * u - 0.58 * v;
    b = y + 2.03 * u;

    auto end_read1 = std::chrono::system_clock::now();
    std::cout << "read1 time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_read1 - start_read1).count() << "ms" << std::endl;
*/
    return 0;
}
#include "preprocess.h"
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

std::mutex extract_mtx;

__global__ void warpaffine_kernel( 
    uint8_t* src, int src_line_size, int src_width, 
    int src_height, float* dst, int dst_width, 
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    //bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    //rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void preprocess_kernel_img(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream) {
    AffineMatrix s2d,d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel<<<blocks, threads, 0, stream>>>(
        src, src_width*3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2s, jobs);

}


//3 channel bgr mat type:bgrbgrbgrbgrbgrbgr
//3 channel yuv mat type:yuvyuvyuvyuvyuvyuv
//1 channel mt taype: xxxxxxxx

__global__ void Plus(uint8_t A[], uint8_t B[], int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    A[i] = 0.3*A[i] + 0.7*B[i];
}


//4:4:4
__global__ void BGR2YUV(uint8_t r[], uint8_t g[], uint8_t b[], uint8_t y[], uint8_t u[], uint8_t v[],int n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    y[i] = 0.2990*r[i] + 0.5870*g[i] + 0.1140*b[i];
	u[i] = -0.1684*r[i]- 0.3316*g[i] + 0.5*b[i] + 128;
	v[i] = 0.5000*r[i] - 0.4187*g[i] - 0.0813*b[i] + 128;
}
__global__ void YUV2BGR(uint8_t r[], uint8_t g[], uint8_t b[], uint8_t y[], uint8_t u[], uint8_t v[],int n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
	r[i] = y[i] + 1.14075 * (v[i]-128);
	g[i] = y[i] - 0.3455 * (u[i]-128) - 0.7169 * (v[i]-128);
	b[i] = y[i] + 1.7799 * (u[i]-128);
}

__global__ void BGR2MAT(uint8_t r[], uint8_t g[], uint8_t b[], uint8_t mat[], int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    mat[i*3+0] = b[i];
    mat[i*3+1] = g[i];
    mat[i*3+2] = r[i];
}

__global__ void MAT2BGR(uint8_t r[], uint8_t g[], uint8_t b[], uint8_t mat[], int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    b[i] = mat[i*3+0];
    g[i] = mat[i*3+1];
    r[i] = mat[i*3+2];
}

__global__ void COLOR2GRAY(uint8_t c3[], uint8_t c1[], int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c1[i] = (c3[i*3+0] + c3[i*3+1] + c3[i*3+2]) / 3;
}

__global__ void bgr2yuv(uint8_t bgr[], uint8_t yuv[]) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
}



//fusion and inference use the same visible frame cuda memory, so the vis_is is same to the ims_device
void fusion(cv::Mat &vis, cv::Mat &ir, uint8_t* vis_cu, cudaStream_t &stream) {
    //mat apply and extract channel <1ms
    uint8_t* ir_cu = nullptr;
    uint8_t* ir_gray_cu = nullptr;
    uint8_t* b_cu =nullptr;
    uint8_t* g_cu =nullptr;
    uint8_t* r_cu =nullptr;
    uint8_t* y_cu =nullptr;
    uint8_t* u_cu =nullptr;
    uint8_t* v_cu =nullptr;
    size_t size = vis.cols * vis.rows * 1 * sizeof(uint8_t);
    

    //cuda malloc and memcpy needs 2ms
    // cudaMalloc((void**)&vis_cu, size*3);
    cudaMalloc((void**)&ir_cu, size*3);
    cudaMalloc((void**)&ir_gray_cu, size);
    cudaMalloc((void**)&b_cu, size);
    cudaMalloc((void**)&g_cu, size);
    cudaMalloc((void**)&r_cu, size);
    cudaMalloc((void**)&y_cu, size);
    cudaMalloc((void**)&u_cu, size);
    cudaMalloc((void**)&v_cu, size);
    cudaMemcpyAsync(vis_cu, vis.data, size*3, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ir_cu, ir.data, size*3, cudaMemcpyHostToDevice, stream);
    dim3 dimBlock(512);
    dim3 dimGrid(vis.cols * vis.rows * 1 / 512);

    COLOR2GRAY<<<dimGrid, dimBlock>>>(ir_cu, ir_gray_cu, vis.cols * vis.rows * 1);
    MAT2BGR<<<dimGrid, dimBlock>>>(r_cu, g_cu, b_cu, vis_cu, vis.cols * vis.rows * 1);
    BGR2YUV<<<dimGrid, dimBlock>>>(r_cu, g_cu, b_cu, y_cu, u_cu, v_cu, vis.cols * vis.rows * 1);
    Plus<<<dimGrid, dimBlock>>>(y_cu, ir_gray_cu, vis.cols * vis.rows * 1);
    YUV2BGR<<<dimGrid, dimBlock>>>(r_cu, g_cu, b_cu, y_cu, u_cu, v_cu, vis.cols * vis.rows * 1);
    BGR2MAT<<<dimGrid, dimBlock>>>(r_cu, g_cu, b_cu, vis_cu, vis.cols * vis.rows * 1);
 
    auto start_ff = std::chrono::system_clock::now();
    cudaMemcpyAsync(vis.data, vis_cu, size*3, cudaMemcpyDeviceToHost);
    auto end_ff = std::chrono::system_clock::now();
    cudaFree(ir_cu); cudaFree(ir_gray_cu); cudaFree(r_cu); cudaFree(g_cu); cudaFree(b_cu); cudaFree(y_cu); cudaFree(u_cu); cudaFree(v_cu);


    std::cout << "cuda fusion time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ff - start_ff).count() << "ms" << std::endl;
}
#include <math.h>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "SiamRPN_tracker.h"

// Anchor stride
int cfg_ANCHOR_STRIDE = 8;

// Anchor ratios
// float cfg_ANCHOR_RATIOS[5] = {0.33, 0.5, 1, 2, 3};

// Anchor scales
// int cfg_ANCHOR_SCALES[1] = {8};

// Anchor number
int cfg_ANCHOR_NUM = sizeof(cfg_ANCHOR_RATIOS) / sizeof(cfg_ANCHOR_RATIOS[0]) *
                     sizeof(cfg_ANCHOR_SCALES) / sizeof(cfg_ANCHOR_SCALES[0]);

// Scale penalty
float cfg_TRACK_PENALTY_K = 0.85;

// Window ifluence
float cfg_TRACK_WINDOW_INFLUENCE = 0.44;

// Interpolation learning rate
float cfg_TRACK_LR = 0.4;

// Exemplar size
int cfg_TRACK_EXEMPLAR_SIZE = 127;

// Instance size
int cfg_TRACK_INSTANCE_SIZE = 287;

// Base size
int cfg_TRACK_BASE_SIZE = 0;

// Context amount
float cfg_TRACK_CONTEXT_AMOUNT = 0.5;

// Long term lost search size
int cfg_TRACK_LOST_INSTANCE_SIZE = 487;

// Long term confidence low
float cfg_TRACK_CONFIDENCE_LOW = 0.75;

// Long term confidence high
float cfg_TRACK_CONFIDENCE_HIGH = 0.998;

std::vector<float> flatten(cv::Mat mat)
{
    std::vector<float> out;
    for (int j = 0; j < mat.rows; j++)
    {
        float *data = mat.ptr<float>(j);
        for (int i = 0; i < mat.cols; i++)
        {
            out.push_back(data[i]);
        }
    }
    return out;
}

std::vector<float> tile(std::vector<float> vec, int expand_scale)
{
    std::vector<float> out = vec;
    for (int i = 0; i < expand_scale - 1; i++)
    {
        out.insert(out.end(), vec.begin(), vec.end());
    }
    return out;
}

std::vector<float> extract_vector(std::vector<float> vec, int pos, int size)
{
    std::vector<float> out;
    for(int i = 0; i < size; i++){
        out.push_back(vec[pos * size + i]);
    }
    return out;
}

std::vector<float> vector_scalar_mul(std::vector<float> vec, float scalar)
{
    std::vector<float> out;
    for (auto i : vec)
        out.push_back(i * scalar);
    return out;
}

std::vector<float> scalar_vector_div(float scalar, std::vector<float> vec)
{
    std::vector<float> out;
    for (auto i : vec)
        out.push_back(scalar / i);
    return out;
}

std::vector<float> vector_add(std::vector<float> vec1, std::vector<float> vec2)
{
    std::vector<float> out;
    for (size_t i = 0; i < vec1.size(); i++)
    {
        out.push_back(vec1[i] + vec2[i]);
    }
    return out;
}

std::vector<float> vector_mul(std::vector<float> vec1, std::vector<float> vec2)
{
    std::vector<float> out;
    for (size_t i = 0; i < vec1.size(); i++)
    {
        out.push_back(vec1[i] * vec2[i]);
    }
    return out;
}

std::vector<float> vector_div(std::vector<float> vec1, std::vector<float> vec2)
{
    std::vector<float> out;
    for (size_t i = 0; i < vec1.size(); i++)
    {
        out.push_back(vec1[i] / vec2[i]);
    }
    return out;
}

std::vector<float> cal_penalty(std::vector<float> vec1, std::vector<float> vec2, float bias = 0, float weight = 0)
{
    // int iter = 0;
    std::vector<float> out;
    for (size_t i = 0; i < vec1.size(); i++)
    {
        out.push_back(exp(-(vec1[i] * vec2[i] + bias) * weight));
        // iter++;
        // if(iter <= 100)
        //     std::cout << exp(-(vec1[i] * vec2[i] + bias) * weight) << " ";
    }
    return out;
}

std::vector<float> sz_vector(std::vector<float> w, std::vector<float> h)
{
    std::vector<float> out;
    for (size_t i = 0; i < w.size(); i++)
    {
        float pad = (w[i] + h[i]) * 0.5;
        float temp = sqrt((w[i] + pad) * (h[i] + pad));
        out.push_back(temp);
    }
    return out;
}

float sz(float w, float h)
{
    float pad = (w + h) * 0.5;
    return sqrt((w + pad) * (h + pad));
}

std::vector<float> change(std::vector<float> r)
{
    std::vector<float> out;
    for (auto i : r)
        out.push_back(std::max(i, float(1.0 / i)));
    return out;
}

SiamRPN_Tracker::SiamRPN_Tracker(std::string init_engine, std::string short_engine, std::string long_engine)
{
    template_engine.putEngine(init_engine);
    track_engine.putEngine(short_engine);
    longterm_engine.putEngine(long_engine);
    anchor_num = cfg_ANCHOR_NUM;
    longterm_state = false;
    center_pos[0] = 0;
    center_pos[1] = 0;
    size[0] = 0;
    size[1] = 0;
}

void SiamRPN_Tracker::init(cv::Mat img, float *bbox)
{
    center_pos[0] = bbox[0] + (bbox[2] - 1) / 2;
    center_pos[1] = bbox[1] + (bbox[3] - 1) / 2;
    size[0] = bbox[2];
    size[1] = bbox[3];

    // calculate z crop size
    int w_z = size[0] + cfg_TRACK_CONTEXT_AMOUNT * (size[0] + size[1]);
    int h_z = size[1] + cfg_TRACK_CONTEXT_AMOUNT * (size[0] + size[1]);
    int s_z = round(sqrt(w_z * h_z));

    // calculate channel average, first 3 of hannel average is valid
    cv::Scalar channel_average_four = cv::mean(img);
    channel_average = cv::Scalar(channel_average_four[0], channel_average_four[1], channel_average_four[2]);

    // get crop
    cv::Mat z_crop = get_subwindow(img, center_pos, cfg_TRACK_EXEMPLAR_SIZE, s_z, channel_average);
    // cv::imwrite("c++.jpg", z_crop);
    // cv::Mat z_crop = cv::imread("c++.jpg");

    // convert img to tensor
    int z_crop_size = z_crop.cols * z_crop.rows * z_crop.channels();
    float *input_z = new float[z_crop_size * sizeof(float)];
    unsigned char *pimage = z_crop.data;
    int image_area = z_crop.cols * z_crop.rows;
    float *phost_b = input_z + image_area * 0;
    float *phost_g = input_z + image_area * 1;
    float *phost_r = input_z + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        phost_b[i] = pimage[0];
        phost_g[i] = pimage[1];
        phost_r[i] = pimage[2];
    }

    template_engine.init(input_z, z_crop_size, zf);
    // for(auto i : zf)
    //     std::cout << i << " ";
    // std::cout << std::endl;
}

bool SiamRPN_Tracker::track(cv::Mat img, float *bbox, float best_score)
{
    int w_z = size[0] + cfg_TRACK_CONTEXT_AMOUNT * (size[0] + size[1]);
    int h_z = size[1] + cfg_TRACK_CONTEXT_AMOUNT * (size[0] + size[1]);
    int s_z = round(sqrt(w_z * h_z));
    float scale_z = float(cfg_TRACK_EXEMPLAR_SIZE) / float(s_z);
    int instance_size = cfg_TRACK_INSTANCE_SIZE;
    if (longterm_state)
        instance_size = cfg_TRACK_LOST_INSTANCE_SIZE;
    int score_size = (instance_size - cfg_TRACK_EXEMPLAR_SIZE) / cfg_ANCHOR_STRIDE + 1 + cfg_TRACK_BASE_SIZE;
    cv::Mat my_hanning = hanning(score_size);
    cv::Mat window(my_hanning * my_hanning.t());
    std::vector<float> window_flatten = flatten(window);
    std::vector<float> window_vec = tile(window_flatten, anchor_num);
    // int iter = 0;
    // for(auto i : window_vec){
    //     iter++;
    //     std::cout << i << " ";
    //     if(iter > 100)
    //         break;
    // }

    float t_x = float(s_z) * float(instance_size) / float(cfg_TRACK_EXEMPLAR_SIZE);
    int s_x = round(t_x);
    // std::cout << "center_pos:" << center_pos[0] << " " << center_pos[1] << std::endl;
    // std::cout << "instance_size:" << instance_size << std::endl;
    // std::cout << "s_x:" << s_x << std::endl;
    // std::cout << "channel_average:" << channel_average << std::endl;
    cv::Mat x_crop = get_subwindow(img, center_pos, instance_size, s_x, channel_average);
    // cv::imwrite("long.jpg", x_crop);
    // cv::Mat x_crop = cv::imread("track.jpg");
    // cv::imshow("debug", x_crop);
    // cv::waitKey(1);
    // convert img to tensor
    int x_crop_size = x_crop.cols * x_crop.rows * x_crop.channels();
    float *input_x = new float[x_crop_size * sizeof(float)];
    unsigned char *pimage = x_crop.data;
    int image_area = x_crop.cols * x_crop.rows;
    float *phost_b = input_x + image_area * 0;
    float *phost_g = input_x + image_area * 1;
    float *phost_r = input_x + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        phost_b[i] = pimage[0];
        phost_g[i] = pimage[1];
        phost_r[i] = pimage[2];
    }

    std::vector<float> score;
    std::vector<float> pred_bbox;
    if(!longterm_state){
    // if(1){
        track_engine.track(input_x, x_crop_size, zf, score, pred_bbox);
    }
    else{
        longterm_engine.track(input_x, x_crop_size, zf, score, pred_bbox);
        std::cout << "start long term track"  << std::endl;
    }

    std::vector<float> pred_w;
    std::vector<float> pred_h;
    if(!longterm_state){
        pred_w = extract_vector(pred_bbox, 2, 2205);
        pred_h = extract_vector(pred_bbox, 3, 2205);
    }
    else{
        pred_w = extract_vector(pred_bbox, 2, 5*46*46);
        pred_h = extract_vector(pred_bbox, 3, 5*46*46);
    }

    // scale penalty
    std::vector<float> sc_sz = sz_vector(pred_w, pred_h);
    float scale_weight = sz(size[0] * scale_z, size[1] * scale_z);
    std::vector<float> sc_change = vector_scalar_mul(sc_sz, float(1.0 / scale_weight));
    std::vector<float> s_c = change(sc_change);

    // ratio penalty
    float ratio_weight = float(size[0]) / float(size[1]);
    std::vector<float> rc_div = vector_div(pred_w, pred_h);
    std::vector<float> rc_change = scalar_vector_div(ratio_weight, rc_div);
    std::vector<float> r_c = change(rc_change);
    // int iter = 0;
    // for(auto i : r_c){
    //     iter++;
    //     std::cout << i << " ";
    //     if(iter > 100)
    //         break;
    // }
    // std::cout << ratio_weight << std::endl;

    std::vector<float> penalty = cal_penalty(r_c, s_c, -1.0, float(cfg_TRACK_PENALTY_K));
    std::vector<float> pscore = vector_mul(penalty, score);

    // window
    if (!longterm_state)
    {
        pscore = vector_scalar_mul(pscore, 1 - cfg_TRACK_WINDOW_INFLUENCE);
        std::vector<float> temp = vector_scalar_mul(window_vec, cfg_TRACK_WINDOW_INFLUENCE);
        pscore = vector_add(pscore, temp);
    }
    else
    {
        pscore = vector_scalar_mul(pscore, 1 - 0.001);
        std::vector<float> temp = vector_scalar_mul(window_vec, 0.001);
        pscore = vector_add(pscore, temp);
    }
    
    int best_idx = max_element(pscore.begin(), pscore.end()) - pscore.begin();

    if(!longterm_state){
        bbox[0] = pred_bbox[best_idx + 0 * 2205] / scale_z;
        bbox[1] = pred_bbox[best_idx + 1 * 2205] / scale_z;
        bbox[2] = pred_bbox[best_idx + 2 * 2205] / scale_z;
        bbox[3] = pred_bbox[best_idx + 3 * 2205] / scale_z;
    }
    else{
        bbox[0] = pred_bbox[best_idx + 0 * 5*46*46] / scale_z;
        bbox[1] = pred_bbox[best_idx + 1 * 5*46*46] / scale_z;
        bbox[2] = pred_bbox[best_idx + 2 * 5*46*46] / scale_z;
        bbox[3] = pred_bbox[best_idx + 3 * 5*46*46] / scale_z; 
    }
    float lr = penalty[best_idx] * score[best_idx] * cfg_TRACK_LR;
    // std::cout << "lr:" << lr << std::endl;
    best_score = score[best_idx];

    // std::cout << "best score:" << best_score << std::endl;
    // std::cout << "best index:" << best_idx << std::endl;
    // std::cout << "pred_bbox:" << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3] << std::endl;

    int cx;
    int cy;
    int width;
    int height;
    if (best_score > 0.05)
    {
        if (best_score < cfg_TRACK_CONFIDENCE_LOW)
        {
            if (longterm_state)
            // if(0)
            {
                // reid but not exist
                longterm_state = true;
                cx = center_pos[0];
                cy = center_pos[1];
                width = size[0];
                height = size[1];
            }
            else
            {
                // from short term exist but confidence low
                longterm_state = false;
                cx = bbox[0] + center_pos[0];
                cy = bbox[1] + center_pos[1];
                width = round(float(size[0]) * (1.0 - lr) + float(bbox[2]) * lr);
                height = round(float(size[1]) * (1.0 - lr) + float(bbox[3]) * lr);
            }
        }
        else
        {
            longterm_state = false;
            cx = bbox[0] + center_pos[0];
            cy = bbox[1] + center_pos[1];
            width = round(float(size[0]) * (1.0 - lr) + float(bbox[2]) * lr);
            height = round(float(size[1]) * (1.0 - lr) + float(bbox[3]) * lr);
        }
    }
    else
    {
        // not exist
        longterm_state = true;
        cx = center_pos[0];
        cy = center_pos[1];
        width = size[0];
        height = size[1];
    }
    center_pos[0] = cx;
    center_pos[1] = cy;
    size[0] = width;
    size[1] = height;

    int img_shape[2] = {img.rows, img.cols};
    int temp_bbox[4];
    bbox_clip(cx, cy, width, height, img_shape, temp_bbox);
    bbox[0] = (temp_bbox[0] - temp_bbox[2] / 2);
    bbox[1] = (temp_bbox[1] - temp_bbox[3] / 2);
    bbox[2] = temp_bbox[2];
    bbox[3] = temp_bbox[3];
    // std::cout << "bbox:" << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3] << std::endl;

    if (best_score > cfg_TRACK_CONFIDENCE_HIGH)
        longterm_state = false;

    delete[] input_x;
    return 1;
}

cv::Mat SiamRPN_Tracker::hanning(int M)
{
    float pi = 3.14;
    cv::Mat hanning_window(M, 1, CV_32FC1);
    for (int i = 0; i < M; i++)
    {
        float w = 0.5 - 0.5 * cos(2 * pi * i / (M - 1));
        // std::cout << w << " ";
        hanning_window.at<float>(i, 0) = w;
    }
    return hanning_window;
}

cv::Mat SiamRPN_Tracker::get_subwindow(cv::Mat &im, int *pos, const int model_sz, const int original_sz, cv::Scalar &avg_chans)
{
    /*
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    */
    int sz = original_sz;
    int im_sz[2];
    im_sz[0] = im.rows; // y nums
    im_sz[1] = im.cols; // x nums
    // std::cout << "im_sz:" << im_sz[0] << " " << im_sz[1] << std::endl;
    float c = float((sz + 1)) / 2.0;

    int context_xmin = floor(float(pos[0]) - c + 0.5);
    int context_xmax = context_xmin + sz - 1;
    int context_ymin = floor(float(pos[1]) - c + 0.5);
    int context_ymax = context_ymin + sz - 1;
    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im_sz[1] + 1));
    int bottom_pad = int(std::max(0, context_ymax - im_sz[0] + 1));

    // std::cout << "pad:" << left_pad << " " << top_pad << " " << right_pad << " " << bottom_pad << std::endl;
    // std::cout << "sz:" << sz << std::endl;
    // std::cout << "context:" << context_xmin << " " << context_xmax << " " << context_ymin << " " << context_ymax << std::endl;

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;

    cv::Mat te_im = cv::Mat::zeros(im_sz[0] + top_pad + bottom_pad, im_sz[1] + left_pad + right_pad, CV_8UC3);;
    cv::Mat im_patch;
    if (top_pad != 0 || bottom_pad != 0 || left_pad != 0 || right_pad != 0)
    {
        // std::cout << "te_im size:" << te_im.size() << std::endl;
        cv::Rect flip1(left_pad, top_pad, im_sz[1], im_sz[0]);
        cv::Mat mask(im_sz[0], im_sz[1], im.depth(), cv::Scalar(1));
        im.copyTo(te_im(flip1), mask);
        if (top_pad != 0)
        {
            cv::Rect flip2(left_pad, 0, im_sz[1], top_pad);
            cv::Mat mat_avg_chans(top_pad, im_sz[1], CV_8UC3, avg_chans);
            mat_avg_chans.copyTo(te_im(flip2));
        }
        if (bottom_pad != 0)
        {
            cv::Rect flip2(left_pad, im_sz[0] + top_pad, im_sz[1], bottom_pad);
            cv::Mat mat_avg_chans(bottom_pad, im_sz[1], CV_8UC3, avg_chans);
            mat_avg_chans.copyTo(te_im(flip2));
        }
        if (left_pad != 0)
        {
            cv::Rect flip2(0, 0, left_pad, im_sz[0] + top_pad + bottom_pad);
            cv::Mat mat_avg_chans(im_sz[0] + top_pad + bottom_pad, left_pad, CV_8UC3, avg_chans);
            mat_avg_chans.copyTo(te_im(flip2));
        }
        if (right_pad != 0)
        {
            cv::Rect flip2(im_sz[1] + left_pad, 0, right_pad, im_sz[0] + top_pad + bottom_pad);
            cv::Mat mat_avg_chans(im_sz[0] + top_pad + bottom_pad, right_pad, CV_8UC3, avg_chans);
            mat_avg_chans.copyTo(te_im(flip2));
        }
        cv::Rect flip3(int(context_xmin), int(context_ymin), int(context_xmax + 1) - int(context_xmin), int(context_ymax + 1) - int(context_ymin));
        // std::cout << "flip3:" << flip3;
        im_patch = te_im(flip3);
        // std::cout << "impatch 1 size:" << im_patch.size() << std::endl;
    }
    else
    {
        cv::Rect flip3(int(context_xmin), int(context_ymin), int(context_xmax + 1) - int(context_xmin), int(context_ymax + 1) - int(context_ymin));
        // std::cout << "flip3:" << flip3.x << " " << flip3.y << " " << flip3.width << " " << flip3.height << " ";
        // std::cout << "flip3:" << context_xmin << " " << context_ymin << " " << context_xmax << " " << context_ymax << " ";
        im_patch = im(flip3);
        // std::cout << "impatch 2 size:" << im_patch.size() << std::endl;
        // std::cout << "im shape:" << im.size() << std::endl;
    }
 
    // std::cout << "impatch shape:" << im_patch.size() << std::endl;

    if (model_sz != original_sz){
        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
    }
    // cv::imwrite("crop.jpg", im_patch);
    // std::cout << "impatch size:" << im_patch.size() << std::endl;
    return im_patch;
}

void SiamRPN_Tracker::bbox_clip(int cx, int cy, int width, int height, int *boundary, int *result)
{
    result[0] = std::max(0, std::min(cx, boundary[1]));
    result[1] = std::max(0, std::min(cy, boundary[0]));
    result[2] = std::max(10, std::min(width, boundary[1]));
    result[3] = std::max(10, std::min(height, boundary[0]));
}
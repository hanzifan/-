#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "trtTracker.h"

class SiamRPN_Tracker{
public:
    SiamRPN_Tracker(std::string init_engine, std::string short_engine, std::string long_engine);
    //bbox is a 4-elements vector, bbox[0] is lt's x, bbox[1] is lt's y, bbox[2] is rectangle's width, bbox[3] is retangle's height
    void init(cv::Mat img, float* bbox);
    bool track(cv::Mat img, float* bbox, float best_score);

private:
    cv::Mat get_subwindow(cv::Mat &im, int *pos, const int model_sz, const int original_sz, cv::Scalar &avg_chans);
    cv::Mat hanning(int M);
    void bbox_clip(int cx, int y, int width, int height, int* boundary, int* result);

private:
    // tracker engine
    TrackerInit template_engine;
    TrtTracker track_engine;
    longTracker longterm_engine;

    // parameter
    int anchor_num;

    // if use longterm
    bool longterm_state;

    // searching region info
    int center_pos[2];
    int size[2];

    // template
    std::vector<float> zf;

    // channel average
    cv::Scalar channel_average; //scalar can just use index to find element
};
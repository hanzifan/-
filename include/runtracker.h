#ifndef __RUNTRACKING_H
#define __RUNTRACKING_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"
#include "SiamRPN_tracker.h"


#include <dirent.h>

int kcftracking(KCFTracker &tracker, cv::Mat &frame_track, bool start_tracking, int xMin = 0, int yMin = 0, int width = 0, int heigh = 0);

int siamtracking(SiamRPN_Tracker &tracker, cv::Mat &frame_track, bool start_tracking, float* init_rect, int* tb);

#endif
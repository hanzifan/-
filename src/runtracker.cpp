#include "runtracker.h"



int kcftracking(KCFTracker &tracker, cv::Mat &frame_track, bool start_tracking, int xMin, int yMin, int width, int height){

	bool SILENT = false;

	// Tracker results
	cv::Rect result;

	// First frame, give the groundtruth to the tracker
	if (!start_tracking) {
		// std::cout << "debug" << std::endl;
		tracker.init( cv::Rect(xMin, yMin, width, height), frame_track );
        // std::cout << xMin << " " << yMin << " " << width << " " << height << std::endl;
		cv::rectangle( frame_track, cv::Point( xMin, yMin ), cv::Point( xMin+width, yMin+height), cv::Scalar( 0, 255, 255 ), 1, 8 );
	}
	// Update
	else{
		result = tracker.update(frame_track);
        std::cout << result.x << " " << result.y << " " << result.width << " " << result.height << std::endl;
		cv::rectangle( frame_track, cv::Point( result.x, result.y ), cv::Point( result.x+result.width, result.y+result.height), cv::Scalar( 0, 255, 255 ), 1, 8 );
	}

	// if (!SILENT){
	// 	cv::waitKey(1);
	// 	cv::resize(frame, frame, cv::Size(640, 480), 0, 0, cv::INTER_AREA);
	// 	cv::imshow("Tracking", frame);
	// }

	return 1;
}

int siamtracking(SiamRPN_Tracker &tracker, cv::Mat &frame_track, bool start_tracking, float* init_rect, int* tb){
	float best_score = 0;
	// First frame, give the groundtruth to the tracker
	if(!start_tracking){
		tracker.init(frame_track, init_rect);
		cv::rectangle(frame_track, cv::Point(init_rect[0], init_rect[1]), cv::Point(init_rect[0]+init_rect[2], init_rect[1]+init_rect[3]), cv::Scalar( 0, 255, 255 ), 1, 8 );
		tb[0] = 0;
		tb[1] = 0;
	}
	// Update
	else{
		int* rect_ori = new int[4];
		rect_ori[0] = int(init_rect[0]);
		rect_ori[1] = int(init_rect[1]);
		rect_ori[2] = int(init_rect[2]);
		rect_ori[3] = int(init_rect[3]);
		tracker.track(frame_track, init_rect, best_score);
		cv::Rect r(init_rect[0], init_rect[1], init_rect[2], init_rect[3]);
		cv::rectangle(frame_track, r, cv::Scalar( 0, 255, 255 ), 1, 8);
		tb[0] = int((r.x + r.width) - (rect_ori[0] + rect_ori[2]));
		tb[1] = int((r.y + r.height) - (rect_ori[1] + rect_ori[3]));
		if(tb[0] > 1920)
			tb[0] = 0;
		if(tb[1] > 1080)
			tb[1] = 0;
		delete[] rect_ori;
		// std::cout << (r.x + r.width) << " " << (init_rect[0] + init_rect[2]) << std::endl;
	}

	return 1;
}

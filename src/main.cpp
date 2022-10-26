#include<iostream>
#include<opencv2/opencv.hpp>
#include<string>

#include "jetsonEncoder.h"

using namespace std;
using namespace cv;

int main(){
    VideoCapture capture;
    string filename = "/home/nvidia/ssd/code/tkDNN/demo/yolo_test.mp4";
    
    jetsonEncoder * temp = new jetsonEncoder(8554);
    jetsonEncoder * temp2 = new jetsonEncoder(8556);

    bool result = capture.open(filename);
    if(!capture.isOpened()){
        cout<<"open video failed!"<<endl;
    }
    Mat frame;// = cv::imread("/home/nvidia/ssd/data/1.jpg");
    Mat frame1 = cv::imread("/home/nvidia/ssd/data/1.jpg");
    int i = 0;
    //rtsp://127.0.0.1:8554/live

    while(1){
        capture>>frame;
        //temp->process(frame);
        temp2->process(frame1);
        // imshow("show window",frame);
        // waitKey(100);
        cout<<"i:"<<i++<<endl;
        waitKey(30);
    }
    capture.release();
    destroyAllWindows();
    return 0;
    

}
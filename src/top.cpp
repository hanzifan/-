#include <stdio.h>
#include <stdlib.h>
#include <mutex>
#include <thread>
#include <pthread.h>
#include <sys/time.h>
#include "top.h"
#include "yolov5.h"
#include "serial.h"
#include "jetsonEncoder.h"
#include <runtracker.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "imageProcess.h"
#include "camera_v4l2_cuda.h"
#include "nvrender.h"

#include <X11/Xlib.h>


std::mutex mtx_push_visbuffer;
std::mutex mtx_pop_visbuffer;

std::mutex mtx_push_showbuffer;
std::mutex mtx_pop_showbuffer;

std::mutex mtx_clear_showbuffer;

std::mutex mtx_tcp;

long getCurrentTime(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 10000 + tv.tv_usec / 1000;
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


Top::Top(){
    // set 
    detection_num = new int[1];
    detection_num[0] = 0;
    tb = new int[2];
    tb[0] = 0;
    tb[1] = 0;


    // track
    init_rect = new float[4];
    is_initialised = 0;
    start_tracking = 0;


    // udp socket
    sock = socket(AF_INET,SOCK_STREAM,0);
    if(sock<0)
    {
        printf("socket()\n");
    }  

    memset(&server_socket, 0, sizeof(server_socket));
    server_socket.sin_family = AF_INET;
    server_socket.sin_addr.s_addr = htonl(INADDR_ANY);

    server_socket.sin_port=htons(_PORT_);
    if(bind(sock,(struct sockaddr*)&server_socket,sizeof(server_socket))==-1)
    {
        printf("bind()\n");
        close(sock);
        exit(0);
    }
    if(listen(sock,_BACKLOG_)<0)
    {
        printf("listen()\n");
        close(sock);
    }

    param = new unsigned char [11];
    param[0] = 0x55;
    param[1] = 0xAA;
    param[2] = 0x01;
    param[3] = 0x00;
    param[4] = 0x00;
    param[5] = 0x00;
    param[6] = 0x00;
    param[7] = 0x00;
    param[8] = 0x00;
    param[9] = 0x00;
    param[10] = 0x00;

    rec_param = new unsigned char [18];
    rec_param[0] = 0x00;
    rec_param[1] = 0x00;
    rec_param[2] = 0x00;
    rec_param[3] = 0x00;
    rec_param[4] = 0x00;
    rec_param[5] = 0x00;
    rec_param[6] = 0x00;
    rec_param[7] = 0x00;
    rec_param[8] = 0x00;
    rec_param[9] = 0x00;
    rec_param[10] = 0x00;
    rec_param[11] = 0x00;
    rec_param[12] = 0x00;
    rec_param[13] = 0x00;
    rec_param[14] = 0x00;
    rec_param[15] = 0x00;
    rec_param[16] = 0x00;
    rec_param[17] = 0x00;


    // serial
    is_focal = 0;
    focal_rec = 1;
    is_detec_distane = 0;
    buffSenData_cam = new unsigned char [1024];
    buffRcvData_cam = new unsigned char [1024];
    buffSenData_razer = new unsigned char [1024];
    buffRcvData_razer = new unsigned char [1024];

    // mode
    mode_frame = 1;
    mode_fun = 1;
}

Top::~Top(){
  delete[] detection_num;
  delete[] tb;
  delete[] init_rect;
  delete[] param;
  delete[] rec_param;
  delete[] buffSenData_cam;
  delete[] buffRcvData_cam;
  delete[] buffSenData_razer;
  delete[] buffRcvData_razer;
}

// tcp 
void Top::tcp_rec()
{
    fcntl(client_sock, F_SETFL, O_NONBLOCK);
    while (1)
    {
      int ren_len = recv(client_sock, rec_param, 18, 0);
      if(ren_len > 0){
        decode_tcp_data();
      }
    }
    
    
}

void Top::tcp_send()
{
    while(1)
    {     
      encode_tcp_data(); 
      int ret  =  send(client_sock, param, 11, 0);  
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

void Top::encode_tcp_data(){
  param[3] = uchar(detection_num[0]);
  param[4] = uchar(tb[0] >> 8);
  param[5] = uchar(tb[0] & 0xFF);
  param[6] = uchar(tb[1] >> 8);
  param[7] = uchar(tb[1] & 0xFF);
  param[8] = buffRcvData_razer[6];
  param[9] = buffRcvData_razer[7];
  param[10] = buffRcvData_razer[8];
}

void Top::decode_tcp_data(){
  // mode_frame
  int frame_type = int(rec_param[3]);
  if(frame_type == 0)
    mode_frame = 1;
  else if(frame_type == 1)
    mode_frame = 2;
  else if(frame_type == 3)
    mode_frame = 3;
  else
    mode_frame = 1;

  // focal
  focal_rec = int(rec_param[4]);
  std::cout << focal_rec << std::endl;

  // init rect
  init_rect[0] = int(rec_param[10] << 8) + int(rec_param[11]);
  init_rect[1] = int(rec_param[12] << 8) + int(rec_param[13]);
  init_rect[2] = int(rec_param[14] << 8) + int(rec_param[15]);
  init_rect[3] = int(rec_param[16] << 8) + int(rec_param[17]);

  init_rect[0] = int(init_rect[0] * 1920 / 1280);
  init_rect[1] = int(init_rect[1] * 1080 / 720);
  init_rect[2] = int(init_rect[2] * 1920 / 1280);
  init_rect[3] = int(init_rect[3] * 1080 / 720);


  // mode_fun
  int is_detection = int(rec_param[5]);
  if(is_detection == 1)
    mode_fun = 2;
  int is_track = int(rec_param[6]);
  if(is_track == 1){
    mode_fun = 3;
    is_initialised = 0;
  }
  if(is_detection == 0 && is_track == 0)
    mode_fun = 1;

  // lazer
  is_detec_distane = int(rec_param[7]);

  // no_use
  int no_use = int(rec_param[8]);

  // no use 2
  int no_use_two = int(rec_param[9]);
}

cv::Mat Top::fusion(cv::Mat &vis, cv::Mat &ir, float gamma){
  //load homo matrix
  std::string txt_path = "../homography.txt";
  std::vector<cv::Mat> homo_buffer;
  load_homo(txt_path, homo_buffer);
  cv::Mat H = homo_buffer[focal_rec-1];

  cv::Mat mask = cv::Mat::ones(ir.rows, ir.cols, CV_8UC1);     //mask of fusion area
  cv::cuda::GpuMat ir_cu(ir);
  cv::cuda::GpuMat mask_cu(mask);
  transform(H, ir_cu, vis);
  transform(H, mask_cu, vis);
  mask_cu.download(mask);
  cv::cuda::cvtColor(ir_cu, ir_cu, CV_BGR2GRAY);
  cv::Mat y_channel;ir_cu.download(ir);                 
  cv::extractChannel(vis, y_channel, 0);
  y_channel = y_channel - gamma*y_channel.mul(mask) + gamma*ir;
  cv::insertChannel(y_channel, vis, 0);
  cv::Mat bgr(vis.rows, vis.cols, CV_8UC3);
  cv::cvtColor(vis, bgr, cv::COLOR_YUV2BGR_YUYV);

  return bgr;
}

cv::Mat Top::search_asyn(std::vector<cv::Mat> &buffer, std::vector<long> &stamp_buf, long stamp_now){
    cv::Mat result;
    for(int iter = 0; iter < stamp_buf.size(); iter++){
        if(iter != stamp_buf.size()){
            if(stamp_buf[iter] >= stamp_now){
                if((stamp_buf[iter]-stamp_now) < (stamp_now-stamp_buf[iter-1])){
                    result = buffer[iter].clone();
                    buffer.erase(buffer.begin(), buffer.begin()+iter+1); stamp_buf.erase(stamp_buf.begin(), stamp_buf.begin()+iter+1);
                    break;
                }
                else{
                    result = buffer[iter-1].clone();
                    buffer.erase(buffer.begin(), buffer.begin()+iter); stamp_buf.erase(stamp_buf.begin(), stamp_buf.begin()+iter);
                    break;
                }
            }
            else
                continue;
        }
        else{
            if(stamp_buf.size() == 1){;
                result = buffer[0].clone();
                buffer.erase(buffer.begin()); stamp_buf.erase(stamp_buf.begin());
                break;
            }
            else{
                result = buffer[stamp_buf.size()-1].clone();
                buffer.erase(buffer.begin(), buffer.begin()+stamp_buf.size()); stamp_buf.erase(stamp_buf.begin(), stamp_buf.begin()+stamp_buf.size());
                break;
            }
        }
    }

    return result;
}

void Top::find_asyn(std::vector<cv::Mat> &buffer_v, std::vector<cv::Mat> &buffer_t, std::vector<long> &stamp_v, std::vector<long> &stamp_t, cv::Mat &vis, cv::Mat &thermel){
    std::clock_t stamp_forword; //此变量表示v跟t两个buffer中第一个元素相对时间轴靠后的那个

    if(stamp_v[0] < stamp_t[0]){
        stamp_forword = stamp_t[0];
        cv::Mat v = search_asyn(buffer_v, stamp_v, stamp_forword);

        //返回并且删除不需要的

	vis = v.clone();
	thermel = buffer_t[0].clone();
        buffer_t.erase(buffer_t.begin()); stamp_t.erase(stamp_t.begin());
    }
    else{
        stamp_forword = stamp_v[0];
        cv::Mat t = search_asyn(buffer_t, stamp_t, stamp_forword);

        //返回并且删除不需要的

	vis = buffer_v[0].clone();
	thermel = t.clone();
        buffer_v.erase(buffer_v.begin()); stamp_v.erase(stamp_v.begin());
    }
    //std::cout << vis.size() << " " << thermel.size() << std::endl;
}

int Top::run(){
  // XInitThreads();


  socklen_t len = 0;
  std::cout << "------------------------------------------------" << std::endl;
  client_sock = accept(sock,(struct sockaddr*)&socket_value,&len);
  thread VisiRun(&Top::visiRun, this);
   thread ThermelRun(&Top::thermelRun, this);
  thread Vis_ori(&Top::vis_ori, this);
   thread Ir_ori(&Top::ir_ori, this);
  thread Fusion_ori(&Top::fusion_ori, this);
  // thread Only_ir(&Top::only_ir, this); 
  thread Only_vis(&Top::only_vis, this); 
  // thread Only_fusion(&Top::only_fusion, this);
  thread Tracking(&Top::tracking, this);
  thread Serial_trans(&Top::serial_transfer, this);
  thread Data_trans(&Top::data_transfer, this);
  std::thread send_serial(&Top::tcp_send, this);
  std::thread rec_serial(&Top::tcp_rec, this); 
  send_serial.detach();
  rec_serial.detach();

  // thread Display(&Top::display, this);
  // thread Keyboard(&Top::keyboard, this);
  // thread Get_capture(&Top::get_capture, &top);

  VisiRun.detach();
  ThermelRun.detach();
  Only_vis.detach(); 
  Vis_ori.detach();
  Ir_ori.detach();
  Fusion_ori.detach();
  // Only_ir.join();
  // Only_fusion.join();
  Tracking.detach();
  Serial_trans.detach();
  Data_trans.join();
  // Display.detach();
  // Keyboard.detach();
  // Get_capture.join();
}

void Top::visiRun(){
  v4l2 visi(0);
  visi.stream_init();
  visi.start_capture(buffer_visi);
  // cv::VideoCapture cap("rtspsrc location=rtsp://192.168.2.119:554/stream0 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false", cv::CAP_GSTREAMER);
  // while(1){
  //   cv::Mat frame;
  //   cap.read(frame);
	// 	buffer_visi.push_back(frame);
	// 	if(buffer_visi.size() > 10){
	// 		buffer_visi.erase(buffer_visi.begin(), buffer_visi.end()-10);
	// 	}
	// }
}
void Top::thermelRun(){
  v4l2 ir(1);
  ir.stream_init();
  ir.start_capture(buffer_thermel);
  // cv::VideoCapture cap("rtspsrc location=rtsp://admin:admin@192.168.1.108 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false", cv::CAP_GSTREAMER);
  // while(1){
  //   cv::Mat frame;
  //   cap.read(frame);
	// 	buffer_thermel.push_back(frame);
	// 	if(buffer_thermel.size() > 10){
	// 		buffer_thermel.erase(buffer_thermel.begin(), buffer_thermel.end()-10);
	// 	}
	// }
}

int Top::vis_ori(){
  while(true){
    cv::waitKey(1);
    if(mode_frame == 1 && mode_fun == 1){
      cv::waitKey(1);
       
      if(buffer_visi.empty() )
        continue;
      cv::Mat frame = buffer_visi.back();
      // mtx_pop_visbuffer.unlock();
      if(frame.rows==0 || frame.cols==0)
          continue;
        
      // std::cout << "start ori" << std::endl;
      cv::Mat bgr(frame.rows, frame.cols, CV_8UC3);
      bgr = frame;
      cv::cvtColor(frame, bgr, cv::COLOR_YUV2BGR_YUYV);
      // mtx_push_showbuffer.lock();
      show_buffer.push_back(bgr);


      //clear buffer
      buffer_visi.clear();
      stamp_visi.clear();
      buffer_thermel.clear();
      stamp_thermel.clear();
      if(show_buffer.size() > 100){
        show_buffer.clear();
      }
      // mtx_push_showbuffer.unlock();
    }
  }

  return 1;
}

int Top::ir_ori(){
  while(true){
    cv::waitKey(1);
    if(mode_frame == 2 && mode_fun == 1){
      cv::waitKey(1);
      if(buffer_thermel.empty() )
        continue;
      cv::Mat frame = buffer_thermel.back();
      std::cout << "start ori" << std::endl;
      // mtx_push_showbuffer.lock();
      show_buffer.push_back(frame);

      //clear buffer
      buffer_visi.clear();
      stamp_visi.clear();
      buffer_thermel.clear();
      stamp_thermel.clear();
      if(show_buffer.size() > 100)
        show_buffer.clear();
      // mtx_push_showbuffer.unlock();
    }
  }

  return 1;
}

int Top::fusion_ori(){
  while(true){
    cv::waitKey(1);
    if(mode_frame == 3 && mode_fun == 1){
      cv::waitKey(1);
      if(buffer_thermel.empty() || buffer_visi.empty())
        continue;
      cv::Mat vis = buffer_visi.back();
      cv::Mat ir = buffer_thermel.back();
      cv::Mat frame = fusion(vis, ir, 0.8);
      show_buffer.push_back(frame);

      //clear buffer
      buffer_visi.clear();
      stamp_visi.clear();
      buffer_thermel.clear();
      stamp_thermel.clear();
      if(show_buffer.size() > 100)
        show_buffer.clear();
    }
  }

  return 1;
}

int Top::only_vis(){
    std::string net = "../yolo4_berkeley_fp16.rt";
    imageProcessor nvProcessor(net);
    while(true){
      cv::waitKey(1);

      auto start_back = std::chrono::system_clock::now();
      if(mode_frame == 1 && mode_fun == 2){
        cv::waitKey(1);
        // std::cout << "detect" << std::endl;
        if(buffer_visi.empty() )
          continue;
        cv::Mat vis = buffer_visi.back();
        if(vis.rows==0 || vis.cols==0)
          continue;
        cv::Mat bgr(vis.rows, vis.cols, CV_8UC3);
        cv::cvtColor(vis, bgr, cv::COLOR_YUV2BGR_YUYV);
        cv::Mat showimg = nvProcessor.ProcessOnce(bgr);
        show_buffer.push_back(showimg);

        //clear buffer
        buffer_visi.clear();
        stamp_visi.clear();
        buffer_thermel.clear();
        stamp_thermel.clear();
        if(show_buffer.size() > 100)
          show_buffer.clear();
      }

      auto end_back = std::chrono::system_clock::now();
      // std::cout << "back time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_back - start_back).count() << "ms" << std::endl;
    }
  return 1;
}

int Top::only_ir(){
  int Is_First = 1;
  Yolov5 my_yolo("flir.engine"); 
  while(true){
    cv::waitKey(1);

    if(mode_frame == 2 && mode_fun == 2){
      cv::waitKey(1);
      if(buffer_thermel.empty() )
        continue;

      cv::Mat yolo_thermel = buffer_thermel.back();
      if(yolo_thermel.rows==0 || yolo_thermel.cols==0)
          continue;

      //yolo
      std::cout << "start detect" << std::endl;
      cv::Mat showimg = my_yolo.yolo(yolo_thermel, bbox_id, detection_num);
      // mtx_push_showbuffer.lock();
      show_buffer.push_back(showimg);

      //clear buffer
      buffer_visi.clear();
      stamp_visi.clear();
      buffer_thermel.clear();
      stamp_thermel.clear();
      if(show_buffer.size() > 100)
        show_buffer.clear();
      // mtx_push_showbuffer.unlock();
      auto end_back = std::chrono::system_clock::now();
    }
  }
  return 1;
}

void Top::set_focal(int f){  
}

int Top::only_fusion(){
    std::string net = "../yolo4_berkeley_fp16.rt";
    imageProcessor nvProcessor(net);
    while(true){
      cv::waitKey(1);
      if(mode_frame == 3 && mode_fun == 2){
        cv::waitKey(1);
        auto start_back = std::chrono::system_clock::now();
        if(buffer_visi.empty() || buffer_thermel.empty())
            continue;
        cv::Mat track_frame = buffer_visi.back();
        cv::Mat track_ir = buffer_thermel.back();
        cv::Mat bgr = fusion(track_frame, track_ir, 0.8);
        cv::Mat showimg = nvProcessor.ProcessOnce(bgr);
        show_buffer.push_back(showimg);

        auto end_back = std::chrono::system_clock::now();
        std::cout << "back time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_back - start_back).count() << "ms" << std::endl;

        //clear buffer
        buffer_visi.clear();
        stamp_visi.clear();
        buffer_thermel.clear();
        stamp_thermel.clear();
        if(show_buffer.size() > 100)
          show_buffer.clear();
      }
    }
}

int Top::kcf_tracking(){
  Yolov5 my_yolo("coco.engine");
  while(true){
    cv::waitKey(1);
    if(mode_frame == 1 && mode_fun == 2){
      cv::waitKey(1);
      // mtx_pop_visbuffer.lock();
      if(buffer_visi.empty())
          continue;

      cv::Mat yolo_vis = buffer_visi.back();

      if(yolo_vis.rows==0 || yolo_vis.cols==0)
      {
	        continue;
      }
      std::cout << "start detect" << std::endl;
      //yolo
      cv::Mat bgr(yolo_vis.rows, yolo_vis.cols, CV_8UC3);
      bgr = yolo_vis;
      cv::cvtColor(yolo_vis, bgr, cv::COLOR_YUV2BGR_YUYV);
      cv::Mat showimg = my_yolo.yolo(bgr, bbox_id, detection_num);
      // mtx_push_showbuffer.lock();
      show_buffer.push_back(showimg);

      //clear buffer
      buffer_visi.clear();
      stamp_visi.clear();
      buffer_thermel.clear();
      stamp_thermel.clear();
      if(show_buffer.size() > 100)
        show_buffer.clear();
      // mtx_push_showbuffer.unlock();

      auto end_back = std::chrono::system_clock::now();
    }
  }
}

//using Siam_RPN
int Top::tracking(){
  // create tracker
  std::string init_engine = "/home/nxsd/yolo/code/TensorRT-SiamRPN/engine/init.engine";
  std::string trt_engine = "/home/nxsd/yolo/code/TensorRT-SiamRPN/engine/st.engine";
  std::string lt_engine = "/home/nxsd/yolo/code/TensorRT-SiamRPN/engine/lt.engine";
  SiamRPN_Tracker tracker(init_engine, trt_engine, lt_engine);

  // run tracking
  while(true){
    // select which kind of frame to track
    cv::waitKey(1);
    if(mode_fun == 3){
      cv::waitKey(1);
      cv::Mat track_frame;
      if(mode_frame == 1){
        if(buffer_visi.empty())
          continue;
        track_frame = buffer_visi.back();
        cv::cvtColor(track_frame, track_frame, cv::COLOR_YUV2BGR_YUYV);
      }
      else if(mode_frame == 2){
        track_frame = buffer_thermel.back();
      }
      else if(mode_frame == 3){
        cv::Mat track_vis = buffer_visi.back();
        cv::Mat track_ir = buffer_thermel.back();
        cv::Mat bgr = fusion(track_frame, track_ir, 0.8);

        track_frame = bgr;
      }


      // start tracking
      std::cout << "strat track" << std::endl;
      if(is_initialised == 0){
        siamtracking(tracker, track_frame, is_initialised, init_rect, tb);
        is_initialised = 1;
        show_buffer.push_back(track_frame);
      }
      else{
        siamtracking(tracker, track_frame, is_initialised, init_rect, tb);
        show_buffer.push_back(track_frame);
      }

      //clear buffer
      buffer_visi.clear();
      stamp_visi.clear();
      buffer_thermel.clear();
      stamp_thermel.clear();
      if(show_buffer.size() > 100)
        show_buffer.clear();
    }
  }
}

void Top::get_capture(){
  int cap_count = 0;
  char ch = 0;
  while(1){
    std::cout << buffer_visi.size() << " " << buffer_thermel.size() << std::endl;
    if(buffer_visi.empty() || buffer_thermel.empty()){
        continue;
      }
    for(int i = 0 ; i < 500; i++){
      if(buffer_visi.empty() || buffer_thermel.empty()){
        std::cout  << "have none frame" <<std::endl;
        continue;
      }
      else{
        cap_count++;
        cv::Mat bgr(1080, 1920, CV_8UC3);
        cv::Mat rgb = buffer_visi.back();
        cv::Mat ir = buffer_thermel.back();
        // if(rgb.rows==0 || rgb.cols==0 || ir.rows==0 || ir.cols==0)
        // {
	      //   continue;
        // }
        cv::cvtColor(rgb, bgr, cv::COLOR_YUV2BGR_YUYV);
        std::string save_name = std::to_string(cap_count) + ".jpg";
        std::string vis_save_name = "../savedata/vis/" + save_name;
        std::string ir_save_name = "../savedata/ir/" + save_name;
        std::cout << "debug" << std::endl;
        cv::imwrite(vis_save_name, bgr);
        cv::imwrite(ir_save_name, ir);
        cv::waitKey(5);
      }
      break;
    }
  }
}

int Top::data_transfer(){
  // // data transfer
  // socklen_t len = 0;
  std::cout << "------------------------------------------------" << std::endl;
  // client_sock = accept(sock,(struct sockaddr*)&socket_value,&len);


  // frame transfer
  jetsonEncoder * temp = new jetsonEncoder(8554);


  while(true){
    cv::waitKey(1);
    // data transfer

    if(show_buffer.empty())
      continue;
    // mtx_pop_showbuffer.lock();
    cv::Mat frame = show_buffer.back();
    if(frame.rows == 0 || frame.cols == 0)
      continue;
    // std::cout << "start rtsp" << std::endl;
    temp->process(frame);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // mtx_pop_showbuffer.unlock();

    // mtx_clear_showbuffer.lock();
    show_buffer.clear();
    // mtx_clear_showbuffer.unlock();
  }
  close(sock);

  return 0;
}

int Top::serial_transfer(){
  // set focal serial command
  buffSenData_cam[0] = 0x81;  
  buffSenData_cam[1] = 0x01;  
  buffSenData_cam[2] = 0x04;  
  buffSenData_cam[3] = 0x47;  
  buffSenData_cam[8] = 0xFF;

  // detec distance serial command
  buffSenData_razer[0] = 0xEE;
  buffSenData_razer[1] = 0x16;
  buffSenData_razer[2] = 0x02;
  buffSenData_razer[3] = 0x03;
  buffSenData_razer[4] = 0x02;
  buffSenData_razer[5] = 0x05;

  Serial serial_viscam, serial_razer;
  serial_viscam.set_serial(1);
  // serial_razer.set_serial(2);
  while(1){
    // if(is_focal == 1){
      // std::cout << focal << " " << focal_rec << std::endl;
      if(focal_rec == 1){
        buffSenData_cam[4] = 0x00;  
        buffSenData_cam[5] = 0x00;  
        buffSenData_cam[6] = 0x00;  
        buffSenData_cam[7] = 0x00;
      }
      else if(focal_rec == 2){
        std::cout << "in" << std::endl;
        buffSenData_cam[4] = 0x01;  
        buffSenData_cam[5] = 0x06;  
        buffSenData_cam[6] = 0x08;  
        buffSenData_cam[7] = 0x00;
      }
      else if(focal_rec == 3){
        buffSenData_cam[4] = 0x02;  
        buffSenData_cam[5] = 0x00;  
        buffSenData_cam[6] = 0x04;  
        buffSenData_cam[7] = 0x00;
      }
      else if(focal_rec == 4){
        buffSenData_cam[4] = 0x02;  
        buffSenData_cam[5] = 0x06;  
        buffSenData_cam[6] = 0x04;  
        buffSenData_cam[7] = 0x00;
      }
      else if(focal_rec == 5){
        buffSenData_cam[4] = 0x02;  
        buffSenData_cam[5] = 0x0A;  
        buffSenData_cam[6] = 0x00;  
        buffSenData_cam[7] = 0x00;
      }
      serial_viscam.serial_send(buffSenData_cam, 9);
      serial_viscam.serial_recieve(buffRcvData_cam);
    //   is_focal= 0;
    // }
    // if(is_detec_distane == 1){
    //   serial_razer.serial_send(buffSenData_razer, 5);
    //   serial_razer.serial_recieve(buffRcvData_razer);
    // }
    // std::cout <<  int(buffRcvData[0]) << " " <<  int(buffRcvData[1]) << " "  <<  int(buffRcvData[2]) << " " <<  int(buffRcvData[3]) << " " 
    //           <<  int(buffRcvData[4]) << " " <<  int(buffRcvData[5]) << " " <<  int(buffRcvData[6]) << std::endl;
  }
}

void Top::keyboard(){
  std::string ch;
  while(true){
    cv::waitKey(1);
    std::cout << "start keyboard" << std::endl;
    ch = getchar();
    std::cout << ch << std::endl;
    track_id = atoi(ch.c_str());
    // track_id = 0;
    start_tracking = 1;
  }
}

void Top::display(){
  nvrenderCfg rendercfg{1920, 1080, 1280, 720, 0, 0, 0};
  nvrender *renderer = new nvrender(rendercfg);
  while (true)
  {
    cv::waitKey(1);
    std::cout << show_buffer.size() << std::endl;
    if(!show_buffer.empty()){
      //show
      auto start_show = std::chrono::system_clock::now();
      renderer->render(show_buffer.back());
      auto end_show = std::chrono::system_clock::now();
      // std::cout << "show time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_show - start_show).count() << "ms" << std::endl;
      show_buffer.clear();
    }
  }
}

//temp test 2022/4/13
void Top::yolotest(){
  Yolov5 my_yolo;
  my_yolo.yolo_test();
}

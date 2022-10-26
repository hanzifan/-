// Compiler: MSVC 19.29.30038.1
// C++ Standard: C++17
#include <iostream>
#include <unistd.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <X11/Xlib.h>
#include "top.h"
using namespace std;


// int main(){
//     unsigned char ch[2];
//     ch[0] = 0x01;
//     ch[1] = 0x2C;

//     int num = int(ch[0] << 8) + int(ch[1]);
//     std::cout << num;

//     return 0;
// }

int main(int argc, char *argv[]){
    int opt = 0;
    bool open_vis = 0; bool open_ir = 0; bool open_fusion = 0; bool get_capture = 0;; bool kcf = 0;

    Top top;
    top.run();
    // thread visiRun(&Top::visiRun, &top);
    // thread thermelRun(&Top::thermelRun, &top);
    // thread capture(&Top::get_capture, &top);

    // capture.join();
    // visiRun.detach();
    // thermelRun.detach();

	return 0;
}

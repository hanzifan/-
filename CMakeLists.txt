cmake_minimum_required(VERSION 2.6)

project(pro)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(NVINFER_VERSION 7)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CUDA_NVCC_FLAGS  ${CUDA_NVCC_FLAGS} --extended-lambda --maxrregcount=32; -G;-g;-std=c++11;) # nvcc flags
set(CUDIR /usr/src/jetson_multimedia_api/samples/common/algorithm/cuda)
set(COMMONCLS /usr/src/jetson_multimedia_api/samples/common/classes)
set(NVENCODER_SOURCE src/nvencoder/jetsonEncoder.cpp src/nvencoder/udp_publisher.cpp)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

set(COMMONCLSFILES
    ${COMMONCLS}/NvBuffer.cpp
    ${COMMONCLS}/NvElement.cpp
    ${COMMONCLS}/NvElementProfiler.cpp
    ${COMMONCLS}/NvEglRenderer.cpp
    ${COMMONCLS}/NvLogging.cpp
    ${COMMONCLS}/NvV4l2Element.cpp
    ${COMMONCLS}/NvV4l2ElementPlane.cpp
    ${COMMONCLS}/NvVideoEncoder.cpp
    ${COMMONCLS}/NvVideoDecoder.cpp
    ${COMMONCLS}/NvLogging.cpp)


include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
#v4l2
include_directories(
    ${CUDA_INCLUDE_DIRS}
    include
    include/nvencoder 
    include/tkDNN
    ./include/net/
    ./include/xop/
    ./include/3rdpart/
    /usr/src/jetson_multimedia_api/include
    /usr/src/jetson_multimedia_api/samples/common/algorithm/cuda/
    ${catkin_INCLUDE_DIRS}
    )
add_library (v4l2 SHARED src/camera_v4l2_cuda.cpp)
#serial
add_library (serial SHARED src/serial.cpp)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
# ircapture
add_library (top SHARED src/top.cpp)
# rtsp
add_library(rtsp SHARED src/jetsonEncoder.cpp)
#yolo
add_library (yolo SHARED src/yolov5.cpp)
#tracking
add_library (track SHARED src/runtracker.cpp)
add_library(siam SHARED src/SiamRPN_tracker.cpp)
add_library(trt SHARED src/trtTracker.cpp)
add_library(kcf SHARED src/kcftracker.cpp)
#nvrender
add_library (nvrender SHARED src/nvrender.cpp)
#tkDNN
if(${NVINFER_VERSION} EQUAL 7)
    include_directories(include/tkDNN7/)
elseif(${NVINFER_VERSION} EQUAL 8)
    include_directories(include/tkDNN/)
endif()
include_directories(
    ${CUDA_INCLUDE_DIRS}
    tkDNN7
    )
add_library (MYtkDNN SHARED src/imageProcess.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(myplugins SHARED src/yololayer.cu src/DepthwiseCorrelation.cu src/ConvertBBoxPlugin.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
# rtsp frame
file(GLOB 3PARTY_SRC src/3rdpart/md5/*.cpp)
file(GLOB NET_SRC src/net/*.cpp)
file(GLOB XOP_SRC src/xop/*.cpp)

cuda_add_executable(pro src/fhog.cpp src/calibrator.cpp main.cpp src/preprocess.cu src/imageProcess.cpp ${COMMONCLSFILES} ${NET_SRC} ${XOP_SRC})

# cuda_add_executable(pro ${sourcefiles} src/preprocess.cu ${COMMONCLSFILES})

target_link_libraries(pro
    -L/usr/lib/aarch64-linux-gnu/tegra/ -lnvbuf_utils
    -L/usr/lib/aarch64-linux-gnu/ -lv4l2 -lEGL -lGLESv2 -lX11 
    -L${CMAKE_CURRENT_SOURCE_DIR} -lkernels -ltkDNN
    ${CUDA_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS}
)
target_link_libraries(pro top v4l2 serial rtsp yolo track siam trt kcf MYtkDNN nvrender pthread jpeg)
target_link_libraries(pro nvinfer)
target_link_libraries(pro cudart)
target_link_libraries(pro myplugins)
target_link_libraries(pro ${OpenCV_LIBS})
target_link_libraries(pro /usr/lib/aarch64-linux-gnu/libX11.so)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)



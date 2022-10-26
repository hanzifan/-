/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <poll.h>
#include <thread>
#include <mutex>



#include "camera_v4l2_cuda.h"

std::mutex mtx_buffer;



#define MJPEG_EOS_SEARCH_SIZE 4096


using namespace std;

static const int CROP_WIDTH = 640;
static const int CROP_HEIGHT = 922;
static const int THERMAL_WIDTH = 640;
static const int THERMAL_HEIGHT = 512;

static int croped_dmabuf_fd;

v4l2::v4l2(){
}

v4l2::v4l2(const int camtype)
{
    if(camtype ==0 || camtype ==1)
        m_type = camtype;
    else
        m_type = 0;
}

v4l2::~v4l2(){
    stop_stream(&ctx);

    if (ctx.cam_fd > 0)
        close(ctx.cam_fd);

    if (ctx.renderer != NULL)
        delete ctx.renderer;

    if (ctx.egl_display && !eglTerminate(ctx.egl_display))
        printf("Failed to terminate EGL display connection\n");

    if (ctx.g_buff != NULL)
    {
        for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++) {
            if (ctx.g_buff[i].dmabuff_fd)
                NvBufferDestroy(ctx.g_buff[i].dmabuff_fd);
        }
        free(ctx.g_buff);
    }

    NvBufferDestroy(ctx.render_dmabuf_fd);
}

void v4l2::set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));

    if(m_type == 0)
        ctx->cam_devname = "/dev/video0";
    else
        ctx->cam_devname = "/dev/video1";
    ctx->cam_fd = -1;
    ctx->cam_pixfmt = V4L2_PIX_FMT_YUYV;
    ctx->cam_w = 1920;
    ctx->cam_h = 1080;
    ctx->frame = 0;
    ctx->save_n_frame = 0;

    ctx->g_buff = NULL;
    ctx->capture_dmabuf = true;
    ctx->renderer = NULL;
    ctx->fps = 30;

    ctx->enable_cuda = false;
    ctx->egl_image = NULL;
    ctx->egl_display = EGL_NO_DISPLAY;

    ctx->enable_verbose = false;
}

static nv_color_fmt nvcolor_fmt[] =
{
    /* TODO: add more pixel format mapping */
    {V4L2_PIX_FMT_UYVY, NvBufferColorFormat_UYVY},
    {V4L2_PIX_FMT_VYUY, NvBufferColorFormat_VYUY},
    {V4L2_PIX_FMT_YUYV, NvBufferColorFormat_YUYV},
    {V4L2_PIX_FMT_YVYU, NvBufferColorFormat_YVYU},
    {V4L2_PIX_FMT_GREY, NvBufferColorFormat_GRAY8},
    {V4L2_PIX_FMT_YUV420M, NvBufferColorFormat_YUV420},
};

NvBufferColorFormat v4l2::get_nvbuff_color_fmt(unsigned int v4l2_pixfmt)
{
    unsigned i;

    for (i = 0; i < sizeof(nvcolor_fmt) / sizeof(nvcolor_fmt[0]); i++)
    {
        if (v4l2_pixfmt == nvcolor_fmt[i].v4l2_pixfmt)
            return nvcolor_fmt[i].nvbuff_color;
    }

    return NvBufferColorFormat_Invalid;
}

bool v4l2::camera_initialize(context_t * ctx)
{
    struct v4l2_format fmt;

    /* Open camera device */
    ctx->cam_fd = open(ctx->cam_devname, O_RDWR);
    if (ctx->cam_fd == -1)
        ERROR_RETURN("Failed to open camera device %s: %s (%d)",
                ctx->cam_devname, strerror(errno), errno);

    /* Set camera output format */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = ctx->cam_w;
    fmt.fmt.pix.height = ctx->cam_h;
    fmt.fmt.pix.pixelformat = ctx->cam_pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(ctx->cam_fd, VIDIOC_S_FMT, &fmt) < 0)
        ERROR_RETURN("Failed to set camera output format: %s (%d)",
                strerror(errno), errno);

    /* Get the real format in case the desired is not supported */
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_G_FMT, &fmt) < 0)
        ERROR_RETURN("Failed to get camera output format: %s (%d)",
                strerror(errno), errno);
    if (fmt.fmt.pix.width != ctx->cam_w ||
            fmt.fmt.pix.height != ctx->cam_h ||
            fmt.fmt.pix.pixelformat != ctx->cam_pixfmt)
    {
        WARN("The desired format is not supported");
        ctx->cam_w = fmt.fmt.pix.width;
        ctx->cam_h = fmt.fmt.pix.height;
        ctx->cam_pixfmt =fmt.fmt.pix.pixelformat;
    }

    printf("fmt.fmt.pix.pixelformat!!!!!!!!!!!:%d\n", fmt.fmt.pix.pixelformat);

    struct v4l2_streamparm streamparm;
    memset (&streamparm, 0x00, sizeof (struct v4l2_streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl (ctx->cam_fd, VIDIOC_G_PARM, &streamparm);

    INFO("Camera ouput format: (%d x %d)  stride: %d, imagesize: %d, frate: %u / %u",
            fmt.fmt.pix.width,
            fmt.fmt.pix.height,
            fmt.fmt.pix.bytesperline,
            fmt.fmt.pix.sizeimage,
            streamparm.parm.capture.timeperframe.denominator,
            streamparm.parm.capture.timeperframe.numerator);

    return true;
}

bool v4l2::display_initialize(context_t * ctx)
{
    /* Create EGL renderer */
    ctx->renderer = NvEglRenderer::createEglRenderer("renderer0",
            ctx->cam_w/2, ctx->cam_h/2, 0, 0);
    if (!ctx->renderer)
        ERROR_RETURN("Failed to create EGL renderer");
    ctx->renderer->setFPS(ctx->fps);

    if (ctx->enable_cuda)
    {
        /* Get defalut EGL display */
        ctx->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (ctx->egl_display == EGL_NO_DISPLAY)
            ERROR_RETURN("Failed to get EGL display connection");

        /* Init EGL display connection */
        if (!eglInitialize(ctx->egl_display, NULL, NULL))
            ERROR_RETURN("Failed to initialize EGL display connection");
    }

    return true;
}

bool v4l2::camera_init_components(context_t * ctx)
{
    if (!camera_initialize(ctx))
        ERROR_RETURN("Failed to initialize camera device");

    // if (!display_initialize(ctx))
    //     ERROR_RETURN("Failed to initialize display");

    INFO("Initialize v4l2 components successfully");
    return true;
}

bool v4l2::display_init_components(context_t * ctx)
{
    if (!display_initialize(ctx))
        ERROR_RETURN("Failed to initialize display");

    INFO("Initialize v4l2 components successfully");
    return true;
}

bool v4l2::stream_init(){
    set_defaults(&ctx);

    ctx.cam_w = 1920;
    ctx.cam_h = 1080;
    ctx.cam_pixfmt = V4L2_PIX_FMT_YUYV;
    ctx.enable_cuda = true;
    ctx.enable_verbose = true;

    camera_init_components(&ctx);
    prepare_buffers(&ctx);
    start_stream(&ctx);
}

bool v4l2::display_init(){
    set_defaults(&ctx);
    ctx.cam_w = 1920;
    ctx.cam_h = 1080;
    ctx.cam_pixfmt = V4L2_PIX_FMT_YUYV;
    ctx.enable_cuda = true;
    ctx.enable_verbose = true;

    display_init_components(&ctx);

    //initial retNvbuf
    // NvBufferCreateParams bufparams = {0};
    // retNvbuf = (nv_buffer *)malloc(sizeof(nv_buffer));
    // bufparams.payloadType = NvBufferPayload_SurfArray;
    // bufparams.width = 1920;
    // bufparams.height = 1080;
    // bufparams.layout = NvBufferLayout_Pitch;
    // bufparams.colorFormat = NvBufferColorFormat_ARGB32;
    // bufparams.nvbuf_tag = NvBufferTag_NONE;
    // if (-1 == NvBufferCreateEx(&retNvbuf[0].dmabuff_fd, &bufparams))
    //         INFO("Failed to create NvBuffer 1920");
}

bool v4l2::request_camera_buff(context_t * ctx)
{
    /* Request camera v4l2 buffer */
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &rb) < 0)
        ERROR_RETURN("Failed to request v4l2 buffers: %s (%d)",
                strerror(errno), errno);
    if (rb.count != V4L2_BUFFERS_NUM)
        ERROR_RETURN("V4l2 buffer number is not as desired");

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        struct v4l2_buffer buf;

        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof buf);
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;

        if (ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &buf) < 0)
            ERROR_RETURN("Failed to query buff: %s (%d)",
                    strerror(errno), errno);

        /* TODO: add support for multi-planer
           Enqueue empty v4l2 buff into camera capture plane */
        buf.m.fd = (unsigned long)ctx->g_buff[index].dmabuff_fd;
        if (buf.length != ctx->g_buff[index].size)
        {
            WARN("Camera v4l2 buf length is not expected");
            ctx->g_buff[index].size = buf.length;
        }

        if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &buf) < 0)
            ERROR_RETURN("Failed to enqueue buffers: %s (%d)",
                    strerror(errno), errno);
    }

    return true;
}

bool v4l2::request_camera_buff_mmap(context_t * ctx)
{
    /* Request camera v4l2 buffer */
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &rb) < 0)
        ERROR_RETURN("Failed to request v4l2 buffers: %s (%d)",
                strerror(errno), errno);
    if (rb.count != V4L2_BUFFERS_NUM)
        ERROR_RETURN("V4l2 buffer number is not as desired");

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        struct v4l2_buffer buf;

        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof buf);
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &buf) < 0)
            ERROR_RETURN("Failed to query buff: %s (%d)",
                    strerror(errno), errno);

        ctx->g_buff[index].size = buf.length;
        ctx->g_buff[index].start = (unsigned char *)
            mmap (NULL /* start anywhere */,
                    buf.length,
                    PROT_READ | PROT_WRITE /* required */,
                    MAP_SHARED /* recommended */,
                    ctx->cam_fd, buf.m.offset);
        if (MAP_FAILED == ctx->g_buff[index].start)
            ERROR_RETURN("Failed to map buffers");

        if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &buf) < 0)
            ERROR_RETURN("Failed to enqueue buffers: %s (%d)",
                    strerror(errno), errno);
    }

    return true;
}

bool v4l2::prepare_buffers(context_t * ctx)
{
    NvBufferCreateParams input_params = {0};

    /* Allocate global buffer context */
    ctx->g_buff = (nv_buffer *)malloc(V4L2_BUFFERS_NUM * sizeof(nv_buffer));
    if (ctx->g_buff == NULL)
        ERROR_RETURN("Failed to allocate global buffer context");

    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = ctx->cam_w;
    input_params.height = ctx->cam_h;
    input_params.layout = NvBufferLayout_Pitch;

    /* Create buffer and provide it with camera */
    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        int fd;
        NvBufferParams params = {0};

        input_params.colorFormat = get_nvbuff_color_fmt(ctx->cam_pixfmt);
        input_params.nvbuf_tag = NvBufferTag_CAMERA;
        if (-1 == NvBufferCreateEx(&fd, &input_params))
            ERROR_RETURN("Failed to create NvBuffer");

        ctx->g_buff[index].dmabuff_fd = fd;

        if (-1 == NvBufferGetParams(fd, &params))
            ERROR_RETURN("Failed to get NvBuffer parameters");

        /* TODO: add multi-planar support
           Currently only supports YUV422 interlaced single-planar */
        if (ctx->capture_dmabuf) {
            if (-1 == NvBufferMemMap(ctx->g_buff[index].dmabuff_fd, 0, NvBufferMem_Read_Write,
                        (void**)&ctx->g_buff[index].start))
                ERROR_RETURN("Failed to map buffer");
        }
    }

    input_params.colorFormat = get_nvbuff_color_fmt(V4L2_PIX_FMT_YUV420M);
    input_params.nvbuf_tag = NvBufferTag_NONE;
    // input_params.width = 1280;
    // input_params.height = 720;

    /* Create Render buffer */
    if (-1 == NvBufferCreateEx(&ctx->render_dmabuf_fd, &input_params))
        ERROR_RETURN("Failed to create NvBuffer");

    if (ctx->capture_dmabuf) {
        if (!request_camera_buff(ctx))
            ERROR_RETURN("Failed to set up camera buff");
    } else {
        if (!request_camera_buff_mmap(ctx))
            ERROR_RETURN("Failed to set up camera buff");
    }

    input_params.colorFormat = get_nvbuff_color_fmt(ctx->cam_pixfmt);
    // input_params.colorFormat = get_nvbuff_color_fmt(V4L2_PIX_FMT_YUV420M);
    input_params.width = CROP_WIDTH;
    input_params.height = CROP_HEIGHT;
    input_params.nvbuf_tag = NvBufferTag_NONE;
    if (-1 == NvBufferCreateEx(&croped_dmabuf_fd, &input_params))
            ERROR_RETURN("Failed to create store_dmabuf_fd");

    NvBufferCreateParams bufparams = {0};
    // retNvbuf = (nv_buffer *)malloc(2*sizeof(nv_buffer));
    retNvbuf = (nv_buffer *)malloc(sizeof(nv_buffer));
    bufparams.payloadType = NvBufferPayload_SurfArray;
    bufparams.width = 1920;
    bufparams.height = 1080;
    bufparams.layout = NvBufferLayout_Pitch;
    bufparams.colorFormat = NvBufferColorFormat_ARGB32;
    bufparams.nvbuf_tag = NvBufferTag_CAMERA;
    if (-1 == NvBufferCreateEx(&retNvbuf[0].dmabuff_fd, &bufparams))
            INFO("Failed to create NvBuffer 1920");

    m_argb = cv::Mat(1080, 1920, CV_8UC2);

    INFO("Succeed in preparing stream buffers");
    return true;
}

bool v4l2::start_stream(context_t * ctx)
{
    enum v4l2_buf_type type;

    /* Start v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMON, &type) < 0)
        ERROR_RETURN("Failed to start streaming: %s (%d)",
                strerror(errno), errno);

    usleep(200);

    INFO("Camera video streaming on ...");
    return true;
}

bool v4l2::start_capture(std::vector<cv::Mat> &buffer_thermel)
{
    struct sigaction sig_action;
    struct pollfd fds[1];
    NvBufferTransformParams transParams;
    NvBufferTransformParams cropTransParams;

    /* Init the NvBufferTransformParams */
    memset(&transParams, 0, sizeof(transParams));
    memset(&cropTransParams, 0, sizeof(cropTransParams));
    transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transParams.transform_filter = NvBufferTransform_Filter_Smart;

    NvBufferRect srcRect, dstRect;
    srcRect.top = 0;
    srcRect.left = 960 - CROP_WIDTH/2;
    srcRect.width = CROP_WIDTH;
    srcRect.height = CROP_HEIGHT;

    dstRect.top = 0;
    dstRect.left = 0;
    dstRect.width = CROP_WIDTH;
    dstRect.height = CROP_HEIGHT;

    
    // transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    cropTransParams.transform_flag = NVBUFFER_TRANSFORM_CROP_SRC;
    cropTransParams.transform_filter = NvBufferTransform_Filter_Smart;
    cropTransParams.src_rect = srcRect;
    cropTransParams.dst_rect = dstRect;


    /* Enable render profiling information */
    // ctx.renderer->enableProfiling();

    fds[0].fd = ctx.cam_fd;
    fds[0].events = POLLIN;

    
    cv::Mat m_iryuv = cv::Mat(CROP_HEIGHT, CROP_WIDTH, CV_8UC2);
    /* Wait for camera event with timeout = 5000 ms */
    while (poll(fds, 1, 5000) > 0)
    {
        if (fds[0].revents & POLLIN) {
            struct v4l2_buffer v4l2_buf;
            // sdkResetTimer(&timer);
            /* Dequeue a camera buff */
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            if (ctx.capture_dmabuf)
                v4l2_buf.memory = V4L2_MEMORY_DMABUF;
            else
                v4l2_buf.memory = V4L2_MEMORY_MMAP;
            if (ioctl(ctx.cam_fd, VIDIOC_DQBUF, &v4l2_buf) < 0)
                ERROR_RETURN("Failed to dequeue camera buff: %s (%d)",
                        strerror(errno), errno);

            ctx.frame++;

            if (ctx.capture_dmabuf) {
                /* Cache sync for VIC operation since the data is from CPU */
                NvBufferMemSyncForDevice(ctx.g_buff[v4l2_buf.index].dmabuff_fd, 0,
                        (void**)&ctx.g_buff[v4l2_buf.index].start);
            } else {
                /* Copies raw buffer plane contents to an NvBuffer plane */
                Raw2NvBuffer(ctx.g_buff[v4l2_buf.index].start, 0,
                            ctx.cam_w, ctx.cam_h, ctx.g_buff[v4l2_buf.index].dmabuff_fd);
            }

            if(m_type == 1)
            {
                if (-1 == NvBufferTransform(ctx.g_buff[v4l2_buf.index].dmabuff_fd, croped_dmabuf_fd,
                            &cropTransParams))
                    ERROR_RETURN("Failed to convert the buffer");

                if(-1 == NvBuffer2Raw(croped_dmabuf_fd, 0, CROP_WIDTH, CROP_HEIGHT, m_iryuv.data))      //m_argb is 2 channel yuyv
                    ERROR_RETURN("Failed to NvBuffer2Raw");
                cv::Mat bgr(CROP_HEIGHT, CROP_WIDTH, CV_8UC3);
                cv::Mat thermalfinal(THERMAL_HEIGHT, THERMAL_WIDTH, CV_8UC3);
                thermalfinal.setTo(255);
                cv::cvtColor(m_iryuv, bgr, cv::COLOR_YUV2BGR_YUYV);

                // for(int i=0;i<120;i++)
                // printf("step::%d,", bgr.step[0]);
                // printf("\n");
                int pixelPerRow = CROP_WIDTH * 3;
                int thermalFinalRow = 0;
                for(int i=0;i<102;i++)
                {   
                    for(int ii=0;ii<5;ii++)
                    {
                        memcpy(thermalfinal.ptr(5*i+ii), bgr.ptr(1+ 9*i+2*ii), pixelPerRow); 
                        // printf("i=%d, ii=%d, thermal line:%d, input line:%d\n",i,ii,5*i+ii, 9*i+2*ii);
                    }
                }
                memcpy(thermalfinal.ptr(510), bgr.ptr(918), pixelPerRow); 
                memcpy(thermalfinal.ptr(511), bgr.ptr(922), pixelPerRow); 

                if(buffer_thermel.size())
                    buffer_thermel.clear();
                buffer_thermel.push_back(thermalfinal);
                buffer_thermel.push_back(thermalfinal);
                cv::waitKey(1);
                // cv::imwrite("thermalfinal.png", thermalfinal);
                // cv::imshow("thermalfinal", thermalfinal);
                // cv::waitKey(10);

            }
            else
            {


            /***** use opencv to display，slow！！ *****/

            // if (-1 == NvBufferTransform(ctx.g_buff[v4l2_buf.index].dmabuff_fd, retNvbuf->dmabuff_fd,
            //             &transParams))
            //     ERROR_RETURN("Failed to convert the buffer");

            // unsigned char *tmp = (unsigned char*)malloc(1920*1080*2);
            // if(-1 == NvBuffer2Raw(ctx->g_buff[v4l2_buf.index].dmabuff_fd, 0, 1920, 1080, tmp))
            //     ERROR_RETURN("Failed to NvBuffer2Raw");

            // cv::Mat mat(1080, 1920, CV_8UC4);
            if(-1 == NvBuffer2Raw(ctx.g_buff[v4l2_buf.index].dmabuff_fd, 0, 1920, 1080, m_argb.data))      //m_argb is 2 channel yuyv
                ERROR_RETURN("Failed to NvBuffer2Raw");


            // cv::Mat mat_rgb;
            // cv::cvtColor(mat, mat_rgb, cv::COLOR_RGBA2BGR);
            // mtx_buffer.lock();
            // std::cout << m_argb.rows << " " << m_argb.cols << std::endl;
            if(buffer_thermel.size())
                buffer_thermel.clear();
            buffer_thermel.push_back(m_argb);
            buffer_thermel.push_back(m_argb);
            cv::waitKey(1);
            }
            // mtx_buffer.unlock();


            /*  Convert the camera buffer from YUV422 to YUV420P */
            // if (-1 == NvBufferTransform(ctx.g_buff[v4l2_buf.index].dmabuff_fd, ctx.render_dmabuf_fd,
            //             &transParams))
            //     ERROR_RETURN("Failed to convert the buffer");



            


            // if (-1 == NvBufferTransform(nvbuf->dmabuff_fd, ctx->render_dmabuf_fd,
            //             &transParams))
            //     ERROR_RETURN("Failed to convert the yuvvvv buffer");

            /* draw black rect */
            // cuda_postprocess(ctx, ctx->render_dmabuf_fd);

            /* Preview */
            // ctx->renderer->render(ctx->render_dmabuf_fd);
            // ctx->renderer->render(nvbuf->dmabuff_fd);
            // ctx->renderer->render(img);

            /* Enqueue camera buffer back to driver */
            if (ioctl(ctx.cam_fd, VIDIOC_QBUF, &v4l2_buf))
                ERROR_RETURN("Failed to queue camera buffers: %s (%d)",
                        strerror(errno), errno);

            // printf("takes:::%f\n", sdkGetTimerValue(&timer));
        }
    }

    /* Print profiling information when streaming stops */
    // ctx.renderer->printProfilingStats();

    if (ctx.cam_pixfmt == V4L2_PIX_FMT_MJPEG)
        delete ctx.jpegdec;

    return true;
}

bool v4l2::stop_stream(context_t * ctx)
{
    enum v4l2_buf_type type;

    /* Stop v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMOFF, &type))
        ERROR_RETURN("Failed to stop streaming: %s (%d)",
                strerror(errno), errno);

    INFO("Camera video streaming off ...");
    return true;
}

bool v4l2::display(cv::Mat &mat){
    if(-1 == Raw2NvBuffer(mat.data, 0, 1920, 1080, retNvbuf->dmabuff_fd))
            ERROR_RETURN("Failed to NvBuffer2Raw");
    ctx.renderer->render(retNvbuf->dmabuff_fd);

    return true;
}



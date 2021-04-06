#ifndef IMG_DEAL_H
#define IMG_DEAL_H

#include <map>
#include <string>
#include <thread>
#include <list>
#include <mutex>
#include <iostream>
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <qstringlist.h>
#include <QDateTime>
#include <fstream>
#include <QTextStream>
#include <QFile>
#include "yololayer.h"
#include "logging.h"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include <queue>
#include <sstream>
#include <chrono>
#include <dirent.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Camera.h>
#include <std_msgs/Int32.h>


using namespace cv;
using namespace std;
using namespace cv;
using namespace nvinfer1;

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


class Tool
{
public:
    Tool();
    struct final_point
    {
        int x;
        int y;
    };
    //成员函数
    //结构体
    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection
    {
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
    Mat preprocess_img(cv::Mat& img, int shape);
    Rect get_rect(cv::Mat& img, float bbox[4]);
    void vision_engin();
    void yolov3();
    Mat vision_generateGaussMask(int width,int height);
    Mat obj_generateGaussMask(int width,int height);
    void multi_algorithm();


private:
    void doInference(IExecutionContext& context, float* input, float* output, int batchSize, int shape, int output_shape);
    int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);
    float sigmoid(float x);
    void nms(std::vector<Detection>& res, float *output, float nms_thresh);
    float iou(float lbox[4], float rbox[4]);
    //成员变量
    static const int queue_size = 5;
    static const int vision_INPUT_H = 224;
    static const int vision_INPUT_W = 224;
    static const int yolo_INPUT_H = 608;
    static const int yolo_INPUT_W = 608;
    static const int vision_OUTPUT_SIZE = 224*224;
    static const int yolo_OUTPUT_SIZE = 1000 * 7 + 1;
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";
    //互斥量
    std::mutex point_mutex;
};




class Img_deal
{
public:
    Img_deal();
    struct vision_out
    {
        int x;
        int y;
     };
    void read_img();
    void show_img();
    void recv_control_msg();
    void send_point_msg();
    void save_img();
    long getCurrentTime();
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback_seg(const sensor_msgs::ImageConstPtr& msg);
    void ros_img();
    vision_out vision_point;
    std::vector<int> route_index;

private:
    //声明互斥量
    std::mutex recv_mutex;
    std::mutex send_mutex;
    std::mutex auto_mutex;
    std::mutex point_mutex;
    //声明普通变量
    bool auto_drive=false;
    bool save=false;
    char recv_buf[20];
    char send_buf[20];
    std::string filename;

    //声明结构体
    struct sockaddr_in addr_serv;
    struct sockaddr_in addr_client;
    //声明私有成员函数
    void ResolveCtrlCmd(const QStringList &CmdList);
    void pixe_move(const QStringList &cmd_list);
    void control_modoul(const QStringList &cmd_list);
    void save_bool(const QStringList &cmd_list);
    bool img_queue();
    bool res_queue_bool();
};



#endif // IMG_DEAL_H

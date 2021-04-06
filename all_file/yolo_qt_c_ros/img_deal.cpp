#include "img_deal.h"


#define SERV_PORT   40001
#define DEST_PORT   10004
#define DSET_IP_ADDRESS  "127.0.0.1"
#define local_IP_ADDRESS  "127.0.0.1"
#define DEVICE 0  // GPU id

REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

// global
cv::Mat global_img;
vector<Mat> global_img_vec;
vector<Tool::Detection> global_res;
struct guide_point
{
    double x=640;
    double y=640;
};
guide_point manul_guide_point;
guide_point auto_guide_point;
guide_point final_auto_guide_point;
vector<guide_point> object_point;
mutex global_res_mutex;
mutex final_auto_guide_point_mutex;
mutex object_point_mutex;
mutex global_img_mute;
mutex model_name_mutex;
mutex dep_img_mutex;
cv::Mat seg_img;

long Img_deal::getCurrentTime()      //直接调用这个函数就行了，返回值最好是int64_t，long long应该也可以
{
   struct timeval tv;
   gettimeofday(&tv,NULL);    //该函数在sys/time.h头文件中
   return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}


Img_deal::Img_deal()
{
}



void Img_deal::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    Tool tool;
    global_img_mute.lock();
    global_img=cv_bridge::toCvShare(msg, "bgr8")->image;
    global_img_mute.unlock();
    if(!global_img.empty())
    {
        Mat img_dispaly;
        resize(global_img,img_dispaly,cv::Size(1280,640));
        cv::circle(img_dispaly,cv::Point(manul_guide_point.x, manul_guide_point.y),15,cv::Scalar(0,0,255));
        cv::circle(img_dispaly,cv::Point(final_auto_guide_point.x, final_auto_guide_point.y),15,cv::Scalar(255,0,0));
        if(auto_drive)
        {
            cv::putText(img_dispaly,"Auto ",cv::Point(20,55),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,0,255),4,4);
        }
        else
        {
            cv::putText(img_dispaly,"Manul ",cv::Point(20,55),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,0,255),4,4);
        }

        if(!global_res.empty())
        {
            vector<Tool::Detection> res = global_res;
//                std::cout<<res.size()<<std::endl;
            for (size_t j = 0; j < res.size(); j++)
            {
                cv::Rect r = tool.get_rect(img_dispaly, res[j].bbox);      //画矩形框
                cv::rectangle(img_dispaly, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img_dispaly, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
//                 cout<<"......."<<endl;
        }
//        cv::namedWindow("1",0);
//        cv::imshow("1", img_dispaly);
//        cv::waitKey(1);
    }
}




void Img_deal::ros_img()
{
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/carla/ego_vehicle/camera/rgb/front/image_color", 1, &Img_deal::imageCallback,this);
    ros::spin();
}



void Img_deal::show_img()
{
    Tool tool;
    cv::VideoCapture capture(0);
    while (true)
    {

        Mat frame;
        //capture>>frame;
        frame = cv::imread("/home/wl/1607684632198.png");
        global_img_mute.lock();
        global_img = frame;
        global_img_mute.unlock();
        if(!global_img.empty())
        {
            Mat img_dispaly;
            resize(global_img,img_dispaly,cv::Size(1280,640));
            if(!global_res.empty())
            {
                vector<Tool::Detection> res = global_res;
    //                std::cout<<res.size()<<std::endl;
                for (size_t j = 0; j < res.size(); j++)
                {
                    cv::Rect r = tool.get_rect(img_dispaly, res[j].bbox);      //画矩形框
                    cv::rectangle(img_dispaly, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img_dispaly, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                }
    //                 cout<<"......."<<endl;
            }
            cv::imshow("1", img_dispaly);
            cv::waitKey(1);

        }


    }
    return ;
}

//创建文件夹程序

int str2char(std::string s, char c[])
{
    size_t l = s.length();
    int i;
    for(i = 0; i < l; i++)
        c[i] = s[i];
    c[i] = '\0';
    return i;
}

void dir_file_exists(std::string dir)
{
    char des_dir[255];
    str2char(dir, des_dir); // 将string 写入到字符数组中
    int state = access(des_dir, R_OK|W_OK); // 头文件 #include <unistd.h>
    if (state!=0)
    {
        dir = "mkdir " + dir;
        str2char(dir, des_dir);
        system(des_dir); // 调用linux系统命令创建文件
    }
}


static Logger gLogger;

Tool::Tool()
{

}

Mat Tool::preprocess_img(cv::Mat& img, int shape)
{
    int w, h, x, y;
    float r_w = shape / (img.cols*1.0);    //img.cols  640  列
    float r_h = shape / (img.rows*1.0);    //img.rows  480  行
    if (r_h > r_w) {
        w = shape;
        h = r_w * img.rows;
        x = 0;
        y = (shape - h) / 2;
    } else {
        w = r_h* img.cols;
        h = shape;
        x = (shape - w) / 2;
        y = 0;
    }   //似乎就是一个裁剪操作，使得图像变为224*224.同时，此处会保证图片的长宽比不变。会有灰度扩充部分。
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(shape, shape, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

void Tool::doInference(IExecutionContext& context, float* input, float* output, int batchSize, int shape, int output_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);   //是为了获取与这个engine相关的输入输出tensor的数量
    void* buffers[2];     //void*型数组，主要用于下面GPU开辟内存。

    //获取与这个engine相关的输入输出tensor的索引。
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

     //为输入输出tensor开辟显存
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * shape * shape * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * output_shape * sizeof(float)));

    // Create stream  创建cuda流，用于管理数据复制，存取，和计算的并发操作
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    //从内存到显存，从CPU到GPU，将输入数据拷贝到显存中
    //input是读入内存中的数据；buffers[inputIndex]是显存上的存储区域，用于存放输入数据
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * shape * shape * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * output_shape * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int Tool::read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    // opendir()用来打开参数name 指定的目录, 并返回DIR*形态的目录流, 和open()类似, 接下来对目录的读取和搜索都要使用此返回值.
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}



Rect Tool::get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = yolo_INPUT_H / (img.cols * 1.0);
    float r_h = yolo_INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (yolo_INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (yolo_INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (yolo_INPUT_H - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (yolo_INPUT_H - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}


float Tool::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Tool::Detection& a, Tool::Detection& b) {
    return a.det_confidence > b.det_confidence;
}

void Tool::nms(vector<Detection>& res, float *output, float nms_thresh = NMS_THRESH) {
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++)
    {
        if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH)
            continue;
        Detection det;
        memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            if(item.class_id ==0 or item.class_id ==1 or item.class_id ==2)
              res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

void Tool::yolov3()
{
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/home/wl/yolo/yolo_bf/yolo_pro/engine/yolov3.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);     //将engine文件存到了trtModelStream中
        file.close();
    }
    static float data[3 * yolo_INPUT_H * yolo_INPUT_W];
    static float prob[yolo_OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;     //用完了 trtModelStream 就赶紧删除
    //到这里，得到的tensorrt相关的变量有：context
//    while(!global_img.empty())
    while(true)
    {
        cv::Mat img1;
        img1 = global_img;
        cv::Mat img;
        if (img1.empty())
            continue;
        cv::resize(img1, img, cv::Size(1280,640));
        cv::Mat pr_img = preprocess_img(img,yolo_INPUT_H);           //图像预处理
        for (int i = 0; i < yolo_INPUT_H * yolo_INPUT_W; i++) {
            data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[i + yolo_INPUT_H * yolo_INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[i + 2 * yolo_INPUT_H * yolo_INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }

        // Run inference
        doInference(*context, data, prob, 1,yolo_INPUT_H,yolo_OUTPUT_SIZE);            //执行推理 输入分别为：context、图片、输出、batchsize
        std::vector<Detection> res;
        nms(res, prob);
        object_point_mutex.lock();
        for(vector<Detection >::iterator iter = res.begin() ; iter!=res.end() ; ++iter)

        {
            guide_point point;
            point.x =iter->bbox[0];
            point.y =iter->bbox[1];
            object_point.push_back(point);
        }
        object_point_mutex.unlock();
        global_res_mutex.lock();
        global_res = res;
        global_res_mutex.unlock();
        usleep(50000);
    }
    context->destroy();
    engine->destroy();
    runtime->destroy();
}


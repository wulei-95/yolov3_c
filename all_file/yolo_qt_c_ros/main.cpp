#include <img_deal.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener1");
    Img_deal img_deal;


    Tool tool;
    std::thread show(&Img_deal::show_img, &img_deal);
    std::thread yolo(&Tool::yolov3, &tool);
//    std::thread ros(&Img_deal::ros_img, &img_deal);
    show.join();
    yolo.join();
//    ros.join();
    return 0;
}

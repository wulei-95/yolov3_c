QT += core
QT -= gui

CONFIG += c++11
QMAKE_CXXFLAGS += -std=c++0x
TARGET = get_img
CONFIG += console
CONFIG -= app_bundle

INCLUDEPATH += /opt/ros/kinetic/include \
                /opt/ros/indigo/include \
                /opt/ros/melodic/include \
               ./include \
               ./ROS_msg \
               /usr/local/cuda/include \
                /opt/ros/kinetic/include/opencv-3.3.1-dev \
                /home/wl/software/TensorRT-7.0.0.11/include \


DEPENDPATH += /opt/ros/indigo/include \
              /opt/ros/melodic/include \
              /opt/ros/kinetic/include

LIBS += -L/opt/ros/kinetic/lib -L/opt/ros/indigo/lib \
        -L/home/wl/software/TensorRT-7.0.0.11/lib \
        -L/home/wl/yolo/yolo_bf/yolo_pro \
        -L/usr/local/lib \
        -L/opt/ros/melodic/lib \
        -L/usr/local/lib -L/usr/lib \
        -L/usr/local/cuda/lib64  \
        -L/usr/lib/ \
        -L/home/ubuntu/yolo/yolo_pro \
        -lcudart -lcublas -lnvonnxparser -lnvinfer -lnvinfer_plugin -lnvparsers  -lyololayer

LIBS += -L/opt/ros/melodic/lib \
    -lroscpp -lrospack -lpthread -lrosconsole \
    -lrosconsole_log4cxx -lrosconsole_backend_interface \
    -lxmlrpcpp -lroscpp_serialization -lrostime \
    -lcpp_common  -lroslib -lpthread -lclass_loader -lmessage_filters -lcv_bridge -limage_transport

 -
LIBS += -L/opt/ros/kinetic/lib/x86_64-linux-gnu\
        -lopencv_highgui -lopencv_core -lopencv_imgproc\
        -lopencv_imgcodecs -lopencv_video -lopencv_videoio\
        -lopencv_videostab

SOURCES += main.cpp \
           img_deal.cpp \



DISTFILES += \
    yololayer.cu

HEADERS += \
    img_deal.h \
    logging.h \
    yololayer.h \
    Camera.h



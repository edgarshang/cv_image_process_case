#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include <imgproc.hpp>

using namespace cv;

#define COSTOM_OBJECT_DETECT_NEGATIVE "D:/project/learnOpencv/opencv_case_study/knife_detection/elec_watchzip/elec_watch/negative"
#define COSTOM_OBJECT_DETECT_POSITIVE "D:/project/learnOpencv/opencv_case_study/knife_detection/elec_watchzip/elec_watch/positive"

#define LOG_ERROR 0
#define LOG_WARN 1
#define LOG_INFO 2
#define LOG_DEBUG 3
#define LOG_VERBOSE 4

extern uint8_t LOG_VDEBUG_LEVEL;

void cv_image_log_print(const char *tag, const char *level, const char *fmt, ...);

#define LOGE(...) \
if(LOG_ERROR <= LOG_VDEBUG_LEVEL) {\
    cv_image_log_print(LOG_TAG, "E", __VA_ARGS__);\
}

#define LOGW(...) \
if(LOG_WARN <= LOG_VDEBUG_LEVEL) {\
    cv_image_log_print(LOG_TAG, "W", __VA_ARGS__);\
}

#define LOGI(...) \
if(LOG_INFO <= LOG_VDEBUG_LEVEL) {\
    cv_image_log_print(LOG_TAG, "I", __VA_ARGS__);\
}

#define LOGD(...) \
if(LOG_DEBUG <= LOG_VDEBUG_LEVEL) {\
    cv_image_log_print(LOG_TAG, "D", __VA_ARGS__);\
}

#define LOGV(...) \
if(LOG_VERBOSE <= LOG_VDEBUG_LEVEL) {\
    cv_image_log_print(LOG_TAG, "V", __VA_ARGS__);\
}

class common
{
public:
	static void sort_Rects(std::vector<cv::Rect> &rects);
};


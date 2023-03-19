#include <iostream>
#include <opencv2\opencv.hpp>
#include <imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "cv_knife_detect.h"
#include "cv_image_process.h"
#include "cv_custom_object_dectection.h"
#include "cv_qrcode_recognize.h"


int main(int argc, char** argv)
{
	std::cout << "Hello World!\n";

	string knife_path = "D:/project/learnOpencv/opencv_case_study/knife_detection/code_image/qrcode.png";

	
	//CV_Image_Process *imageProcess = new CV_Knife_Detect(knife_path, 0);
	//CV_Image_Process *imageProcess = new CustomObjectDectection(knife_path);
	CV_Image_Process *imageProcess = new CV_Qrcode_Recognize(knife_path, 1);
	

	imageProcess->image_process();

	delete imageProcess;
	
}

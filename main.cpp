#include <iostream>
#include <opencv2\opencv.hpp>
#include <imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "cv_knife_detect.h"
#include "cv_image_process.h"


int main()
{
	std::cout << "Hello World!\n";

	string path = "D:/project/learnOpencv/opencv_case_study/knife_detection/code_image/ce_01.jpg";
	
	CV_Image_Process *imageProcess = new CV_Knife_Detect(path, 1);

	imageProcess->image_process();

	delete imageProcess;
	
}

#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include <imgproc.hpp>

using namespace cv;

class common
{
public:
	static void sort_Rects(std::vector<cv::Rect> &rects);
};


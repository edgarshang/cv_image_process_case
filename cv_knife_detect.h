#pragma once
#include "cv_image_process.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include <imgproc.hpp>

using namespace cv;
using namespace std;


class CV_Knife_Detect : public CV_Image_Process
{

public:
	CV_Knife_Detect(std::string &file, int base);
	void init(void);
	void detect_defect(Mat &binary, vector<Rect> rects, vector<Rect> &defect, map<int, vector<vector<Point>>> &map_contours);
	void showResult(Mat &src, vector<Rect> &defect, map<int, vector<vector<Point>>> map_contours);
public:
	virtual void image_process();

public:
	string imagePath;
	cv::Mat src;
	cv::Mat gray;
	cv::Mat binary;
	cv::Mat tpl;
	int base_number;
	std::vector<cv::Rect> rects;
	vector<Rect> defect;
	map<int, vector<vector<Point>>> map_contours; // 残次品的轮廓点
};


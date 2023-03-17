#pragma once
#include "cv_image_process.h"
#include "common.h"

#include <opencv2\opencv.hpp>
#include <imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

class CustomObjectDectection : public CV_Image_Process
{
public:
	CustomObjectDectection(std::string &file);

public:
	virtual void image_process();

public:
	string postive_dir;
	string negative_dir;
	string imagePath;
	cv::Mat src;
	void test();
	void get_hog_descriptor(Mat &image, vector<float> &desc);
	void generate_dataset(Mat &trainData, Mat &labels);
	void svm_train(Mat &trainData, Mat &labels);
};


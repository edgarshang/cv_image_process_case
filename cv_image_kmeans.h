#pragma once
#include "cv_image_process.h"

#include <opencv2\opencv.hpp>
#include <imgproc.hpp>

using namespace std;
using namespace cv;

class CV_Image_KMeans : public CV_Image_Process
{
public:



	CV_Image_KMeans(std::string &file, int base);


	void(*kmeans_demo_pter)(void);
public:
	virtual void image_process();

public:
};


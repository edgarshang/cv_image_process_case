#pragma once
#include "cv_image_process.h"

#include <opencv2\opencv.hpp>
#include <imgproc.hpp>

using namespace cv;
using namespace std;


class CV_Qrcode_Recognize : public CV_Image_Process
{
public:
	CV_Qrcode_Recognize(std::string &file, int base);
	void scanAndDectectQRCode(Mat &image, int index);
	bool isXCorner(Mat &image);
	bool isYCorner(Mat &image);
	Mat transformCorner(Mat &image, RotatedRect &rect);

public:
	Mat src;

public:
	virtual void image_process();
};


#include "cv_custom_object_dectection.h"
#include "common.h"

#define LOG_TAG "CUSTOM_OBJECT_DECTECTION"


void CustomObjectDectection::image_process()
{
	std::cout << "hello, world" << endl;
	test();
}

CustomObjectDectection::CustomObjectDectection(std::string &file):postive_dir(COSTOM_OBJECT_DETECT_POSITIVE), 
                                                                  negative_dir(COSTOM_OBJECT_DETECT_NEGATIVE)
{
	/*std::cout << "file is " << file;*/
	LOGD("file is %s\n", file.c_str());
}

void CustomObjectDectection::test()
{
	std::cout << "test()" << endl;
	// read the image and generate dataset
	Mat trainData = Mat::zeros(Size(3780, 26), CV_32FC1);
	Mat trainLabels = Mat::zeros(Size(1, 26), CV_32SC1);
	generate_dataset(trainData, trainLabels);


	// train  SVM train adn save model

	// load model

	// detection custom object
}

void CustomObjectDectection::get_hog_descriptor(Mat &image, vector<float> &desc)
{
	HOGDescriptor hog;
	int h = image.rows;
	int w = image.cols;

	float rate = 64.0 / w;
	Mat img, gray;
	resize(image, img, Size(64, int(rate*h)));  // 这里按宽按比例缩小到64，缩小了w/64. 所以高也要按比例缩小w/64.  h / (w/64) = 64h/w

	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat result = Mat::zeros(Size(64, 128), CV_8UC1);
	result = Scalar(127);
	Rect roi;
	roi.x = 0;
	roi.width = 64;
	roi.y = (128 - gray.rows) / 2;
	roi.height = gray.rows;
	gray.copyTo(result(roi));

	hog.compute(result, desc, Size(8, 8), Size(0, 0));

	LOGD("desc len %d\n", desc.size());

}
void CustomObjectDectection::generate_dataset(Mat &trainData, Mat &labels)
{
	vector<string> images;
	glob(postive_dir, images);
	int pos_num = images.size();
	cout << "pos_num = " << pos_num;
	for (size_t i = 0; i < pos_num; i++)
	{
		Mat image = imread(images[i].c_str());
		vector<float> fv;
		get_hog_descriptor(image, fv);

		for (size_t j = 0; j < fv.size(); j++)
		{
			trainData.at<float>(i, j) = fv[j];
		}
		labels.at<int>(i, 0) = 1;
	}

	images.clear();
	glob(negative_dir, images);
	int neg_num = images.size();
	for (size_t i = 0; i < neg_num; i++)
	{
		Mat image = imread(images[i].c_str());
		vector<float> fv;
		get_hog_descriptor(image, fv);

		for (size_t j = 0; j < fv.size(); j++)
		{
			trainData.at<float>(i + pos_num, j) = fv[j];
		}
		labels.at<int>(i + pos_num, 0) = -1;
	}
}
void CustomObjectDectection::svm_train(Mat &trainData, Mat &label)
{

}

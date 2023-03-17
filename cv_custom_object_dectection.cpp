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
	//Mat trainData = Mat::zeros(Size(3780, 26), CV_32FC1);
	//Mat trainLabels = Mat::zeros(Size(1, 26), CV_32SC1);
	//generate_dataset(trainData, trainLabels);


	//// train  SVM train adn save model
	//svm_train(trainData, trainLabels);

	// load model
	Ptr<SVM> svm = SVM::load("./hog_elec.xml");

	// detection custom object
	Mat test = imread("D:/project/learnOpencv/opencv_case_study/knife_detection/elec_watchzip/elec_watch/test/scene_03.jpg");

	LOGE("before test.size() = %d %d", test.cols, test.rows);
	
	//resize(test, test, Size(0,0), 0.2, 0.2);
	resize(test, test, Size(384, 216), 0, 0);
	LOGE("after test.size() = %d %d", test.cols, test.rows);
	imshow("input", test);

	Rect winRect;
	winRect.width = 64;
	winRect.height = 128;

	int sum_x = 0;
	int sum_y = 0;
	int count = 0;

	// 开窗监测
	for (size_t row = 64; row < test.rows - 64; row+=1)
	{
		for (size_t col = 32; col < test.cols - 32; col+=1)
		{
			winRect.x = col - 32;
			winRect.y = row - 64;
			vector<float> fv;
			//get_hog_descriptor(test(winRect), fv);
			Mat testImage = test(winRect);
			get_hog_descriptor(testImage, fv);
			Mat one_row = Mat::zeros(Size(fv.size(), 1), CV_32FC1);

			/*float* fv_t = one_row.ptr<float>(0);*/
			float* one_row_t = one_row.ptr<float>(0);
			float *fv_t = &fv[0];

			int size = fv.size();
			for (size_t i = 0; i < size; i+=32)
			{
				//one_row.at<float>(0, i) = fv[i];
				//*one_row_t++ = *fv_t++;

				memcpy(one_row_t, fv_t, 32);
				one_row_t += 32;
				fv_t += 32;
			}

			float result = svm->predict(one_row);
			if (result > 0)
			{
				//rectangle(test, winRect, Scalar(0, 0, 255), 1, 8, 0);
				count += 1;
				sum_x += winRect.x;
				sum_y += winRect.y;
			}
		}
	}

	winRect.x = sum_x / count;
	winRect.y = sum_y / count;
	rectangle(test, winRect, Scalar(0, 0, 255), 2, 8, 0);

	imshow("object detection result", test);

	waitKey(0);

	return;
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
void CustomObjectDectection::svm_train(Mat &trainData, Mat &labels)
{
	LOGD("start SVM training...");
	Ptr<ml::SVM> svm = ml::SVM::create();

	svm->setC(2.67);
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setGamma(5.383);

	svm->train(trainData, ROW_SAMPLE, labels);

	clog << "...[Done]" << endl;

	LOGD("end training!");

	// save xml
	svm->save("./hog_elec.xml");
}

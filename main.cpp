#include <iostream>
#include <opencv2\opencv.hpp>
#include <imgproc.hpp>
#include <opencv2/opencv.hpp>

using  namespace cv;
using namespace std;

RNG rang(12345);
void sort_box(vector<Rect> &boxs);
void detect_defect(Mat &binary, vector<Rect> rects, vector<Rect> &defect, map<int, vector<vector<Point>>> &mask_contours);
Mat tpl;
int main()
{
	std::cout << "Hello World!\n";
	Mat src = imread("D:/project/learnOpencv/opencv_case_study/knife_detection/code_image/ce_01.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		std::cout << "image is empty!!!" << endl;
		return 0;
	}

	namedWindow("input", WINDOW_AUTOSIZE);


	// ͼ���ֵ��
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	// ��ֵ��
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	// ������Ԫ��
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));

	morphologyEx(binary, binary, MORPH_OPEN, se);

	// ��������
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Rect> rects;
	findContours(binary, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

	int height = src.rows;
	int width = src.cols;

	for (size_t i = 0; i < contours.size(); i++)
	{
		Rect rect = boundingRect(contours[i]);
		if (rect.height >= height / 2 || rect.width >= width / 2)
		{
			continue;
		}
		double area = contourArea(contours[i]);
		if (area < 150)
		{
			continue;
		}
		rects.push_back(rect);
	}
	sort_box(rects);

	// �ο���������
	tpl = binary(rects[1]);

	vector<Rect> defect;
	map<int, vector<vector<Point>>> map_contours; // �д�Ʒ��������
	detect_defect(binary, rects, defect, map_contours);

	for (size_t i = 0; i < defect.size(); i++)
	{
		rectangle(src, defect[i], Scalar(255, 0, 0), 2, 8, 0);
		putText(src, format("bad"), (defect[i].tl()), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8, false);
	}

	for (auto iter = map_contours.begin(); iter != map_contours.end(); iter++)
	{
		for (size_t i = 0; i < iter->second.size(); i++)
		{
			Mat pts;
			approxPolyDP(iter->second[i], pts, 3, true);
			for (size_t i = 0; i < pts.rows; i++)
			{
				Vec2i pt = pts.at<Vec2i>(i, 0);
				circle(src, Point(pt[0] + rects[iter->first].x, pt[1] + rects[iter->first].y), 2, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}

	imshow("input", src);
	waitKey(0);
	destroyAllWindows();
}

void detect_defect(Mat &binary, vector<Rect> rects, vector<Rect> &defect, map<int, vector<vector<Point>>> &map_contours)
{
	int h = tpl.rows;
	int w = tpl.cols;
	int size = rects.size();
	Mat mask;

	for (size_t i = 0; i < size; i++)
	{
		cout << "i = " << i << endl;
		// ����diff
		Mat roi = binary(rects[i]);

		resize(roi, roi, tpl.size());
		subtract(tpl, roi, mask);

		Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		morphologyEx(mask, mask, MORPH_OPEN, se);
		threshold(mask, mask, 0, 255, THRESH_BINARY);

		// ����diff ���Ҳ�ͬȱ��,��ֵ��
		int count = 0;
		for (size_t row = 0; row < h; row++)
		{
			for (size_t col = 0; col < w; col++)
			{
				int pv = mask.at<uchar>(row, col);
				if (pv == 255)
				{
					count++;
				}
			}
		}

		//��������,  ���һ�����ؿ�
		int mh = mask.rows + 2;
		int mw = mask.cols + 2;
		Mat m1 = Mat::zeros(Size(mw, mh), mask.type());
		Rect mroi;
		mroi.x = 1;
		mroi.y = 1;
		mroi.height = mask.rows;
		mroi.width = mask.cols;
		mask.copyTo(m1(mroi));

		// ��������
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		vector<vector<Point>> mask_contours;
		findContours(m1, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
		bool isfind = false;
		for (size_t i = 0; i < contours.size(); i++)
		{
			Rect rect = boundingRect(contours[i]);
			float ratio = ((float)rect.width) / ((float)rect.height);
			if (ratio > 4.0 && (rect.y < 5 || m1.rows - (rect.height + rect.y) < 10))
			{

				continue;
			}

			double area = contourArea(contours[i]);
			if (area > 10)
			{
				cout << format("ratio: %.2f, area : %.2f\n", ratio, area);
				isfind = true;
				mask_contours.push_back(contours[i]);

			}
		}

		// �������ش���50��ȷ�����Ǳ�Ե
		if (count > 50 && isfind)
		{
			defect.push_back(rects[i]);

			map_contours.insert(pair<int, vector<vector<Point>>>(i, mask_contours));
		}

	}

	// ���ؽ��
}

void sort_box(vector<Rect> &boxs)
{
	int size = boxs.size();
	for (size_t i = 0; i < size; i++)
	{
		for (size_t j = i; j < size; j++)
		{
			int x = boxs[j].x;
			int y = boxs[j].y;
			if (y < boxs[i].y)
			{
				Rect temp = boxs[i];
				boxs[i] = boxs[j];
				boxs[j] = temp;
			}
		}
	}
}

#include "common.h"

void common::sort_Rects(std::vector<cv::Rect> &rects)
{
	int size = rects.size();
	for (size_t i = 0; i < size; i++)
	{
		for (size_t j = i; j < size; j++)
		{
			int x = rects[j].x;
			int y = rects[j].y;
			if (y < rects[i].y)
			{
				Rect temp = rects[i];
				rects[i] = rects[j];
				rects[j] = temp;
			}
		}
	}
}

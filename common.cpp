#include "common.h"

uint8_t LOG_VDEBUG_LEVEL = LOG_DEBUG;

void cv_image_log_print(const char *tag, const char *level, const char *fmt, ...)
{
	va_list ap;
	char buf[1024];

	va_start(ap, fmt);
	vsnprintf(buf, 1024, fmt, ap);
	va_end(ap);

	printf("[%s]%s:%s\n",  level, tag, buf);
}

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

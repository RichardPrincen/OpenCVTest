#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>


using namespace cv;
using namespace std;

class MyPoint {
public:
	int x, y;
	MyPoint(int x, int y) {
		this->x = x;
		this->y = y;
	}
};

class Image : public Mat {
public:
	int radiusForCheck;
	vector<MyPoint> outOfThreshold;
	Image(/*int radiusForCheck*/) {
		//            this->radiusForCheck = radiusForCheck;
	}

};

bool houghCircle(Mat cannied, int &radius, vector<int> &xCenter, vector<int> &yCenter);

#define PI 3.1418

bool houghCircle(Mat cannied, int &radius, vector<int> &xCenter, vector<int> &yCenter) 
{
	int min_r = 80, max_r = 81/*cannied.cols/2*/;
	vector<Image> radiuses(max_r - min_r);
	for (int r = min_r; r < max_r; r++) {
		Image& image = radiuses.at(r - min_r);
		image.radiusForCheck = r;

		int** matrix = new int*[cannied.cols];
		for (int i = 0; i < cannied.cols; ++i)
			matrix[i] = new int[cannied.rows];

		for (int x = 0; x < cannied.cols; x++) {
			for (int y = 0; y < cannied.rows; y++) {
				matrix[x][y] = 0;
			}
		}
		int pointsThreshold = 30;
		for (int x = 0; x < cannied.cols; x++) {
			for (int y = 0; y < cannied.rows; y++) {
				if ((int)cannied.at<uchar>(y, x) == 255) {
					for (int k = 0; k < 360; k += 2) {
						int xCurrent = x + r*cos((double)(k*PI) / 180.00);
						int yCurrent = y + r*sin((double)(k*PI) / 180.00);
						if (xCurrent <= 0 || xCurrent >= cannied.cols - 1
							|| yCurrent <= 0 || yCurrent >= cannied.rows - 1
							|| matrix[xCurrent][yCurrent] >= pointsThreshold) {
							continue;
						}
						matrix[xCurrent][yCurrent]++;
						if (matrix[xCurrent][yCurrent] >= pointsThreshold) {
							image.outOfThreshold.push_back(MyPoint(xCurrent, yCurrent));
						}
					}
				}
			}
		}
	}
	for (int i = 0; i < radiuses.size(); i++) {
		Image image = radiuses.at(i);
		if (image.outOfThreshold.size()>0 && image.radiusForCheck >= min_r) {
			radius = image.radiusForCheck;
			for (int j = 0; j < 1; j++) {
				int x = image.outOfThreshold.at(j).x;
				int y = image.outOfThreshold.at(j).y;
				xCenter.push_back(x);
				yCenter.push_back(y);
			}
			std::cout << image.outOfThreshold.size() << "\n";
			return true;
		}
	}
	return false;
}
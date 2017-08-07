// OpenCVTest.cpp : Defines the entry point for the console application.
//
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <list>  
#include "segment.h"

#define PI 3.14159265358979323846

using namespace std;
using namespace cv;

/** Function Headers */
void detectIris(Mat frame);
list<int> LBP(Mat iris);
void segmentIris(Mat &src, Mat &dst);
Mat CannyTransform(Mat input);
Mat EdgeContour(Mat input);
Mat normalize(Mat &src, int pupilx, int pupily, int pupilRadius, int irisRadius);

/** Global variables */
String eyes_cascade_name = "haarcascade_eye.xml";
CascadeClassifier eyes_cascade2;
string window = "Output";
RNG rng(12345);


int main(int argc, const char** argv)
{
	Mat frame = imread("face10.jpg");

	if (!eyes_cascade2.load(eyes_cascade_name)) 
	{ 
		printf("Failed to load cascade\n"); 
		return -1;
	};

	detectIris(frame);
	waitKey(0);
	return 0;

}

void segmentIris(Mat &src, Mat &dst)
{
	Segment segment;
	segment.findPupilEdge(src, dst);
	segment.findIrisEdge(dst, dst);

	int pupil_x, pupil_y, pupil_r, iris_r;
	pupil_x = segment.pupil_x;
	pupil_y = segment.pupil_y;
	pupil_r = segment.pupil_r;
	iris_r = segment.iris_r;
}

void detectIris(Mat frame)
{
	//Eye localization begin
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	//equalizeHist(frame_gray, frame_gray);

	int height = frame.size().height;
	int minSize = height * 0.15;

	std::vector<Rect> eyes;
	eyes_cascade2.detectMultiScale(frame_gray, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));
	
	Rect eyeRegion(eyes[0].x, eyes[0].y, eyes[0].width, eyes[0].height);
	frame = frame_gray(eyeRegion);

	imshow(window, frame);
	waitKey(0);

	/*segmentIris(frame, frame);
	imshow(window, frame);
	waitKey(0);*/
	//destroyAllWindows();

	//Eye localization end
	//Iris localization begin

	Mat blurredFrame;
	GaussianBlur(frame, blurredFrame, Size(9, 9), 5, 5);
	imshow(window, blurredFrame);
	waitKey(0);
	destroyAllWindows();

	Mat processedFrame;
	/*= EdgeContour(blurredFrame);
	imshow(window, processedFrame);
	waitKey(0);*/

	/*Mat processedFrame = CannyTransform(blurredFrame);
	imshow("Canny", processedFrame);
	waitKey(0);
	destroyAllWindows();*/

	int minRadius = blurredFrame.size().height * 0.1;
	int maxRadius = blurredFrame.size().height * 0.3;
	cout << frame.size().height << endl;
	cout << minRadius << endl;

	Mat frame_bw;
	double highVal = threshold(blurredFrame, processedFrame , 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	double lowVal = highVal * 0.5;

	processedFrame = CannyTransform(processedFrame);
	imshow(window, processedFrame);
	waitKey(0);

	vector<Vec3f> circles;
	HoughCircles(processedFrame, circles, CV_HOUGH_GRADIENT, 1, frame_gray.rows / 8, 255, 30, 0, 0);
	int irisRadius;
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		irisRadius = cvRound(circles[i][2]);

		circle(frame, center, irisRadius, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow(window, frame);
	waitKey(0);

	//Iris localization end
	//Iris extraction begin

	Vec3f circ = circles[0];
	Mat1b mask(frame.size(), uchar(0));
	circle(mask, Point(circ[0], circ[1]), circ[2], Scalar(255), CV_FILLED);
	Rect bbox(circ[0] - circ[2], circ[1] - circ[2], 2 * circ[2], 2 * circ[2]);
	Mat iris(200, 200, CV_8UC3, Scalar(255, 255, 255));

	frame.copyTo(iris, mask);
	iris = iris(bbox);

	imshow(window, iris);
	waitKey(0);

	//Pupil location
	Mat blurredIris;
	GaussianBlur(iris, blurredIris, Size(9, 9), 5, 5);

	Mat cannyImage;
	//Mat frame_bw;
	highVal = threshold(blurredIris, blurredIris, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	lowVal = highVal * 0.5;

	cout << "Lower threshold: " << lowVal << endl << "High threshold: " << highVal << endl;

	Canny(blurredIris, cannyImage, lowVal, highVal, 3, false);
	imshow(window, cannyImage);
	waitKey(0);
	//destroyAllWindows();

	//int pupilMin = iris.size().height*0.05, pupilMax = iris.size().height*0.1;
	int pupilx, pupily, pupilRadius;
	HoughCircles(blurredIris, circles, CV_HOUGH_GRADIENT, 1, iris.rows / 8, 255 ,30, 0,0);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		pupilx = cvRound(circles[i][0]), pupily = cvRound(circles[i][1]);
		pupilRadius = cvRound(circles[i][2]);

		circle(iris, center, pupilRadius*1.1, Scalar(0, 0, 0), CV_FILLED);
	}


	imshow(window, iris);
	waitKey(0);

	Mat normalized = normalize(iris, pupilx, pupily, pupilRadius, irisRadius);

	imshow(window, normalized);
	waitKey(0);
	destroyAllWindows();

}

list<int> LBP(Mat iris)
{
	cout << "Rows:" << iris.rows / 16 << endl << "Columns: " << iris.cols/16 << endl;
	for (size_t i = 0; i < iris.rows; i++)
	{
		for (size_t j = 0; j < iris.cols; j++)
		{
			int point = (int)iris.at<char>(i, j);
			cout << point << ",";
		}
		cout << endl << endl;
	}
	list<int> memes;
	memes.push_back(1);
	return memes;
}

Mat CannyTransform(Mat input)
{
	Mat processedFrame;
	Mat frame_bw;
	double highVal = threshold(input, frame_bw , 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double lowVal = highVal * 0.3;

	cout << "Lower threshold: " << lowVal << endl << "High threshold: " << highVal << endl;

	Canny(input, processedFrame, lowVal, highVal, 3, false);
	return processedFrame;
}

Mat EdgeContour(Mat input)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat processedFrame;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Scharr(input, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	//Sobel(input, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Scharr(input, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	//Sobel(input, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, processedFrame);
	return processedFrame;
}

Mat normalize(Mat &src, int pupilx, int pupily, int pupilRadius, int irisRadius)
{
	
	int theta = 360;
	int nr = 80;
	Mat normalized = Mat(nr, theta, CV_8U, Scalar(255));
	for (int i = 0; i < theta; i++)
	{
		double alpha = 2 * PI * i / theta;
		for (int j = 0; j < nr; j++)
		{
			double r = 1.0*j / nr;
			int x = (int)((1 - r)*(pupilx + pupilRadius*cos(alpha)) + r*(pupilx + irisRadius*cos(alpha)));
			int y = (int)((1 - r)*(pupily + pupilRadius*sin(alpha)) + r*(pupily + irisRadius*sin(alpha)));
			if (y < 0)
			{
				//cout << y << endl;
				y = 0;
			}
			if (x < 0)
			{
				//cout << x << endl;
				x = 0;
			}
			normalized.at<uchar>(j, i) = src.at<uchar>(y, x);
		}
	}
	return normalized;
}
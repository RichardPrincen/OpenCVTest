// OpenCVTest.cpp : Defines the entry point for the console application.
//
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <list>  

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
list<int> LBP(Mat iris);

/** Global variables */
String eyes_cascade_name = "haarcascade_eye.xml";
CascadeClassifier eyes_cascade;
string window = "Output";
RNG rng(12345);

/** @function main */
int main(int argc, const char** argv)
{
	Mat frame = imread("face5.jpg");

	if (!eyes_cascade.load(eyes_cascade_name)) 
	{ 
		printf("Failed to load cascade\n"); 
		return -1;
	};

	detectAndDisplay(frame);
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	//Eye localization begin
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	int height = frame.size().height;
	int minSize = height * 0.15;

	std::vector<Rect> eyes;
	eyes_cascade.detectMultiScale(frame_gray, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));
	
	Rect eyeRegion(eyes[0].x, eyes[0].y, eyes[0].width, eyes[0].height);
	frame = frame_gray(eyeRegion);

	imshow(window, frame);
	waitKey(0);
	//destroyAllWindows();

	//Eye localization end
	//Iris localization begin

	Mat blurredFrame;
	GaussianBlur(frame, blurredFrame, Size(9, 9), 5, 1);
	/*imshow(window, blurredFrame);
	waitKey(0);
	destroyAllWindows();*/

	/*Mat cannyImage;
	Mat frame_bw;
	double highVal = cv::threshold(frame, frame_bw , 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double lowVal = highVal * 0.3;

	cout << "Lower threshold: " << lowVal << endl << "High threshold: " << highVal << endl;

	Canny(frame, cannyImage, lowVal, highVal, 3, false);
	imshow("Canny", cannyImage);
	waitKey(0);
	destroyAllWindows();*/

	int minRadius = blurredFrame.size().height * 0.0725;
	int maxRadius = blurredFrame.size().height * 0.3;
	cout << frame.size().height << endl;
	cout << minRadius << endl;

	vector<Vec3f> circles;
	HoughCircles(blurredFrame, circles, CV_HOUGH_GRADIENT, 1, frame_gray.rows / 8, 255, 1, minRadius, maxRadius);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		circle(frame, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow(window, frame);
	waitKey(0);
	//destroyAllWindows();

	//Iris localization end
	//Iris extraction begin

	Vec3f circ = circles[0];

	Mat1b mask(frame.size(), uchar(0));
	circle(mask, Point(circ[0], circ[1]), circ[2], Scalar(255), CV_FILLED);

	Rect bbox(circ[0] - circ[2], circ[1] - circ[2], 2 * circ[2], 2 * circ[2]);

	Mat iris(320, 240, CV_8UC3, Scalar(255, 255, 255));

	frame.copyTo(iris, mask);

	iris = iris(bbox);

	imshow(window, iris);
	waitKey(0);
	//destroyAllWindows();

	list<int> DankMemes = LBP(iris);

	//Pupil location
	Mat blurredIris;
	GaussianBlur(iris, blurredIris, Size(9, 9), 3, 1);

	Mat cannyImage;
	Mat frame_bw;
	double highVal = cv::threshold(blurredIris, frame_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double lowVal = highVal * 0.3;

	cout << "Lower threshold: " << lowVal << endl << "High threshold: " << highVal << endl;

	Canny(blurredIris, cannyImage, lowVal, highVal, 3, false);
	imshow(window, cannyImage);
	waitKey(0);
	//destroyAllWindows();

	int pupilMin = iris.size().height*0.3, pupilMax = iris.size().height*0.5;

	HoughCircles(blurredIris, circles, CV_HOUGH_GRADIENT, 1, iris.rows / 8, 255, 1, pupilMin,pupilMax);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		circle(iris, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow(window, iris);
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
// OpenCVTest.cpp : Defines the entry point for the console application.
//
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String eyes_cascade_name = "haarcascade_eye.xml";
CascadeClassifier eyes_cascade;
RNG rng(12345);

/** @function main */
int main(int argc, const char** argv)
{
	Mat frame = imread("face5.jpg");

	//-- 1. Load the cascades

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

	imshow("Detected eye region", frame);
	waitKey(0);
	destroyAllWindows();

	//Eye localization end
	//Iris localization begin

	GaussianBlur(frame, frame, Size(9, 9), 5, 1);

	imshow("Blurred", frame);
	waitKey(0);
	destroyAllWindows();

	Mat cannyImage;
	Mat frame_bw;
	double highVal = cv::threshold(frame, frame_bw , 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double lowVal = highVal * 0.3;

	cout << "Lower threshold: " << lowVal << endl << "High threshold: " << highVal << endl;

	Canny(frame, cannyImage, lowVal, highVal, 3, false);
	imshow("Canny", cannyImage);
	waitKey(0);
	destroyAllWindows();

	int minRadius = cannyImage.size().height * 0.0725;
	int maxRadius = cannyImage.size().height * 0.3;
	cout << frame.size().height << endl;
	cout << minRadius << endl;

	vector<Vec3f> circles;
	HoughCircles(frame, circles, CV_HOUGH_GRADIENT, 1, frame_gray.rows / 8, 255, 1, minRadius, maxRadius);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		circle(frame, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("Detected circles", frame);
	waitKey(0);
	destroyAllWindows();
}
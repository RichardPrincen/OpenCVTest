
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <list>  
#include "segment.h"

#define PI 3.14159265358979323846

using namespace std;
using namespace cv;

/** Function Headers */
void segmentIris(Mat &src, Mat &dst);
Mat detectIris(Mat frame);
Mat findEye(Mat input);
Mat blurImage(Mat input);
Mat CannyTransform(Mat input);
Mat EdgeContour(Mat input);
Mat findAndExtractIris(Mat &input, Mat &unprocessed, Mat &original);
Mat findPupil(Mat input);
Mat normalize(Mat input, int pupilx, int pupily, int pupilRadius, int irisRadius);
vector<int> LBP(Mat iris);
double hammingDistance(vector<int> savedCode, vector<int> inputCode);
void showCurrentImage(Mat input);

/** Global variables */
String eyes_cascade_name = "haarcascade_eye.xml";
CascadeClassifier eyes_cascade2;
string window = "Output";
int pupilx, pupily, pupilRadius, irisRadius;

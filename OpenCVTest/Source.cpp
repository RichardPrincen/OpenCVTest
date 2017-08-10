// OpenCVTest.cpp : Defines the entry point for the console application.
//
#include "Source.h"

int main(int argc, const char** argv)
{
	if (!eyes_cascade2.load(eyes_cascade_name)) 
	{ 
		printf("Failed to load cascade\n"); 
		return -1;
	};

	Mat frame1 = imread("face1.jpg");
	Mat normalized1 = detectIris(frame1);
	vector<int> eye1 = LBP(normalized1);

	Mat frame2 = imread("face5.jpg");
	Mat normalized2 = detectIris(frame2);
	vector<int> eye2 = LBP(normalized2);

	cout << hammingDistance(eye1, eye2) << endl;
	int hold;
	cin >> hold;

	return 0;
}

void segmentIris(Mat &src, Mat &dst)
{
	cout << "segmenting" << endl;
	Segment segment;
	segment.findPupilEdge(src, dst);
	segment.findIrisEdge(dst, dst);

	int pupil_x, pupil_y, pupil_r, iris_r;
	pupil_x = segment.pupil_x;
	pupil_y = segment.pupil_y;
	pupil_r = segment.pupil_r;
	iris_r = segment.iris_r;
}

Mat detectIris(Mat input)
{
	Mat currentImage = input;

	//Eye localization
	currentImage = findEye(currentImage);
	Mat unprocessed = currentImage;
	showCurrentImage(currentImage);

	//Find and extract iris
	currentImage = findAndExtractIris(currentImage, unprocessed, input);
	showCurrentImage(currentImage);

	//Find pupil
	currentImage = findPupil(currentImage);

	//Normalize
	currentImage = normalize(currentImage, pupilx, pupily, pupilRadius, irisRadius);

	showCurrentImage(currentImage);
	destroyAllWindows();
	return currentImage;
}

Mat findEye(Mat input)
{
	Mat gray;

	cvtColor(input, gray, CV_BGR2GRAY);
	//equalizeHist(input_gray, input_gray);

	int minSize = gray.size().height * 0.15;

	vector<Rect> eyes;
	eyes_cascade2.detectMultiScale(gray, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));

	Rect eyeRegion;
	if (eyes[0].x > eyes[1].x)
		eyeRegion = Rect(eyes[0].x, eyes[0].y, eyes[0].width, eyes[0].height);
	else
		eyeRegion = Rect(eyes[1].x, eyes[1].y, eyes[1].width, eyes[1].height);
	return gray(eyeRegion);
}

Mat blurImage(Mat input)
{
	Mat blurredFrame;
	GaussianBlur(input, blurredFrame, Size(9, 9), 5, 5);
	return blurredFrame;
}

Mat CannyTransform(Mat input)
{
	Mat processed;
	double highVal = threshold(input, processed , 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double lowVal = highVal * 0.5;
	showCurrentImage(processed);

	Canny(processed, processed, lowVal, highVal, 3, false);
	return processed;
}

Mat EdgeContour(Mat input)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat processed;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Scharr(input, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	//Sobel(input, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Scharr(input, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	//Sobel(input, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, processed);
	return processed;
}

Mat findAndExtractIris(Mat &input, Mat &unprocessed, Mat &original)
{
	Mat processed;
	/*processed = EdgeContour(input);*/

	GaussianBlur(input, processed, Size(9, 9), 5, 5);
	double highVal = threshold(processed, processed, 0.8, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double lowVal = highVal * 0.5;

	processed = CannyTransform(processed);
	showCurrentImage(processed);

	vector<Vec3f> circles;
	HoughCircles(processed, circles, CV_HOUGH_GRADIENT, 1, original.rows / 8, 255, 30, 0, 0);
	for (size_t i = 0; i < 1; i++)//circles.size()
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		irisRadius = cvRound(circles[i][2]);

		circle(unprocessed, center, irisRadius, Scalar(0, 0, 255), 2, 8, 0);
	}

	showCurrentImage(unprocessed);

	Vec3f circ = circles[0];
	Mat1b mask(unprocessed.size(), uchar(0));
	circle(mask, Point(circ[0], circ[1]), circ[2], Scalar(255), CV_FILLED);
	Rect bbox(circ[0] - circ[2], circ[1] - circ[2], 2 * circ[2], 2 * circ[2]);
	Mat iris(200, 200, CV_8UC3, Scalar(255, 255, 255));

	unprocessed.copyTo(iris, mask);
	iris = iris(bbox);
	return iris;
}

Mat findPupil(Mat input)
{
	Mat cannyImage;
	GaussianBlur(input, cannyImage, Size(9, 9), 5, 5);

	Mat processed;
	double highVal = threshold(input, processed, 0.5, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double lowVal = highVal * 0.5;
	showCurrentImage(processed);

	Canny(processed, cannyImage, lowVal, highVal, 3, false);

	//cannyImage = CannyTransform(cannyImage);
	showCurrentImage(cannyImage);

	vector<Vec3f> circles;
	HoughCircles(cannyImage, circles, CV_HOUGH_GRADIENT, 20, input.rows / 8, 255, 30, 0, 0);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		pupilx = cvRound(circles[i][0]), pupily = cvRound(circles[i][1]);
		pupilRadius = cvRound(circles[i][2] * 1.1);

		circle(input, center, pupilRadius, Scalar(0, 0, 0), CV_FILLED);
	}

	showCurrentImage(input);
	return input;
}

Mat normalize(Mat input, int pupilx, int pupily, int pupilRadius, int irisRadius)
{
	
	int theta = 360;
	int differenceRadius = 80;
	cout << input.size().width << endl << input.size().height << endl;

	Mat normalized = Mat(differenceRadius, theta, CV_8U, Scalar(255));
	for (int i = 0; i < theta; i++)
	{
		double alpha = 2 * PI * i / theta;
		for (int j = 0; j < differenceRadius; j++)
		{
			double r = 1.0*j / differenceRadius;
			int x = (int)((1 - r)*(pupilx + pupilRadius*cos(alpha)) + r*(pupilx + irisRadius*cos(alpha)));
			int y = (int)((1 - r)*(pupily + pupilRadius*sin(alpha)) + r*(pupily + irisRadius*sin(alpha)));
			if (x < 0)
				x = 0;
			if (y < 0)
				y = 0;
			if (x > input.size().width-1)
				x = input.size().width-1;
			if (y > input.size().height-1)
				y = input.size().height-1;
			normalized.at<uchar>(j, i) = input.at<uchar>(y, x);
		}
	}
	Rect reducedSelection(0, 5, 360, 60);
	normalized = normalized(reducedSelection);
	return normalized;
}

vector<int> LBP(Mat input)
{
	vector<int> outputVector;
	for (size_t i = 1; i < input.rows - 1; i++)
	{
		for (size_t j = 1; j < input.cols - 1; j++)
		{
			//Currently centered pixel
			Scalar otherIntensity = input.at<uchar>(i, j);
			int vectorValue = 0;
			int pixelIntensity = otherIntensity.val[0];

			//Top left
			otherIntensity = input.at<uchar>(i - 1, j - 1);
			if (otherIntensity.val[0] < pixelIntensity)
				vectorValue += 128;

			//Top middle
			otherIntensity = input.at<uchar>(i, j - 1);
			if (otherIntensity.val[0] < pixelIntensity)
				vectorValue += 64;

			//Top right
			otherIntensity = input.at<uchar>(i + 1, j - 1);
			if (otherIntensity.val[0] < pixelIntensity)
				vectorValue += 32;

			//Right
			otherIntensity = input.at<uchar>(i + 1, j);
			if (otherIntensity.val[0] < pixelIntensity)
				vectorValue += 16;

			//Bottom right
			otherIntensity = input.at<uchar>(i + 1, j + 1);
			if (otherIntensity.val[0] < pixelIntensity)
				vectorValue += 8;

			//Botttom middle
			otherIntensity = input.at<uchar>(i, j + 1);
			if (otherIntensity.val[0] < pixelIntensity)
				vectorValue += 4;

			//Bottom left
			otherIntensity = input.at<uchar>(i - 1, j + 1);
			if (otherIntensity.val[0] < pixelIntensity)
				vectorValue += 2;

			//Left
			otherIntensity = input.at<uchar>(i - 1, j);
			if (otherIntensity.val[0] < pixelIntensity)
				vectorValue += 1;

			outputVector.push_back(vectorValue);
		}
	}
	return outputVector;
}

double hammingDistance(vector<int> savedCode, vector<int> inputCode)
{
	int currentDistance = 0;
	int averageDistance = 0;
	for (int i = 0; i < inputCode.size(); i++)
	{
		currentDistance = 0;
		unsigned  val = savedCode[i] ^ inputCode[i];

		while (val != 0)
		{
			currentDistance++;
			val &= val - 1;
		}
		averageDistance += currentDistance;
	}
	return 1.0*averageDistance / inputCode.size();
}

void showCurrentImage(Mat input)
{
	imshow(window, input);
	waitKey(0);
}
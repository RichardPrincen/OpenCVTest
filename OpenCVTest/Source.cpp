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


	Mat frame1 = imread("faceh2.jpg");
	Mat normalized1 = detectIris(frame1);
	vector<int> eye1 = LBP(normalized1);

	Mat frame2 = imread("faceh4.jpg");
	Mat normalized2 = detectIris(frame2);
	vector<int> eye2 = LBP(normalized2);

	cout << chiSquared(eye1, eye2) << endl;
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

	//Locate, extract and normalize iris
	currentImage = findAndExtractIris(currentImage, unprocessed, input);
	showCurrentImage(currentImage);

	destroyAllWindows();
	return currentImage;
}

Mat findEye(Mat input)
{
	Mat gray;

	cvtColor(input, gray, CV_BGR2GRAY);
	//equalizeHist(gray, gray);

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

Mat cannyTransform(Mat input)
{
	Mat processed;
	Canny(input, processed, 100, 120, 3, false);
	return processed;
}

Mat edgeContour(Mat input)
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

	processed = fillHoles(input);
	showCurrentImage(processed);

	GaussianBlur(processed, processed, Size(9, 9), 3, 3);
	
	//threshold(processed, processed, 70, 255, CV_THRESH_BINARY);
	//showCurrentImage(processed);

	showCurrentImage(processed);

	vector<Vec3f> circles;
	HoughCircles(processed, circles, CV_HOUGH_GRADIENT, 2, original.rows / 8, 255, 30, 0, 0);
	for (size_t i = 0; i < 1; i++)//circles.size()
	{
		

		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		pupilx = cvRound(circles[i][0]), pupily = cvRound(circles[i][1]);
		pupilRadius = cvRound(circles[i][2]);
		irisRadius = findIrisRadius(unprocessed, center, pupilRadius);

		circle(unprocessed, center, pupilRadius, Scalar(0, 0, 0), CV_FILLED);
		circle(unprocessed, center, irisRadius, Scalar(0, 0, 255), 2, 8, 0);
	}

	showCurrentImage(unprocessed);
	Mat iris = normalize(unprocessed);
	return iris;
}

int findIrisRadius(Mat input , Point startPoint, int radius)
{
	Mat processed;
	threshold(input, processed, 180, 255, CV_THRESH_BINARY);
	showCurrentImage(processed);
	int rightIntensity;
	int leftIntensity;
	int position = startPoint.y - (radius+20);
	int newRadius = radius+20;
	while (true)
	{
		rightIntensity = processed.at<uchar>(position, startPoint.y);
		position -= 10;
		newRadius += 10;
		leftIntensity = processed.at<uchar>(position, startPoint.y);
		if (leftIntensity != rightIntensity)
			return newRadius-10;
	}
	return 0;
}

Mat fillHoles(Mat input)
{
	Mat thresholded;
	threshold(input, thresholded, 70, 255, THRESH_BINARY_INV);

	Mat floodfilled = thresholded.clone();
	floodFill(floodfilled, Point(0, 0), Scalar(255));

	bitwise_not(floodfilled, floodfilled);

	return (thresholded | floodfilled);
}

Mat findPupil(Mat input, Rect eye)
{
	Mat cannyImage;
	GaussianBlur(input, cannyImage, Size(9, 9), 3, 3);

	Mat processed;
	double highVal = threshold(input, processed, 70, 255, CV_THRESH_BINARY);
	double lowVal = highVal * 0.5;
	showCurrentImage(processed);

	cannyImage = cannyTransform(processed);
	showCurrentImage(cannyImage);

	vector<Vec3f> circles;
	HoughCircles(cannyImage, circles, CV_HOUGH_GRADIENT, 20, input.rows, 255, 30, 0, 0);
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

Mat normalize(Mat input) // , int pupilx, int pupily, int pupilRadius, int irisRadius
{
	int yNew = 360;
	int xNew = 100;

	Mat normalized = Mat(xNew, yNew, CV_8U, Scalar(255));
	for (int i = 0; i < yNew; i++)
	{
		double alpha = 2 * PI * i / yNew;
		for (int j = 0; j < xNew; j++)
		{
			double r = 1.0*j / xNew;
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
	Rect reducedSelection(0, 5, 360, 75);
	normalized = normalized(reducedSelection);
	return normalized;
}

//Uniform LBP
vector<int> LBP(Mat input)
{
	vector<int> outputHist(59);
	fill(outputHist.begin(), outputHist.end(), 0);

	for (int i = 1; i < input.rows-1; i++)
	{
		for (int j = 1; j < input.cols-1; j++)
		{
			//Currently centered pixel
			Scalar otherIntensity = input.at<uchar>(i, j);
			int vectorValue = 0;
			vector<int> binaryCode;
			int pixelIntensity = otherIntensity.val[0];

			//Top left
			otherIntensity = input.at<uchar>(i - 1, j - 1);
			if (otherIntensity.val[0] < pixelIntensity)
			{
				vectorValue += 128;
				binaryCode.push_back(1);
			}
			else
				binaryCode.push_back(0);

			//Top middle
			otherIntensity = input.at<uchar>(i, j - 1);
			if (otherIntensity.val[0] < pixelIntensity)
			{
				vectorValue += 64;
				binaryCode.push_back(1);
			}
			else
				binaryCode.push_back(0);

			//Top right
			otherIntensity = input.at<uchar>(i + 1, j - 1);
			if (otherIntensity.val[0] < pixelIntensity)
			{
				vectorValue += 32;
				binaryCode.push_back(1);
			}
			else
				binaryCode.push_back(0);

			//Right
			otherIntensity = input.at<uchar>(i + 1, j);
			if (otherIntensity.val[0] < pixelIntensity)
			{
				vectorValue += 16;
				binaryCode.push_back(1);
			}
			else
				binaryCode.push_back(0);

			//Bottom right
			otherIntensity = input.at<uchar>(i + 1, j + 1);
			if (otherIntensity.val[0] < pixelIntensity)
			{
				vectorValue += 8;
				binaryCode.push_back(1);
			}
			else
				binaryCode.push_back(0);

			//Botttom middle
			otherIntensity = input.at<uchar>(i, j + 1);
			if (otherIntensity.val[0] < pixelIntensity)
			{
				vectorValue += 4;
				binaryCode.push_back(1);
			}
			else
				binaryCode.push_back(0);

			//Bottom left
			otherIntensity = input.at<uchar>(i - 1, j + 1);
			if (otherIntensity.val[0] < pixelIntensity)
			{
				vectorValue += 2;
				binaryCode.push_back(1);
			}
			else
				binaryCode.push_back(0);

			//Left
			otherIntensity = input.at<uchar>(i - 1, j);
			if (otherIntensity.val[0] < pixelIntensity)
			{
				vectorValue += 1;
				binaryCode.push_back(1);
			}
			else
				binaryCode.push_back(0);

			if (checkUniform(binaryCode))
			{
				for (int x = 0; x < 59; x++)
					if (histogramValues[x] == vectorValue)
						outputHist[x]++;
			}
			else
				outputHist[58]++;
		}
	}
	return outputHist;
}

bool checkUniform(vector<int> binaryCode)
{
	int transitionCount = 0;
	for (int i = 1; i < 8; i++)
	{
		if (binaryCode[i] ^ binaryCode[i - 1] == 1)
			transitionCount++;

		if (transitionCount > 2)
			return false;
	}
	return true;
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
		averageDistance += currentDistance/8;
	}
	return 1.0*averageDistance / inputCode.size();
}

double chiSquared(vector<int> hist1, vector<int> hist2)
{
	double chiSquaredValue = 0.0;

	vector<double> normalizedHist1(59);
	vector<double> normalizedHist2(59);

	for (int i = 0; i < 58; i++)
	{
		normalizedHist1[i] = (double)hist1[i] / hist1[58];
		normalizedHist2[i] = (double)hist2[i] / hist2[58];
	}

	normalizedHist1[58] = 1.0;
	normalizedHist2[58] = 1.0;

	for (int i = 1; i < 59; i++)
	{
		if (hist1[i] + hist2[i] != 0)
		{
			chiSquaredValue += pow(normalizedHist1[i] - normalizedHist2[i], 2) / (normalizedHist1[i] + normalizedHist2[i]);
		}
	}
	return (chiSquaredValue);
}

void showCurrentImage(Mat input)
{
	imshow(window, input);
	waitKey(0);
}













class MyPoint 
{
public:
	int x, y;
	MyPoint(int x, int y) 
	{
		this->x = x;
		this->y = y;
	}
};

class Image : public Mat 
{
public:
	int radiusForCheck;
	vector<MyPoint> outOfThreshold;
	Image(/*int radiusForCheck*/) 
	{
	}
};

bool houghCircle(Mat cannied, int &radius, vector<int> &xCenter, vector<int> &yCenter)
{
	int min_r = radius, max_r = radius*5;
	vector<Image> radiuses(max_r - min_r);
	for (int r = min_r; r < max_r; r++) 
	{
		Image& image = radiuses.at(r - min_r);
		image.radiusForCheck = r;

		int** matrix = new int*[cannied.cols];
		for (int i = 0; i < cannied.cols; ++i)
			matrix[i] = new int[cannied.rows];

		for (int x = 0; x < cannied.cols; x++) 
		{
			for (int y = 0; y < cannied.rows; y++) 
			{
				matrix[x][y] = 0;
			}
		}

		int pointsThreshold = 30;
		for (int x = 0; x < cannied.cols; x++) 
		{
			for (int y = 0; y < cannied.rows; y++) 
			{
				if ((int)cannied.at<uchar>(y, x) == 255)
				{
					for (int k = 0; k < 360; k += 2) 
					{
						int xCurrent = x + r*cos((double)(k*PI) / 180.00);
						int yCurrent = y + r*sin((double)(k*PI) / 180.00);
						if (xCurrent <= 0 || xCurrent >= cannied.cols - 1 || yCurrent <= 0 || yCurrent >= cannied.rows - 1 || matrix[xCurrent][yCurrent] >= pointsThreshold) 
						{
							continue;
						}
						matrix[xCurrent][yCurrent]++;
						if (matrix[xCurrent][yCurrent] >= pointsThreshold) 
						{
							image.outOfThreshold.push_back(MyPoint(xCurrent, yCurrent));
						}
					}
				}
			}
		}
	}
	for (int i = 0; i < radiuses.size(); i++) 
	{
		Image image = radiuses.at(i);
		if (image.outOfThreshold.size()>0 && image.radiusForCheck >= min_r) 
		{
			radius = image.radiusForCheck;
			for (int j = 0; j < 1; j++) 
			{
				int x = image.outOfThreshold.at(j).x;
				int y = image.outOfThreshold.at(j).y;
				xCenter.push_back(x);
				yCenter.push_back(y);
			}
			cout << image.outOfThreshold.size() << "\n";
			return true;
		}
	}
	return false;
}
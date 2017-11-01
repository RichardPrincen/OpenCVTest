// OpenCVTest.cpp : Defines the entry point for the console application.
//
#include "Source.h"

//loads images
int main(int argc, const char** argv)
{
	if (!eyes_cascade2.load(eyes_cascade_name)) 
	{ 
		printf("Failed to load cascade\n"); 
		return -1;
	};


	Mat frame1 = imread("faceh4.jpg");
	Mat normalized1 = detectIris(frame1);
	vector<int> eye1 = LBP(normalized1);

	Mat frame2 = imread("faceh2.jpg");
	Mat normalized2 = detectIris(frame2);
	vector<int> eye2 = LBP(normalized2);

	cout << chiSquared(eye1, eye2) << endl;

	vector<int> eye1NBP = NBP(normalized1);
	vector<int> eye2NBP = NBP(normalized2);
	cout << hammingDistance(eye1NBP, eye2NBP) << endl;


	int hold;
	cin >> hold;

	return 0;
}

//calls the iris recognition fucntions
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

//uses haar cascade to find the eye
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

//gaussian blurs the image using opencv
Mat blurImage(Mat input)
{
	Mat blurredFrame;
	GaussianBlur(input, blurredFrame, Size(9, 9), 5, 5);
	return blurredFrame;
}

//canny edge detection using opencv
Mat cannyTransform(Mat input)
{
	Mat processed;
	Canny(input, processed, 100, 120, 3, false);
	return processed;
}

//sobel and scharr edge detection using opencv - unused
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

//find iris and pupil using CHT
Mat findAndExtractIris(Mat &input, Mat &unprocessed, Mat &original)
{

	Mat processed;

	processed = fillHoles(input);
	showCurrentImage(processed);
	
	/*vector<int> circleOut = CHT(processed, 20, 50);
	Point center(cvRound(circleOut[0]), cvRound(circleOut[1]));
	pupilx = cvRound(circleOut[0]), pupily = cvRound(circleOut[1]);
	pupilRadius = cvRound(circleOut[2]);
	irisRadius = findIrisRadius(unprocessed, center, pupilRadius);

	circle(unprocessed, center, pupilRadius*1.1, Scalar(0, 0, 0), CV_FILLED);
	circle(unprocessed, center, irisRadius, Scalar(0, 0, 255), 2, 8, 0);*/

	GaussianBlur(processed, processed, Size(9, 9), 3, 3);
	showCurrentImage(processed);

	vector<Vec3f> circles;
	HoughCircles(processed, circles, CV_HOUGH_GRADIENT, 2, original.rows / 8, 255, 30, 0, 0);
	for (size_t i = 0; i < 1; i++)//circles.size()
	{
		

		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		pupilx = cvRound(circles[i][0]), pupily = cvRound(circles[i][1]);
		pupilRadius = cvRound(circles[i][2]);
		irisRadius = findIrisRadius(unprocessed, center, pupilRadius);

		circle(unprocessed, center, pupilRadius*1.1, Scalar(0, 0, 0), CV_FILLED);
		circle(unprocessed, center, irisRadius, Scalar(0, 0, 255), 2, 8, 0);
	}

	showCurrentImage(unprocessed);
	Mat iris = normalize(unprocessed);
	return iris;
}

//finds the iris radius using pupil radius and thresholding
int findIrisRadius(Mat input , Point startPoint, int radius)
{
	Mat processed;
	threshold(input, processed, 180, 255, CV_THRESH_BINARY);
	showCurrentImage(processed);
	int rightIntensity;
	int leftIntensity;
	int position = startPoint.x + (radius+20);
	int newRadius = radius+20;
	while (true)
	{
		rightIntensity = processed.at<uchar>(startPoint.y, position);
		position += 10;
		newRadius += 10;
		leftIntensity = processed.at<uchar>(startPoint.y, position);
		if (leftIntensity != rightIntensity)
			return newRadius-10;
	}
	return 0;
}

//fills "holes" created by reflections
Mat fillHoles(Mat input)
{
	Mat thresholded;
	threshold(input, thresholded, 70, 255, THRESH_BINARY_INV);

	showCurrentImage(thresholded);

	Mat floodfilled = thresholded.clone();
	floodFill(floodfilled, Point(0, 0), Scalar(255));

	bitwise_not(floodfilled, floodfilled);

	return (thresholded | floodfilled);
}

//finds the pupil - unused
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

//normalizes the circular image to rectangular
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
	Rect reducedSelection(0, 5, 360, 60);
	normalized = normalized(reducedSelection);
	return normalized;
}

//uniform LBP feature extraction method
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

//checks if a pixel vector is uniform
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

//NBP feature extraction method
vector<int> NBP(Mat input)
{
	Mat NBPimage = Mat(input.rows, input.cols, CV_8U, Scalar(255));
	vector<int> NBPcode;

	for (int i = 1; i < input.rows - 1; i++)
	{
		for (int j = 1; j < input.cols - 1; j++)
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

			NBPimage.at<uchar>(i, j) = vectorValue;
		}
	}

	showCurrentImage(NBPimage);
	vector<vector<int>> means(0);
	vector<int> rowmeans(0);
	for (int j = 0; j < 6; j++)
	{
		rowmeans = vector<int>(0);
		for (int i = 0; i < 6; i++)
		{
			int blockmean = 0;
			for (int x = i * 60; x < i * 60 + 60; x++)
			{
				for (int y = j * 10; y < j * 10 + 10; y++)
				{
					blockmean += NBPimage.at<uchar>(y, x);
				}
			}
			rowmeans.push_back(blockmean/(60*20));
		}
		means.push_back(rowmeans);
	}

	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			if (means.at(i).at(j) > means.at(i).at(j + 1))
				NBPcode.push_back(1);
			else
				NBPcode.push_back(0);
		}
	}
	return NBPcode;
}

//hamming distance to compare two vectors
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

//chi square distance to compare two histograms
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

//shows an image in output window
void showCurrentImage(Mat input)
{
	imshow(window, input);
	waitKey(0);
}

//circular hough transform
vector<int> CHT(Mat input, int minRadius, int maxRadius)
{
	vector<int> outputVector = vector<int>(3);
	Mat cannyimage = input;
	Mat cannyimageLarge = cannyimage;

	Size newSize(cannyimage.cols/4, cannyimage.rows/4);
	resize(cannyimage, cannyimage, newSize);
	maxRadius = maxRadius / 4;
	minRadius = minRadius / 4;
	cannyimage = myEdgeDetetor(cannyimage);
	cannyimageLarge = myEdgeDetetor(cannyimageLarge);
	showCurrentImage(cannyimage);
	int xdim = cannyimage.cols;
	int ydim = cannyimage.rows;
	int rdim = maxRadius;

	vector<vector<vector<double>>> accumulator(xdim, vector<vector<double>>(ydim, vector<double>(rdim)));

	for (int x = 0; x < cannyimage.cols; x++)
	{
		for (int y = 0; y < cannyimage.rows; y++)
		{
			if (cannyimage.at<uchar>(x,y) == 255)
			{
				for (int r = minRadius; r < maxRadius; r++)
				{
					for (int theta = 0; theta < 360; theta++)
					{
						int a = x - r * cos(theta * PI / 180);
						int b = y - r * sin(theta * PI / 180);
						if (a > 0 & b > 0 & a < cannyimage.cols & b < cannyimage.rows)
							accumulator[a][b][r] = accumulator[a][b][r] + 1;
					}
				}
			}
		}
	}

	int centerx = -1;
	int centery = -1;
	int finalRadius = -1;
	int max = 0;
	for (int x = 0; x < cannyimage.cols; x++)
	{
		for (int y = 0; y < cannyimage.rows; y++)
		{
			for (int r = minRadius; r < maxRadius; r++)
			{
				if (accumulator[x][y][r] > max)
				{
					centerx = x;
					centery = y;
					finalRadius = r;
					max = accumulator[x][y][r];
				}
			}
		}
	}
	showCurrentImage(cannyimageLarge);
	outputVector[0] = centerx * 4;
	outputVector[1] = centery * 4;
	outputVector[2] = finalRadius * 4;
	circle(cannyimageLarge, Point(centerx*4, centery*4), finalRadius*4, Scalar(255, 255, 255),2);
	showCurrentImage(cannyimageLarge);
	return outputVector;
}

//thresholds the image based on the input value
Mat myThreshold(Mat input, int threshold)
{
	Mat threshImage = Mat(input.rows, input.cols, CV_8U, Scalar(255));
	Scalar pixelIntensity;
	for (int x = 0; x < input.cols; x++)
	{
		for (int y = 0; y < input.rows; y++)
		{
			pixelIntensity = input.at<uchar>(x, y);
			if (pixelIntensity.val[0] > threshold)
				threshImage.at<uchar>(x, y) = 0;
		}
	}
	return threshImage;
}

//vertical edge detection
Mat myEdgeDetetor(Mat thresholdInput)
{
	Mat edgeImage = Mat(thresholdInput.rows, thresholdInput.cols, CV_8U, Scalar(0));
	Scalar pixelIntensity;
	Scalar OtherIntensity;
	for (int x = 1; x < thresholdInput.cols-1; x++)
	{
		for (int y = 1; y < thresholdInput.rows-1; y++)
		{
			pixelIntensity = thresholdInput.at<uchar>(x, y);
			OtherIntensity = thresholdInput.at<uchar>(x+1, y);
			if (pixelIntensity != OtherIntensity)
				edgeImage.at<uchar>(x+1, y) = 255;
			OtherIntensity = thresholdInput.at<uchar>(x-1, y);
			if (pixelIntensity != OtherIntensity)
				edgeImage.at<uchar>(x-1, y) = 255;
		}
	}
	return edgeImage;
}

#include "gaborfilter.h"
#include <fstream>
#include <iostream>

#define M_PI 3.1415

Gaborfilter::Gaborfilter()
{

}

Gaborfilter::~Gaborfilter()
{

}

bool Gaborfilter::create_kernel(int r, int theta, int alpha, int beta, double omega, int ktype)
{
    int xmin,ymin,xmax,ymax;

    if(theta > 0)
        xmax = theta/2;

    if(r > 0)
        ymax = r/2;

    xmin = -xmax;
    ymin = -ymax;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );

    realKernel = Mat(ymax - ymin + 1, xmax - xmin + 1, ktype);
    imagKernel = Mat(ymax - ymin + 1, xmax - xmin + 1, ktype);

    for( int y = ymin; y <= ymax; y++ )
    {
        for( int x = xmin; x <= xmax; x++ )
        {
            double o = x * M_PI / 180;
            double c = cos(-omega*o),s = sin(-omega*o);
            double er = -((pow(y,2))/(pow(alpha,2)));
            double eo = -((pow(x,2))/(pow(beta,2)));

            double rv = exp(er + eo)*c;
            double iv = exp(er + eo)*s;
            if(ktype == CV_32F)
			{
                realKernel.at<float>(ymax - y, xmax - x) = (float)rv;
                imagKernel.at<float>(ymax - y, xmax - x) = (float)iv;
            }
            else
			{
                realKernel.at<double>(ymax - y, xmax - x) = rv;
                imagKernel.at<double>(ymax - y, xmax - x) = iv;
            }
        }
    }
    isCreatedKernel = true;
    return isCreatedKernel;
}

cv::Mat Gaborfilter::getRealKernel()
{
    return realKernel;
}

cv::Mat Gaborfilter::getImagKernel()
{

    return imagKernel;

}

std::vector<char> Gaborfilter::getIrisCode()
{
    return irisCode;
}

void Gaborfilter::clearIrisCode()
{
    irisCode.clear();
}

double Gaborfilter::filterGabor(cv::Mat &src, cv::Mat &kernel)
{
    double sum = 0;
    for(int p = 0;p < src.rows;p++)
    {
        uchar *data = src.ptr<uchar>(p);
        double *k = kernel.ptr<double>(p);
        for(int q = 0;q < src.cols;q++)
        {
            sum += data[q]*k[q]*p;
        }
    }
    return sum;
}

void Gaborfilter::gaborCode(Mat input)
{
    int nr = input.rows;
    int theta = input.cols;

    Mat subMat1(input,Rect(0,0,theta*1/6,nr*1/2));
    Mat subMat2(input,Rect(theta*1/6,0,theta*1/6,nr*1/2));
    Mat subMat3(input,Rect(theta*2/6,0,theta*1/6,nr*1/2));
    Mat subMat4(input,Rect(theta*3/6,0,theta*1/6,nr*1/2));
    Mat subMat5(input,Rect(theta*4/6,0,theta*1/6,nr*1/2));
    Mat subMat6(input,Rect(theta*5/6,0,theta*1/6,nr*1/2));
    Mat subMat7(input,Rect(0,nr*1/2,theta*1/6,nr*1/2));
    Mat subMat8(input,Rect(theta*1/6,nr*1/2,theta*1/6,nr*1/2));
    Mat subMat9(input,Rect(theta*2/6,nr*1/2,theta*1/6,nr*1/2));
    Mat subMat10(input,Rect(theta*3/6,nr*1/2,theta*1/6,nr*1/2));
    Mat subMat11(input,Rect(theta*4/6,nr*1/2,theta*1/6,nr*1/2));
    Mat subMat12(input,Rect(theta*5/6,nr*1/2,theta*1/6,nr*1/2));

    for(int i = 0;i < 100;i++)
    {
        double maxw = 0;
        if(create_kernel(40,60,40,60,0.1+0.1*i,CV_64F))
        {
            Mat realK = getRealKernel();
            Mat imagK = getImagKernel();
            for(int j = 1;j < 13;j++)
            {
                Mat subMat;
                switch(j)
                {
					case 1:
						subMat = subMat1;
						break;
					case 2:
						subMat = subMat2;
						break;
					case 3:
						subMat = subMat3;
						break;
					case 4:
						subMat = subMat4;
						break;
					case 5:
						subMat = subMat5;
						break;
					case 6:
						subMat = subMat6;
						break;
					case 7:
						subMat = subMat7;
						break;
					case 8:
						subMat = subMat8;
						break;
					case 9:
						subMat = subMat9;
						break;
					case 10:
						subMat = subMat10;
						break;
					case 11:
						subMat = subMat11;
						break;
					case 12:
						subMat = subMat12;
						break;
					default:break;
                }
                double real = filterGabor(subMat,realK);
                double imag = filterGabor(subMat,imagK);

                if(real >= 0)
                {
					irisCode.push_back('1');
					maxw += real;
				}
                else
                {
					irisCode.push_back('0');
					maxw += -real;
				}
                if(imag >= 0)
                {
					irisCode.push_back('1');
					maxw += imag;
				}
                else
                {
					irisCode.push_back('0');
					maxw += -imag;
				}
            }
       }
       suitmap.insert({maxw,0.1+0.1*i});
    }
}

void Gaborfilter::quicksort(vector<double> a, int left, int right)
{
    if(left < right)
    {
        int i = left;
        int j = right;
        double x = a[i];
        while(i < j)
        {
            while(i < j && a[j] < x)
                j--;
            if(i < j)
                a[i++] = a[j];
            while(i < j && a[i] > x)
                i++;
            if(i < j)
                a[j--] = a[i];
        }
        a[i] = x;
        quicksort(a,left,i-1);
        quicksort(a,i+1,right);
    }
}

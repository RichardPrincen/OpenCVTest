#include "segment.h"
#include <iostream>

#define HAVE_STRUCT_TIMESPEC

#include "pthread.h"

#define THREADS_NUM 4
#define M_PI 3.1415

static r_attr* calMaxEdge(t_attr *attr)
{
    int max = 0;
    r_attr *rattr = new r_attr;
    for(int i = attr->i0;i < attr->i1;i++)
    {
        for(int j = attr->j0;j < attr->j1;j++)
        {
            for(int r = attr->r0;r < attr->r1;r++)
            {
                int die = attr->seg.calCircleSum(attr->seg.gaussMat,j,i,r+1)
                        - attr->seg.calCircleSum(attr->seg.gaussMat,j,i,r-1);
                if(die > max)
                {
                    max = die;
                    rattr->x = j;
                    rattr->y = i;
                    rattr->r = r;
                    rattr->max = max;
                }
            }
        }
    }
	return rattr;
}

Segment::Segment()
{

}

Segment::~Segment()
{

}

void Segment::findPupilEdge(cv::Mat &src, cv::Mat &dst)
{
    int px,py,pr;
    dst = src.clone();
    int height = src.rows;
    int width = src.cols;
    t_attr attr[THREADS_NUM];
    for(int i = 0;i < THREADS_NUM;i++)
    {
        attr[i].i0 = 150;attr[i].i1 = height-150;
        attr[i].j0 = 200;attr[i].j1 = width-200;
        attr[i].r0 = 20+i*10;attr[i].r1 = 30+i*10;
        attr[i].seg.gaussMat = src;
		r_attr *maxattr = calMaxEdge(&attr[i]);
		int themax = 0;
        if(maxattr->max > themax)
        {
            themax = maxattr->max;
            px = maxattr->x;
            py = maxattr->y;
            pr = maxattr->r;
        }
        delete maxattr;
    }
    pupil_x = px;
    pupil_y = py;
    pupil_r = pr;

    drawCircle(dst,pupil_x,pupil_y,pupil_r+1);
}

void Segment::findIrisEdge(cv::Mat &src, cv::Mat &dst)
{
    dst = src.clone();
    int imax = 0;
    for(int r1 = pupil_r * 1.5;r1 < pupil_r * 3.5;r1++)
    {
        int die1 = calCircleSum(src,pupil_x,pupil_y,r1+1)-calCircleSum(src,pupil_x,pupil_y,r1-1);
        if(die1 > imax)
        {
            iris_r = r1;
            imax = die1;
        }
    }
    drawCircle(dst,pupil_x,pupil_y,iris_r-1);
}

int Segment::calCircleSum(cv::Mat &img,int x,int y,int r)
{
   int cs = 2 * M_PI * r;
   double alpha;
   int cx,cy;
   int sum = 0;

   for(int i = 0;i < cs;i++)
   {
       alpha = (double)(2 * M_PI * i / cs);
       cx = x + r * cos(alpha);
       cy = y + r * sin(alpha);
	   uchar *data = img.ptr<uchar>(cy);

       sum += data[cx-1]+data[cx]+data[cx+1];
   }
   return sum;
}

void Segment::drawCircle(cv::Mat &img, int x, int y, int r)
{
    int cs = 2 * M_PI * r;
    double alpha;
    int cx,cy;
    for(int i = 0;i < cs;i++)
    {
        alpha = (double)(2 * M_PI * i / cs);
        cx = x + r * cos(alpha);
        cy = y + r * sin(alpha);
        img.at<uchar>(cy,cx) = 255;
    }
}

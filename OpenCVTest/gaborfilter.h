#ifndef GABORFILTER_H
#define GABORFILTER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <map>

using namespace cv;
using namespace std;

class Gaborfilter
{
public:
    Gaborfilter();
    ~Gaborfilter();

    bool create_kernel(int r,int theta,int alpha,int beta,double omiga,int ktype);
    Mat getRealKernel();
    Mat getImagKernel();
    vector<char> getIrisCode();
    void clearIrisCode();
    double filterGabor(Mat &src,Mat &kernel);
    void gaborCode(Mat input);
    void quicksort(vector<double> a,int left,int right);

private:
    Mat realKernel,imagKernel;
    vector<char> irisCode;
    bool isCreatedKernel = false;
    //vector<double> suitw;
    multimap<double,double> suitmap;
};

#endif // GABORFILTER_H

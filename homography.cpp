//
// Created by subigya on 4/30/17.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

string getRawData();
vector<float> getFloatVectorData();
void convertFloatVectorToTwoPoint2f(vector<float> &original, vector<Point2f> &firstPoint2f, vector<Point2f> &secondPoint2f);
void printVector(vector<Point2f> vect);
void printVector(vector<float> vect);

int main(){
    Point2f pt = Point2f(4.4122e+02, 3.5669e+02);
    cout<< pt<< endl;
    cout << getRawData();
    vector<float> data = getFloatVectorData();
    vector<Point2f> firstPoints, secondPoints;
    convertFloatVectorToTwoPoint2f(data, firstPoints, secondPoints);
    printVector(firstPoints);
     cout << firstPoints.size() << " second " << secondPoints.size() << " original " << data.size() << endl;

    Mat homography = findHomography(firstPoints, secondPoints, CV_RANSAC, 5, noArray(), 5000, 0.999);
    cout << "Homography: " << homography << endl;

    Mat fundamental = findFundamentalMat(firstPoints, secondPoints);
    cout << "Fundamental: " << fundamental << endl;

    Mat W, U, Vt;
    SVD::compute(fundamental, W, U, Vt);
    cout << "W: " << W << " U: " << U << " vt: " << Vt << endl;

//    Mat essential = findEssentialMat(firstPoints, secondPoints)
    return 0;
}

void printVector(vector<Point2f> vect){
    for (auto i = vect.begin(); i < vect.end(); ++i) {
        cout << "x: " << i->x << " y: " << i->y << endl;
    }
}
void printVector(vector<float> vect){
    for (auto i = vect.begin(); i < vect.end(); ++i) {
        cout << *i << endl;
    }
}

void convertFloatVectorToTwoPoint2f(vector<float> &original, vector<Point2f> &firstPoint2f, vector<Point2f> &secondPoint2f){
    unsigned long size = original.size();
    for(unsigned long i=0; i< size; i=i+2){
        // i = 0,2,4,6,8.. so
        if (i % 4 ==0 ){
            // for first two
            firstPoint2f.push_back(Point2f(original.at(i), original.at(i+1)));
        }
        else {
            // for second two
            secondPoint2f.push_back(Point2f(original.at(i), original.at(i+1)));
        }
    }
}

vector<float> getFloatVectorData(){
    stringstream ss(getRawData());
    vector<float> result;

    while(ss.good()){
        string substr;
        getline(ss, substr, ' ');
        result.push_back(stof(substr));
    }
    return result;
}

string getRawData(){
    return "4.4122e+02 3.5669e+02 -2.3601e+00 -1.7256e-01 "
            "1.4102e+02 4.5156e+02 -1.2853e+00 -3.9569e+00 "
            "1.3801e+01 4.0302e+02 1.1422e+02 4.6162e+02 "
            "4.2605e+02 3.4917e+02 2.6323e+02 1.5949e+02 "
            "5.1989e+02 2.3802e+02 8.2251e+01 3.9666e+02 2.6386e+02 2.7864e+02 3.1431e+02 7.5606e+01 "
            "1.3909e+02 1.2320e+02 -3.3999e+00 -1.1976e+00 "
            "5.5778e+01 4.5022e+02 4.7082e+01 3.6045e+02 "
            "4.3003e+02 2.7685e+02 -3.0426e+00 -5.0519e-01 "
            "6.3113e+02 4.6688e+02 -3.8098e+00 -2.1954e-01 "
            "2.5758e+02 3.3289e+02 -1.5893e+00 -5.5791e-01 "
            "1.8767e+02 3.8154e+02 -2.9630e+00 -8.7075e-01 "
            "3.7777e+02 4.7581e+02 2.3548e+02 1.7307e+02 "
            "1.8511e+02 4.6053e+02 4.9810e+02 2.8688e+02 "
            "1.9170e+02 1.1619e+02 -4.6653e+00 2.9691e-01 "
            "3.7304e+02 3.9895e+01 -4.4937e+00 2.0168e-01 "
            "6.0214e+02 4.4121e+02 -3.2844e+00 4.8423e-01 "
            "4.3120e+02 1.1175e+02 9.0500e+01 7.3717e+01 "
            "5.9259e+02 4.5582e+02 -5.2435e-01 -1.1419e+00 "
            "6.0639e+01 4.6427e+02 8.4312e-01 -5.9158e+00 "
            "2.7896e+02 9.9037e+01 1.9727e+02 3.1475e+02 "
            "4.5324e+02 1.0598e+02 -4.1944e+00 2.7562e-01 "
            "3.0354e+02 9.4212e+01 -3.2864e+00 3.8957e-01 "
            "1.3696e+02 3.1481e+02 -1.7484e+00 -2.4544e+00 "
            "1.5759e+02 2.5293e+02 -3.6322e+00 -1.3200e+00 "
            "3.1094e+02 2.0235e+02 -3.8414e+00 -1.0901e+00 "
            "1.7343e+02 4.2333e+02 -3.4569e+00 -3.9371e+00 "
            "3.6762e+01 3.5834e+02 -5.5590e-03 -5.3016e+00 "
            "5.1635e+01 3.1612e+02 9.7439e-01 -5.7353e+00 "
            "5.9343e+02 4.2102e+02 -2.8265e+00 -1.2313e+00 "
            "4.8460e+02 3.9469e+02 2.6387e+02 2.2237e+02 "
            "5.5373e+02 3.7907e+02 -3.0684e+00 1.9494e-01 "
            "1.9614e+02 2.1430e+02 -3.1453e+00 -2.0243e+00 "
            "2.9462e+02 4.0101e+02 5.6956e+02 2.7592e+02 "
            "2.8031e+02 3.1582e+02 -2.3848e+00 -1.3164e+00 "
            "5.4560e+02 2.0473e+02 4.4105e+02 9.8418e+01 "
            "3.9384e+02 3.9231e+02 -3.0280e+00 -1.7480e+00 "
            "2.8618e+02 1.1187e+02 -2.9945e+00 2.5118e-02 "
            "2.7596e+02 9.2459e+01 -4.7321e+00 -3.9205e-01 2.5694e+02 9.5804e+01 5.2348e+02 2.3251e+02 "
            "3.3113e+01 1.3068e+02 1.0516e+02 5.9488e+01 "
            "2.7123e+02 2.0288e+02 5.2148e+02 2.2484e+02 "
            "5.5682e+01 3.3354e+02 -2.1124e+00 -4.6259e+00 "
            "6.3525e+02 3.7279e+02 -4.9155e+00 -5.6509e-01 "
            "1.5485e+02 1.3192e+02 -4.4178e+00 -6.0131e-01 "
            "2.4838e+02 2.5302e+02 5.4069e+02 1.0883e+02 "
            "8.0836e+00 4.2469e+02 -3.1172e-01 -7.3603e+00 "
            "3.0633e+02 3.7853e+02 -3.4256e+00 -1.4643e+00 "
            "7.6750e+00 3.9950e+02 5.7335e-01 -6.6080e+00 "
            "5.6985e+02 1.2186e+02 5.5998e+02 4.6348e+02 "
            "6.0806e+02 1.6507e+02 4.9816e+02 2.0361e+01 "
            "3.7305e+01 3.5775e+01 -5.1847e+00 4.9041e+00 "
            "4.7051e+02 4.4193e+02 -3.3045e+00 -1.4723e+00 "
            "2.7733e+02 4.6868e+02 -2.1378e+00 -2.2239e+00 "
            "2.9923e+02 4.3434e+02 -3.7926e+00 -2.1575e+00 "
            "3.1605e+01 4.0135e+02 8.9749e+01 6.5536e+01 "
            "5.1532e+01 2.8080e+02 -7.7820e-01 -4.7095e+00 "
            "4.8577e+02 3.4410e+02 -1.3216e+00 -2.7655e+00 "
            "5.4407e+02 3.2195e+02 -2.7440e+00 5.8397e-01 "
            "5.2658e+01 3.0577e+02 -1.7303e+00 -5.0547e+00 "
            "3.2859e+02 2.0973e+01 -5.7084e+00 6.7488e-01 "
            "5.4132e+02 2.5211e+02 2.1954e+02 4.0518e+02 "
            "2.9132e+02 2.5204e+02 -3.5410e+00 -1.7173e+00 "
            "6.2327e+02 3.8025e+02 -5.1229e+00 -9.3900e-01 "
            "4.6068e+02 4.2227e+02 -2.8825e+00 -1.0694e+00 "
            "2.6908e+02 2.1610e+02 -4.7266e+00 -1.7918e+00 "
            "5.9754e+02 9.9301e+01 -3.9602e+00 1.3621e+00 "
            "3.5477e+02 3.3707e+02 -3.0081e+00 -1.5099e+00 "
            "4.3446e+02 3.2005e+02 -1.7924e+00 -2.2342e+00 "
            "3.5697e+02 1.4707e+02 -3.0883e+00 5.8945e-01 "
            "2.7596e+02 4.9161e+01 5.1468e+02 4.7889e+02 "
            "2.5540e+02 2.6893e+02 -3.8181e+00 -1.8020e+00 "
            "4.8040e+01 4.7137e+02 3.7674e+02 6.5381e+01 3.3635e+02 3.3346e+02 -2.3085e+00 -1.6912e+00 "
            "3.3770e+02 2.2970e+02 6.0892e+02 1.1993e+02 "
            "5.4549e+02 9.6269e+00 -5.5029e+00 5.2547e-01 "
            "4.0142e+02 2.0608e+02 -4.7220e+00 -1.4602e-01 "
            "2.3762e+02 1.8168e+02 -2.9049e+00 -5.7438e-01 "
            "5.3360e+02 2.8132e+02 4.5625e+02 2.3194e+01 "
            "3.1358e+02 2.2596e+02 -3.5245e+00 -1.9968e+00 "
            "1.1554e+02 9.0529e+01 2.4397e+01 3.7688e+02 "
            "2.7537e+01 2.8960e+02 1.6481e+02 4.6060e+02 "
            "2.7134e+02 1.7282e+02 3.6070e+02 7.0852e+01 "
            "3.6942e+02 2.9681e+02 -3.1730e+00 -1.5885e+00 "
            "2.3496e+02 1.2060e+02 -2.2845e+00 -3.8655e-01 "
            "1.9637e+02 2.1952e+02 -1.9660e+00 -2.6976e+00 "
            "9.8016e+01 1.7705e+02 -3.4515e+00 7.1545e-03 "
            "2.3174e+02 1.7245e+02 -3.3155e+00 6.6318e-02 "
            "6.1424e+02 4.0837e+02 -3.3307e+00 -9.2676e-01 "
            "4.4880e+02 2.2458e+02 4.0821e+02 2.1841e+02 "
            "5.8479e+02 3.4771e+02 -2.6328e+00 1.8879e-01 "
            "5.0884e+02 3.3654e+01 -5.1868e+00 -8.7152e-01 "
            "8.2269e+01 8.2086e+01 -2.3295e+00 1.0687e+00 "
            "8.7864e+00 1.9540e+01 2.2020e+01 2.3156e+02 "
            "1.7974e+02 3.7486e+02 1.9284e+02 3.1235e+02 "
            "1.8886e+02 4.2250e+02 5.1811e+02 5.6626e+01 "
            "5.9219e+02 3.0671e+02 -3.8343e+00 -1.2813e+00 "
            "7.8470e+01 1.6990e+02 -1.7389e+00 -2.4384e+00 "
            "1.6837e+02 3.3281e+02 6.1567e+02 7.9392e+01 "
            "2.3116e+02 4.0308e+02 -1.2515e+00 -2.1539e+00";
}

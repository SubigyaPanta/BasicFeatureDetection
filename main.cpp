#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


void listKeypoints(vector<KeyPoint> keypoints);

void displayImageKeypoints(Mat &image, vector<KeyPoint> &keypoints, string windowName = "Keypoints");


int main() {
    cout << "Program started ..." << std::endl;
    Mat img1 = imread("../me1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("../me2.jpg", IMREAD_GRAYSCALE);

    // can use SURT::create() or ORB::create() also
    Ptr<Feature2D> f2d = SIFT::create();//SURF::create();//SIFT::create();

    cout << "Getting Keypoints..." << endl;
    // step 1: detect the keypoints
    vector<KeyPoint> KeyPoints1, KeyPoints2;
    f2d->detect(img1, KeyPoints1);
    f2d->detect(img2, KeyPoints2);
    listKeypoints(KeyPoints1);
    displayImageKeypoints(img1, KeyPoints1, "Keypoints 1");
    displayImageKeypoints(img2, KeyPoints2, "Keypoints 2");

    cout << "Getting features..." << endl;
    // step 2: calculate descriptors (feature vectors)
    Mat features1, features2;
    f2d->compute(img1, KeyPoints1, features1);
    f2d->compute(img2, KeyPoints2, features2);

    // just print some statistics returned
    Size size1 = features1.size();
    Size size2 = features2.size();
    cout << "Size of features1 is; " << size1 << endl;
    cout << "Size of features2 is; " << size2 << endl;

    // step 3: Matching features(descriptors) using BFMatcher
    BFMatcher matcher;
    vector<DMatch> matches;
    Mat output, outputKeypoint;
    matcher.match(features1, features2, matches);

    // step 4: showing the matches in window
    drawMatches(img1, KeyPoints1, img2, KeyPoints2, matches, output);
    namedWindow("output", CV_WINDOW_NORMAL);
    imshow("output", output);

    waitKey(0);
}


void listKeypoints(vector<KeyPoint> keypoints) {
    unsigned long size = keypoints.size();
    for(unsigned long i=0; i<size; i++){
        KeyPoint kp = keypoints.at(i);
        cout << "Point is:" << kp.pt << endl;
    }
}

void displayImageKeypoints(Mat &image, vector<KeyPoint> &keypoints, string windowName) {
    Mat output;
    drawKeypoints(image, keypoints, output);
    namedWindow(windowName, CV_WINDOW_NORMAL);
    imshow(windowName, output);
}


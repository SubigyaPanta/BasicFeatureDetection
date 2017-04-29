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

void filterMatchesByDistance(vector<DMatch> originalMatches, vector<DMatch> &filteredMatches,
                             float maxDistanceThreshold, float minDistanceThreshold);

void showMatchesData(vector<DMatch> &filteredMatches, vector<KeyPoint> &query, vector<KeyPoint> &train);

int main() {
    cout << "Program started ..." << std::endl;
    Mat img1 = imread("../me1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("../me2.jpg", IMREAD_GRAYSCALE);

    // can use SURT::create() or ORB::create() also
    Ptr<Feature2D> f2d = SURF::create(800);//SURF::create();//SIFT::create();

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
    BFMatcher matcher(NORM_L2, true);
    vector<DMatch> matches, filteredMatches;
    vector<vector<DMatch>> knnMatches; // just to test two different matches

    Mat output, outputKeypoint;
    matcher.match(features1, features2, matches);
//    matcher.knnMatch(features1, features2, knnMatches, 2);

    // first sort
    sort(matches.begin(), matches.end(), [](DMatch a, DMatch b){return a.distance > b.distance;}); // this is a c++ lambda expression only works in c++11
    // now filter
    float maxdistanceThreshold = 310, minDistanceThreshold = 250;
    //filterMatches(matches, filteredMatches, maxdistanceThreshold, minDistanceThreshold);
    filteredMatches = matches;
    cout << "Normal Match size: " << matches.size() << " Filtered match size: " << filteredMatches.size() << endl;
    //try to see all matches i.e correspoint matching points
    showMatchesData(filteredMatches, KeyPoints1, KeyPoints2);

    // step 4: showing the matches in window
    drawMatches(img1, KeyPoints1, img2, KeyPoints2, filteredMatches, output);
    namedWindow("output", CV_WINDOW_NORMAL);
    imshow("output", output);

    waitKey(0);
}

void showMatchesData(vector<DMatch> &filteredMatches, vector<KeyPoint> &query, vector<KeyPoint> &train) {
    unsigned long size = filteredMatches.size();
    for(unsigned long i=0; i < size; i++){
        DMatch dMatch = filteredMatches.at(i);
        int queryIndex, trainIndex;
        queryIndex = dMatch.queryIdx;
        trainIndex = dMatch.trainIdx;

        // Now see the corresponding points
        cout << "Query Coordinate: " << query.at(queryIndex).pt << " Train Coordinate: " << train.at(trainIndex).pt << endl;
        cout << "Match distance is: " << dMatch.distance << endl;
        cout << "WTF is imgIdx" << dMatch.imgIdx<< endl;
    }
}

void filterMatchesByDistance(vector<DMatch> originalMatches, vector<DMatch> &filteredMatches,
                             float maxDistanceThreshold, float minDistanceThreshold){
    // filter the matches according to some threshold value
    cout << "max: " << maxDistanceThreshold << "min: " << minDistanceThreshold << endl;
    for (vector<DMatch>::iterator i = originalMatches.begin(); i < originalMatches.end(); ++i) {
        cout << "from filter.. distance: " << i->distance << endl;
        if(i->distance > maxDistanceThreshold || i->distance < minDistanceThreshold ){
            filteredMatches.push_back((*i));
            cout << "from filter if.. distance: " << i->distance << endl;
        }
    }
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


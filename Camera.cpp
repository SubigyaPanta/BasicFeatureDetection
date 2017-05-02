#include <opencv2/opencv.hpp>

#include "Camera.h"

using namespace std;
using namespace cv;

Camera::Camera(string sCalibFilename)
{
    FileStorage fs(sCalibFilename, FileStorage::READ);

    string sCalibTime = (string)fs["calibration_time"];
    _cCols = (int)fs["image_width"];
    _cRows = (int)fs["image_height"];
    fs["camera_matrix"] >> _matK;
    fs["distortion_coefficients"] >> _matDistCoeffs;

    cout << "Camera calibrated on " << sCalibTime << endl;
    cout << "Width " << _cCols << " height " << _cRows << " focal length " << _matK.at<double>(0,0) << ", " <<
            _matK.at<double>(1,1) << " principal point (" << _matK.at<double>(0,2) << ", " << _matK.at<double>(1,2) <<
            ")" << endl;
    cout << "Distortion coefficients " << _matDistCoeffs << endl;
}


Camera::~Camera()
{
}


Mat Camera::getK()
{
    return _matK;
}


Mat Camera::getDistCoeffs()
{
    return _matDistCoeffs;
}


double Camera::getFocalX()
{
    return _matK.at<double>(0,0);
}


int Camera::getRows() { return _cRows; }

int Camera::getCols() { return _cCols; }


void Camera::unnormalize(vector<Point3f> &vPointsIn, vector<Point2f> &vPointsOut)
{
    int cPoints = vPointsIn.size();
    vPointsOut.resize(cPoints);
    if (cPoints == 0) return;
    Mat matR = Mat::zeros(3, 1, CV_32F);
    Mat matT = Mat::zeros(3, 1, CV_32F);
    Mat matP;
    cout << "Projecting " << vPointsIn.size() << " points." << endl;
    projectPoints(vPointsIn, matR, matT, _matK, _matDistCoeffs, matP);
    assert(matP.rows == cPoints);
    assert(matP.cols == 1);
    assert(matP.type() == CV_32FC2);
    for (int iPoint = 0; iPoint < cPoints; iPoint++)
    {
        Vec2f point = matP.at<Vec2f>(iPoint, 0);
        vPointsOut[iPoint].x = point[0];
        vPointsOut[iPoint].y = point[1];
    }
}


void Camera::unnormalize(vector<Point2f> &vPointsIn, vector<Point2f> &vPointsOut)
{
    int cPoints = vPointsIn.size();
    vector<Point3f> vPointsCam(cPoints);
    for (int iPoint = 0; iPoint < cPoints; iPoint++)
    {
        vPointsCam[iPoint].x = vPointsIn[iPoint].x;
        vPointsCam[iPoint].y = vPointsIn[iPoint].y;
        vPointsCam[iPoint].z = 1;
    }
    unnormalize(vPointsCam, vPointsOut);
}


void Camera::unnormalize(Point3f pointIn, Point2f &pointOut)
{
    vector<Point3f> vPointsIn(1);
    vPointsIn[0] = pointIn;
    vector<Point2f> vPointsOut(1);
    unnormalize(vPointsIn, vPointsOut);
    pointOut = vPointsOut[0];
}


void Camera::normalize(Point2f pointOrig, Point3f &pointNormalized)
{
    vector<Point2f> vPoints(1);
    vPoints[0] = pointOrig;
    Mat matPointUndistorted;
    undistortPoints(vPoints, matPointUndistorted, _matK, _matDistCoeffs);
    assert(matPointUndistorted.type() == CV_32FC2);
    Vec2f pointUndistorted = matPointUndistorted.at<Vec2f>(0,0);
    pointNormalized = Point3f(pointUndistorted[0], pointUndistorted[1], 1);
}

void Camera::normalize(std::vector<cv::Point2f> &pointsOrig, std::vector<cv::Point2f> &pointsNormalized)
{
    Mat matPointsNormalizedTemp;
    undistortPoints(pointsOrig, matPointsNormalizedTemp, _matK, _matDistCoeffs);
    assert(matPointsNormalizedTemp.type() == CV_32FC2);
    for (int iPoint = 0; iPoint < (int)pointsOrig.size(); iPoint++)
    {
        Vec2f point = matPointsNormalizedTemp.at<Vec2f>(0, iPoint);
        pointsNormalized.push_back(Point2f(point));
    }
}

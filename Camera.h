#ifndef CAMERA_H_
#define CAMERA_H_

#include <string>

class Camera
{

public:
    Camera(std::string sCalibFilename);
    virtual ~Camera();
    cv::Mat getK();
    cv::Mat getDistCoeffs();
    double getFocalX();
    int getRows();
    int getCols();
    void unnormalize(std::vector<cv::Point2f> &vPointsIn, std::vector<cv::Point2f> &vPointsOut);
    void unnormalize(std::vector<cv::Point3f> &vPointsIn, std::vector<cv::Point2f> &vPointsOut);
    void unnormalize(cv::Point3f pointIn, cv::Point2f &pointOut);
    void normalize(cv::Point2f pointOrig, cv::Point3f &pointNormalized);
    void normalize(std::vector<cv::Point2f> &pointsOrig, std::vector<cv::Point2f> &pointsNormalized);

private:
    int _cCols;
    int _cRows;
    cv::Mat _matK;
    cv::Mat _matDistCoeffs;
};

#endif /* CAMERA_H_ */

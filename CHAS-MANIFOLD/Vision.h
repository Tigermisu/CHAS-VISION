#pragma once
#include <iostream>
#include <opencv2\core\core.hpp>

int calcHoughSegmentCuda(char* filename);
std::vector<cv::Vec4i> filterVerticalLines(std::vector<cv::Vec4i> lines, int imgWidth, int angleCompensation, int minVerticalLineDistance);
bool isOverWhiteThreshold(cv::Mat frame, float requiredPercent, int thresholdAmount, bool previewImage);
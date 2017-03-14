#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <opencv2\cudawarping.hpp>
#include <opencv2\cudafilters.hpp>
#include <opencv2\cudaarithm.hpp>
#include <opencv2\imgproc.hpp>
#include <list>
#include "Vision.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	/*
		char* fname = "D:\\1-1.m4v";
		return calcHoughSegmentCuda(fname);
	*/
	if (argc != 5) {
		printf("Usage: program.exe [filepath (absolute or relative)] [requiredPercent (0-1)] [treshold (0-255)] [previewImage (0 or 1)]\n");
		return 0;
	}
	Mat testImage = imread(argv[1]);
	
	isOverWhiteThreshold(testImage, stof(argv[2]), stoi(argv[3]), stoi(argv[4]) == 1) ? printf("Over Threshold") : printf("Not over Threshold.");
	
	return 1;
}

bool isOverWhiteThreshold(Mat frame, float requiredPercent, int thresholdAmount, bool previewImage) {
	Mat dstFrame;
	float ratio;
	cvtColor(frame, dstFrame, COLOR_RGB2GRAY);
	threshold(dstFrame, dstFrame, thresholdAmount, 255, THRESH_BINARY);

	ratio = ((float)countNonZero(dstFrame) / (float)(dstFrame.rows * dstFrame.cols));

	cout << "White Ratio: " << ratio << endl;

	if (previewImage) {
		imshow("Preview", dstFrame);
		while (1) if (waitKey(30) >= 0) break;
	}

	return ratio > requiredPercent;
}

int calcHoughSegmentCuda(char* filename) {
	VideoCapture cap(filename);

	if (!cap.isOpened()) return -1;

	int64 start;
	double timeSec;

	Ptr<cuda::CannyEdgeDetector> cannyEdgeDetector = cuda::createCannyEdgeDetector(0, 30);
	Ptr<cuda::Filter> gaussFilter;
	Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float)(CV_PI / 180.0f), 200, 150);
	Mat frame, cannyFrame, processedHoughMat, tresholdedFrame, greenTreshold, redTreshold;
	vector<Vec4i> lineVector;
	cuda::GpuMat gpuFrame, gpuEdges, gpuLines, gpuPlanes[3], gpuTresholded;


	gaussFilter = cuda::createGaussianFilter(gpuFrame.type(), gpuFrame.type(), Size(29, 29), 1.5);

	for (;;) {

		cap >> frame;
		start = getTickCount();
		gpuFrame = cuda::GpuMat(frame);
		cuda::resize(gpuFrame, gpuFrame, Size(), 0.6, 0.6, INTER_NEAREST);
		cuda::split(gpuFrame, gpuPlanes);
		cuda::cvtColor(gpuFrame, gpuEdges, COLOR_RGB2GRAY);
		gaussFilter->apply(gpuEdges, gpuEdges);
		cannyEdgeDetector->detect(gpuEdges, gpuEdges);	

		hough->detect(gpuEdges, gpuLines);

		if (!gpuLines.empty())
		{
			lineVector.resize(gpuLines.cols);
			Mat h_lines(1, gpuLines.cols, CV_32SC4, &lineVector[0]);
			gpuLines.download(h_lines);
		}

		processedHoughMat = Mat(gpuFrame.rows, gpuFrame.cols, gpuFrame.type(), Scalar(0, 0, 0));

		cout << lineVector.size() << "  ";

		lineVector = filterVerticalLines(lineVector, processedHoughMat.cols, 50, 50);

		cout << lineVector.size() << "  ";

		for (int i = 0; i < lineVector.size(); i++) {
			Vec4i l = lineVector[i];
			line(processedHoughMat, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
		}

		gpuFrame.download(frame);

		gpuEdges.download(cannyFrame);

		for(int i = 0; i < 3; i++)
			gaussFilter->apply(gpuPlanes[i], gpuPlanes[i]);

		gpuPlanes[0].download(tresholdedFrame);
		gpuPlanes[1].download(greenTreshold);
		gpuPlanes[2].download(redTreshold);

		threshold(tresholdedFrame, tresholdedFrame, 200, 255, THRESH_BINARY);
		threshold(greenTreshold, greenTreshold, 200, 255, THRESH_BINARY);
		threshold(redTreshold, redTreshold, 200, 255, THRESH_BINARY);

		tresholdedFrame -= greenTreshold;
		tresholdedFrame -= redTreshold;

		medianBlur(tresholdedFrame, tresholdedFrame, 5);
		medianBlur(tresholdedFrame, tresholdedFrame, 5);

		//gpuTresholded = cuda::GpuMat(tresholdedFrame);

		frame += processedHoughMat;

				
		timeSec = (getTickCount() - start) / getTickFrequency();
		cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;

		imshow("tresholded", tresholdedFrame);
		//imshow("gpuhough", processedHoughMat);
		//imshow("canny", cannyFrame);
		imshow("gpucanny", frame);

		if (waitKey(30) >= 0) break;
	}

	return 0;
	
}



std::vector<cv::Vec4i> filterVerticalLines(std::vector<cv::Vec4i> lines, int imgWidth, int angleCompensation = 50, int minVerticalLineDistance = 15) {
	const int aCD = angleCompensation; // angleCompensationDeviation, compensates for the distortion on the edges of the image
	list<Vec4i> lineList;
	for (int i = 0; i < lines.size(); i++) lineList.push_back(lines[i]);

	for (list<Vec4i>::iterator it = lineList.begin(); it != lineList.end();) {
		Vec4i current = *it;
		float xMidpoint = (current[0] + current[2]) / 2,
			angle = atan(abs(current[3] - current[1]) / (abs(current[0] - current[2]) + 0.0001f)) * 180 / CV_PI,
			xDistribution = xMidpoint / imgWidth,
			offset = -4 * aCD * xDistribution * xDistribution + 4 * aCD * xDistribution - aCD,
			sqrMag;
		if (angle < 70 + offset) { // Filter out horizontal lines
			it = lineList.erase(it);
		}
		else {
			sqrMag = powf(current[0] + current[2], 2) + powf(current[1] + current[3], 2);
			for (list<Vec4i>::iterator innerIt = lineList.begin(); innerIt != lineList.end();) {
				if (it != innerIt) {
					Vec4i candidate = *innerIt;
					float candidateXMidpoint = (candidate[0] + candidate[2]) / 2,
						candidateSqrMag;
					if (abs(candidateXMidpoint - xMidpoint) < minVerticalLineDistance) {
						candidateSqrMag = powf(candidate[0] + candidate[2], 2) + powf(candidate[1] + candidate[3], 2);
						if (candidateSqrMag < sqrMag) {
							innerIt = lineList.erase(innerIt);
							continue;
						}
					}
				}
				innerIt++;
			}
			it++;
		}
	}

	lines.resize(lineList.size());

	int i = 0;
	for (list<Vec4i>::iterator it = lineList.begin(); it != lineList.end(); it++) {
		Vec4i current = *it;
		lines[i] = current;
		i++;
	}

	return lines;
}
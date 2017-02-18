#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <opencv2\cudawarping.hpp>
#include <opencv2\cudafilters.hpp>
#include <opencv2\cudaarithm.hpp>
#include <list>

using namespace cv;
using namespace std;

int calcCanny(char *filename) {
	VideoCapture cap(filename); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	int64 start;
	double timeSec;

	namedWindow("edges", 1);
	for (;;)
	{
		Mat edges;
		Mat frame;
		Mat planes[3];
		cap >> frame; // get a new frame from camera
		start = getTickCount();
		resize(frame, frame, Size(), 0.4, 0.4, INTER_NEAREST);
		cvtColor(frame, edges, COLOR_BGR2GRAY);
		split(frame, planes);
		edges = planes[1];
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30);
		timeSec = (getTickCount() - start) / getTickFrequency();
		cout << "CPU Time : " << timeSec * 1000 << " ms" << endl;

		imshow("edges", edges);
		if (waitKey(30) >= 0) break;
	}
	return 0;
}

int calcHoughSegmentCuda(char* filename) {
	VideoCapture cap(filename);

	if (!cap.isOpened()) return -1;

	int64 start;
	double timeSec;

	Ptr<cuda::CannyEdgeDetector> cannyEdgeDetector = cuda::createCannyEdgeDetector(0, 30);
	Ptr<cuda::Filter> gaussFilter;
	Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float)(CV_PI / 180.0f), 200, 150);
	Mat frame, cannyFrame, processedHoughMat;
	vector<Vec4i> lineVector;
	cuda::GpuMat gpuFrame, gpuEdges, gpuLines;


	gaussFilter = cuda::createGaussianFilter(gpuFrame.type(), gpuFrame.type(), Size(29, 29), 1.5);

	for (;;) {

		cap >> frame;
		start = getTickCount();
		gpuFrame = cuda::GpuMat(frame);
		cuda::resize(gpuFrame, gpuFrame, Size(), 0.6, 0.6, INTER_NEAREST);
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

		//Process line vectors
		const int aCD = 50; // angleCompensationDeviation, compensates for the distortion on the edges of the image
		list<Vec4i> lineList;
		for (int i = 0; i < lineVector.size(); i++) lineList.push_back(lineVector[i]);

		for (list<Vec4i>::iterator it = lineList.begin(); it != lineList.end();) {
			Vec4i current = *it;
			float xMidpoint = (current[0] + current[2]) / 2,
				angle = atan(abs(current[3] - current[1]) / (abs(current[0] - current[2]) + 0.0001f)) * 180 / CV_PI,
				xDistribution = xMidpoint / processedHoughMat.cols,
				offset = -4 * aCD * xDistribution * xDistribution + 4 * aCD * xDistribution - aCD,
				sqrMag;
			if (angle < 70 + offset) { // Filter out horizontal lines
				it = lineList.erase(it);
			} else {
				sqrMag = powf(current[0] + current[2], 2) + powf(current[1] + current[3], 2);
				for (list<Vec4i>::iterator innerIt = lineList.begin(); innerIt != lineList.end();) {
					if (it != innerIt) {
						Vec4i candidate = *innerIt;
						float candidateXMidpoint = (candidate[0] + candidate[2]) / 2,
							candidateSqrMag;
						if (abs(candidateXMidpoint - xMidpoint) < 15) {
							candidateSqrMag = powf(candidate[0] + candidate[2], 2) + powf(candidate[1] + candidate[3], 2);
							if(candidateSqrMag < sqrMag) {
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

		for (list<Vec4i>::iterator it = lineList.begin(); it != lineList.end(); it++) {
			Vec4i l = *it;
			float angle = atan(abs(l[3] - l[1]) / (abs(l[0] - l[2]) + 0.0001f)) * 180 / CV_PI,
				xMidpoint = (l[0] + l[2]) / 2,
				xDistribution = xMidpoint / processedHoughMat.cols,
				offset = -4 * aCD * xDistribution * xDistribution + 4 * aCD * xDistribution - aCD;
			if (angle > 70 + offset) { // Filter out horizontal lines
				line(processedHoughMat, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
			}
		}

		gpuFrame.download(frame);

		gpuEdges.download(cannyFrame);

		frame += processedHoughMat;

				
		timeSec = (getTickCount() - start) / getTickFrequency();
		cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;

		//imshow("gpuhough", processedHoughMat);
		imshow("canny", cannyFrame);
		imshow("gpucanny", frame);

		if (waitKey(30) >= 0) break;
	}

	return 0;
	
}

int calcHoughLinesCuda(char* filename) {
	VideoCapture cap(filename); // open the default camera

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	int64 start;
	double timeSec;

	Ptr<cuda::CannyEdgeDetector> cannyEdgeDetector = cuda::createCannyEdgeDetector(0, 30);
	Ptr<cuda::Filter> gaussFilter;
	Ptr<cuda::HoughLinesDetector> hough = cuda::createHoughLinesDetector(1.0f, (float)(CV_PI / 180.0f), 30, 250);
	Mat frame, processedHoughMat, rawHough;
	cuda::GpuMat gpuFrame, gpuEdges, gpuLines;
	cuda::GpuMat* gpuPlanes = new cuda::GpuMat[3];
	
	gaussFilter = cuda::createGaussianFilter(gpuFrame.type(), gpuFrame.type(), Size(7, 7), 3);

	

	for (;;) {
		vector<Vec2i> lineVector;
		cap >> frame;
		start = getTickCount();
		gpuFrame = cuda::GpuMat(frame);
		cuda::split(gpuFrame, gpuPlanes);
		gpuEdges = gpuPlanes[1];
		cuda::resize(gpuEdges, gpuEdges, Size(), 0.35, 0.35, INTER_NEAREST);
		gaussFilter->apply(gpuEdges, gpuEdges);
		cannyEdgeDetector->detect(gpuEdges, gpuEdges);
		hough->detect(gpuEdges, gpuLines);

		cout << gpuLines.empty();
		lineVector.resize(gpuLines.cols);
		Mat temp_Mat(1, gpuLines.cols, CV_32SC2, &lineVector[0]);
		gpuLines.download(temp_Mat);

		processedHoughMat = Mat(gpuEdges.rows, gpuEdges.cols, frame.type());

		cout << lineVector[50]; // For some reason, the array isn't being downloaded into the line vector.

		/*
		for (int i = 0; i < lineVector.size(); i++)
		{			
			float rho = lineVector[i][0], theta = lineVector[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;

			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));

			line(processedHoughMat, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
		}
		*/

		gpuEdges.download(frame);

		timeSec = (getTickCount() - start) / getTickFrequency();
		cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;

		imshow("gpuhough", processedHoughMat);
		imshow("gpucanny", frame);

		if (waitKey(30) >= 0) break;
	}

	return 0;

}

int houghAlternate(char* filename) {
	VideoCapture cap(filename);

	if (!cap.isOpened()) return -1;

	int64 start;
	double timeSec;

	Ptr<cuda::CannyEdgeDetector> cannyEdgeDetector = cuda::createCannyEdgeDetector(0, 30);
	Ptr<cuda::Filter> gaussFilter;
	Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float)(CV_PI / 180.0f), 300, 50);
	Mat frame, cannyFrame, processedHoughMat;
	vector<Vec4i> lineVector;
	cuda::GpuMat gpuFrame, gpuEdges, gpuLines;


	gaussFilter = cuda::createGaussianFilter(gpuFrame.type(), gpuFrame.type(), Size(7, 7), 3);

	for (;;) {

		cap >> frame;
		start = getTickCount();
		gpuFrame = cuda::GpuMat(frame);
		cuda::resize(gpuFrame, gpuFrame, Size(), 0.5, 0.5, INTER_NEAREST);
		cuda::cvtColor(gpuFrame, gpuEdges, COLOR_RGB2GRAY);
		gaussFilter->apply(gpuEdges, gpuEdges);
		cannyEdgeDetector->detect(gpuEdges, gpuEdges);

		gpuEdges.download(cannyFrame);

		processedHoughMat = Mat(gpuFrame.rows, gpuFrame.cols, gpuFrame.type(), Scalar(0, 0, 0));

		vector<Vec2f> lines;
		HoughLines(cannyFrame, lines, 1, CV_PI / 180, 200);

		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			if (theta > CV_PI / 180 * 85 && theta < CV_PI / 180 * 95) {
				Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000 * (-b));
				pt1.y = cvRound(y0 + 1000 * (a));
				pt2.x = cvRound(x0 - 1000 * (-b));
				pt2.y = cvRound(y0 - 1000 * (a));
				line(processedHoughMat, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
			}
		}

		gpuFrame.download(frame);

		frame += processedHoughMat;

		timeSec = (getTickCount() - start) / getTickFrequency();
		cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;

		//imshow("gpuhough", processedHoughMat);
		imshow("gpucanny", frame);

		if (waitKey(30) >= 0) break;
	}

	return 0;

}

int main(int argc, char **argv) {
	char* fname = "D:\\1-1.m4v";
	//calcCanny(fname);
	return calcHoughSegmentCuda(fname);
	//return calcHoughLinesCuda(fname);
	//return houghAlternate(fname);
}
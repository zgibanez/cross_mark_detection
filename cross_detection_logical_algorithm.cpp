//ALGORITHM:
//0-- Load template
//1-- Find keypoints on template
//2-- Load frame and preproccess it for better shape recognition
//3-- Find squares and their quadrants on frame
//4-- Match the quadrants with the template 
//5-- If there is no good candidate (low match percentage or not squares found)
// find the brightest region of frame and mark it as "possible mark"

#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

//Additional functions
vector<vector<Point>> FindSquares(Mat * thresh, Mat * frame);
vector<double> ExtractRegionMean(vector<vector<Point>>candidates, Mat *frame);
void DrawPoligon(vector<Point> poligon, Mat *frame, Scalar color);
vector<float> ExtractSlopeCoefficent(vector<vector<Point>> candidates);

int main(int argc, char** argv)
{

	///-- STEP 1 : Extract features from template

	//load template and blur it
	Mat img = imread(argv[1], 0);
	GaussianBlur(img, img, Size(3, 3), 4);

	if (img.empty()) {
		cout << "can't read image" << endl;
		return -1;
	}


	/// -- STEP 2 : Load video and binarize frame

	//load video
	VideoCapture capture("C:/images/video4.mp4");
	if (!capture.isOpened())
		throw "Error: cannot read video";

	for (; ; )
	{
		Mat frame, frame_gray, frame_thresh;

		//load frame
		capture >> frame;
		if (frame.empty()) break;

		//blur it, sharpen it, threshold it
		GaussianBlur(frame, frame, Size(3, 3), 4);
		addWeighted(frame, 1.5, frame, -0.5, 0, frame);
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		adaptiveThreshold(frame_gray, frame_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 2);

		///-- STEP 3: Find squares and score them
		vector<vector<Point>> candidates;
		vector<float> slope_coefficents;

		candidates = FindSquares(&frame_thresh, &frame);
		if (!candidates.empty())
		{
			//Extract slope coefficents
			slope_coefficents = ExtractSlopeCoefficent(candidates);
			cout << "SLOPE COEFFICENTS EXTRACTED" << endl;

			// Extract region means
			vector<double> region_means;
			vector<Point> best_candidate;
			double highest_mean = 0;
			region_means = ExtractRegionMean(candidates, &frame);

			for (int i = 0; i < slope_coefficents.size(); i++)
			{
				//TEST
				if (region_means[i] > highest_mean) {
					best_candidate = candidates[i];
				}
				//TEST

				Moments M; float cX, cY;
				M = moments(candidates[i]);
				cX = M.m10 / M.m00;
				cY = M.m01 / M.m00;
				Point center = Point(cX, cY);
				putText(frame, to_string(slope_coefficents[i]),center,FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,Scalar(0,255,0));
				//DrawPoligon(candidates[i], &frame, Scalar(255,0,0));
			}

			//Paint with red best candidate
			DrawPoligon(best_candidate, &frame, Scalar(0, 0, 255));


		}

		imshow("frame", frame);

		//exit with ESC key
		if (waitKey(0) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
}

//This function returns the 4 sided poligons found
// in the frame
vector<vector<Point>> FindSquares(Mat * thresh, Mat * frame)
{
	vector<vector<Point>> cnts;
	vector<Point> poligon;
	vector<vector<Point>> candidates;
	Mat frame_thresh_copy = thresh->clone();

	//preprocess image
	dilate(frame_thresh_copy, frame_thresh_copy, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	erode(frame_thresh_copy, frame_thresh_copy, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	GaussianBlur(frame_thresh_copy, frame_thresh_copy, Size(3, 3), 4);

	//find contours on the frame
	Mat canny;
	cout << " Buscando contornos... ";
	Canny(frame_thresh_copy.clone(), canny, 250, 250 * 3);
	findContours(canny, cnts, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	cout << cnts.size() << "  contornos detectados" << endl;

	//approximate contours to poligons
	int i;
	double peri;
	float MinArea = 500;

	for (i = 0; i < cnts.size(); i++) {
		peri = arcLength(cnts[i], 1);
		approxPolyDP(cnts[i], poligon, peri*0.05, 1);

		//find contours with 4 sides and
		//a minimun area
		Moments M;
		M = moments(cnts[i], true);
		if (poligon.size() == 4)
		{
			if (M.m00 > MinArea) {
				candidates.push_back(poligon);
			}
		}
	}

	//return the possible candidates
	if (candidates.empty()) cout << "No se han encontrado candidatos" << endl;
	else					cout << candidates.size() << "  candidatos encontrados" << endl;
	return candidates;
}

//This function determines wheter there is a
//circle in the contour or not
vector<double> ExtractRegionMean(vector<vector<Point>>candidates, Mat *frame) {
	vector<double> region_means;

	for (int i = 0; i < candidates.size(); i++) {

		//Create a mask to isolate each square
		Rect roi = boundingRect(candidates[i]);
		Mat mask = Mat::zeros(frame->size(), CV_8UC1);
		drawContours(mask, candidates, i, Scalar(255), CV_FILLED);
		
		//Isolate the contour
		Mat img_roi;
		Mat contour_region;
		frame->copyTo(img_roi, mask);
		contour_region = img_roi(roi);

		//Find mean
		Scalar temp_value = mean(contour_region);
		region_means.push_back(temp_value[0]);
	}

	return region_means;
}

//This function returns a coefficent <=1 describing how parallel
//are the lines of the squares found
vector<float> ExtractSlopeCoefficent(vector<vector<Point>> candidates) {
	
	vector<float> slope_coefficents;

	for (int i = 0; i < candidates.size(); i++) {
		//find slope between opposite sides
		// slope = (y2 - y1) / (x2 - x1)
		float k1a, k1b, k2a, k2b;
		float maxk1, mink1, maxk2, mink2;
		float c1, c2;
			k1a = abs((candidates[i][1].y- candidates[i][0].y) / (float)(candidates[i][1].x - candidates[i][0].x));
			k2a = abs((candidates[i][2].y - candidates[i][1].y) / (float)(candidates[i][2].x - candidates[i][1].x));
			k1b = abs((candidates[i][3].y - candidates[i][2].y) / (float)(candidates[i][3].x - candidates[i][2].x));
			k2b = abs((candidates[i][0].y - candidates[i][3].y) / (float)(candidates[i][0].x - candidates[i][3].x));

			//take the best coefficent of the square found
			maxk1 = max(k1a, k1b); mink1 = min(k1a, k1b);
			maxk2 = max(k2a, k2b); mink2 = min(k2a, k2b);
			c1 = mink1 / maxk1; c2 = mink2 / maxk2;
			slope_coefficents.push_back(max(c1, c2));
	}

	return slope_coefficents;
}

//This function recieves an set of points and draws
//on the Mat "frame" the poligon they form
void DrawPoligon(vector<Point> poligon, Mat *frame, Scalar color) {

	for (int i = 0; i < poligon.size(); i++) {
		putText(*frame, "P", Point(poligon[i].x, poligon[i].y), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0));
		if (i<poligon.size() - 1)
			line(*frame, poligon[i], poligon[i + 1], color);
		else
			line(*frame, poligon[poligon.size() - 1], poligon[0], color);
	}

}

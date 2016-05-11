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
Mat FindMark(Mat* thresh, Mat* frame);
void DrawOutline(vector<Point> poligon, Mat *frame);

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

	//feature detection variables
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	//set feature detector
	Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create(0, 3, 0.04, 10, 0.8);
	detector->detect(img, keypoints_1);

	//set descriptor extractor
	Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
	extractor->compute(img, keypoints_1, descriptors_1);

	//show keypoints detected
	Mat display = img.clone();
	drawKeypoints(display, keypoints_1, display);
	imshow("display", display);
	waitKey(-1);

	/// -- STEP 2 : Load video and binarize frame

	//load video
	VideoCapture capture("C:/images/video2.mp4");
	if (!capture.isOpened())
		throw "Error: cannot read video";

	for (; ; )
	{
		Mat frame, frame_gray, frame_thresh;

		//load frame, binarize it
		capture >> frame;
		if (frame.empty()) break;

		GaussianBlur(frame, frame, Size(3, 3), 4);
		addWeighted(frame, 1.5, frame, -0.5, 0, frame);
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		adaptiveThreshold(frame_gray, frame_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 2);
		imshow("Adaptative threshold", frame_thresh);

		///-- STEP 3: Find in the thresholded frame the mark's outline with shape detection
		/// we will use the outline as our ROI
		Mat ROI;
		ROI = FindMark(&frame_thresh, &frame);

		/// -- STEP 4 : Extract features of the ROI
		detector->detect(ROI, keypoints_2);
		extractor->compute(ROI, keypoints_2, descriptors_2);
		cout << "Keypoints de frame computados" << endl;

		/// -- STEP 5 : Find and filter matches
		BFMatcher matcher;

		//if the frame descriptor is empty
		//skip matching
		if (!descriptors_2.empty())
		{
			//Filter 1: Lowe's ratio
			std::vector<vector<cv::DMatch>> matches;
			matcher.knnMatch(descriptors_1, descriptors_2, matches, 5);
			vector<DMatch> good_matches;
			const float ratio = 0.8;
			for (int i = 0; i < matches.size(); i++)
			{
				if (matches[i][0].distance < ratio * matches[i][1].distance)
				{
					good_matches.push_back(matches[i][0]);
				}

			}
			cout << good_matches.size() << " puntos filtrados con Lowe's Ratio " << endl;


			//Filter 2: Cross checking
			vector<DMatch> matches12, matches21, good_matches2;
			matcher.match(descriptors_1, descriptors_2, matches12);
			matcher.match(descriptors_2, descriptors_1, matches21);

			for (size_t i = 0; i < matches12.size(); i++)
			{
				DMatch forward = matches12[i];
				DMatch backward = matches21[forward.trainIdx];
				if (backward.trainIdx == forward.queryIdx)
					good_matches2.push_back(forward);
			}

			cout << good_matches2.size() << " puntos filtrados con Cross-checking " << endl;

			//Filter 3: Symmetry Test

			///--Step 6: Present data
			Mat img_matches;
			drawMatches(img, keypoints_1, ROI, keypoints_2, good_matches2, img_matches, Scalar(0, 255, 0), Scalar(0, 255, 0),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			imshow("matches", img_matches);
			imshow("frame", frame);

			//empty vectors
			good_matches.clear();
			good_matches2.clear();
			matches.clear();
			keypoints_2.clear();
		}
		else {
			cout << "El descriptor del frame estaba vacio" << endl;
			imshow("ROI", ROI);
		}

		//exit with ESC key
		if (waitKey(-1) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
}

//This function returns a ROI 
//enclosing the biggest 4 sided shape it finds
Mat FindMark(Mat * thresh, Mat * frame)
{
	vector<vector<Point>> cnts;
	vector<Point> poligon, best_candidate;
	Rect rectBoundedToMark;
	Mat frame_thresh_copy = thresh->clone();

	//preprocess image
	dilate(frame_thresh_copy, frame_thresh_copy, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	erode(frame_thresh_copy, frame_thresh_copy, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
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
	float MinArea = 300;
	bool ROI_found = false;

	for (i = 0; i < cnts.size(); i++) {
		peri = arcLength(cnts[i], 1);
		approxPolyDP(cnts[i], poligon, peri*0.03, 1);

		//find the contour with 4 or 5 sides and the biggest area
		Moments M;
		M = moments(cnts[i], true);
		if (poligon.size() == 4 || poligon.size() == 5)
		{
			if (M.m00 > MinArea) {
				best_candidate = poligon;
				MinArea = M.m00;
				ROI_found = true;
			}
		}
	}

	//Draw the result
	if (!best_candidate.empty()) {
		DrawOutline(best_candidate, frame);
	}

	//Return a Rect enclosing the biggest rectangle
	//and prevent assertion errors
	//if there is no candidate
	//search on the whole image
	Mat frame_copy = frame->clone();
	rectBoundedToMark = boundingRect(best_candidate);

	if (ROI_found) {
		cout << "Marca encontrada" << endl;
		return frame_copy(rectBoundedToMark);
	}
	else {
		cout << "Marca no encontrada" << endl;
		return frame_copy;
	}
}

//This function recieves an set of points and draws
//on the Mat "frame" the poligon they form
void DrawOutline(vector<Point> poligon, Mat *frame) {

		for (int i = 0; i < poligon.size(); i++) {
			putText(*frame, "P", Point(poligon[i].x, poligon[i].y), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0));
			if (i<poligon.size() - 1)
				line(*frame, poligon[i], poligon[i + 1], Scalar(255, 0, 0));
			else
				line(*frame, poligon[poligon.size() - 1], poligon[0], Scalar(255, 0, 0));
		}
}

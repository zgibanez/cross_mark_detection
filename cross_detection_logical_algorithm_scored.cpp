/* ALGORITHM:

1-- Load frame and preproccess it

2-- If mark==NOT_FOUND or mark==NO_CANDIDATES:
	2.1-- Find squares(candidates) on frame
	2.2-- Evaluate the slope coefficent and the mean value of each square
	2.3-- Sum scores and take the best one if it passes the minimum score
	2.4a-- If it doesn't pass, go to step 1. mark==NO_CANDIDATES
	2.4b-- If it passes, extract its histogram. mark==FOUND

3-- If mark== FOUND
	3.1 Use CamShift to track the histogram of best candidate.
	3.2 If tracking window dimentions == 0, mark==NOT_FOUND
*/


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
int				ScoreSquares(vector<vector<Point>> candidates, Mat* frame);
void			TrackBestCandidate(vector<Point> best_candidate, Mat backproj, Mat *frame);
Mat				ExtractRegion(vector<Point> poligon, Mat * frame);
vector<double>	ExtractRegionMean(vector<vector<Point>>candidates, Mat *frame);
Mat				ExtractBackProjection(vector<Point> best_candidate, Mat *frame);
vector<float>	ExtractSlopeCoefficent(vector<vector<Point>> candidates);
void			DrawPoligon(vector<Point> poligon, Mat *frame, Scalar color);
void			DrawCandidates(vector<vector<Point>> candidates, Mat * frame, int idx);

//Global variables
enum Status {FOUND, NOT_FOUND, NO_CANDIDATES, TRACKED};
Status mark_status = NOT_FOUND;

int main(int argc, char** argv)
{

	//load video
	VideoCapture capture("C:/images/video6.mp4");
	if (!capture.isOpened())
		throw "Error: cannot read video";

	for (; ; )
	{
		Mat frame, frame_gray, frame_thresh, frame_copy;

		//load frame
		capture >> frame;
		frame_copy = frame.clone();
		if (frame.empty()) break;

		//blur it, sharpen it, threshold it
		GaussianBlur(frame, frame, Size(3, 3), 4);
		addWeighted(frame, 1.5, frame, -0.5, 0, frame);
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		adaptiveThreshold(frame_gray, frame_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 2);
		imshow("threshold", frame_thresh);

		vector<vector<Point>> candidates;
		static vector<Point> best_candidate; 
		int idx;
		Mat backproj;
		
		//is mark is to be found, search for candidates
		//evaluate them, and select the best of them
		if (mark_status == NOT_FOUND || mark_status == NO_CANDIDATES) {
			
			candidates = FindSquares(&frame_thresh, &frame);

			if (!candidates.empty()) {
				idx = ScoreSquares(candidates, &frame);
				if (mark_status != NO_CANDIDATES) {
					best_candidate = candidates[idx];
					DrawCandidates(candidates, &frame_copy, idx);
					imshow("candidates", frame_copy);
				}
			}
		}
	
		if (mark_status == FOUND || mark_status == TRACKED) {
			backproj = ExtractBackProjection(best_candidate, &frame);
			TrackBestCandidate(best_candidate, backproj, &frame);
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
	erode(frame_thresh_copy, frame_thresh_copy, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	dilate(frame_thresh_copy, frame_thresh_copy, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	GaussianBlur(frame_thresh_copy, frame_thresh_copy, Size(3, 3), 4);

	//find contours on the frame
	Mat canny;
	cout << " Buscando contornos... ";
	Canny(frame_thresh_copy.clone(), canny, 250, 250 * 3);
	//findContours(canny, cnts, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	findContours(canny, cnts, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_KCOS);
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
	if (candidates.empty()) cout << "FindSquares: No candidates found" << endl;
	else					cout << "FindSquares: " << candidates.size() << "  candidates found" << endl;
	return candidates;
}

// This function returns the index of the candidate that
//scores more points.
// If none of the candidates have a minimun score it returns a -1.
int ScoreSquares(vector<vector<Point>> candidates, Mat* frame) {

	vector<float> slope_coefficents;
	slope_coefficents = ExtractSlopeCoefficent(candidates);

	vector<double> region_means;
	region_means = ExtractRegionMean(candidates, frame);

	float highest_score = 0, temp_score;
	int idx=-1;
	for (int i = 0; i < candidates.size(); i++) {
		temp_score = slope_coefficents[i] + region_means[i];
		if (temp_score > highest_score) {
			highest_score = temp_score;
			idx = i;
		}
		cout << "ScoreSquares: Score for square number " << i << ": " << temp_score << endl;
	}

	//if none of the candidates passes the minimun score
	//then we suppose the mark is not found
	if (highest_score < 0.3) {
		idx = -1;
		mark_status = NO_CANDIDATES;
		cout << "ScoreSquares: None of the candidates passed the test" << endl;
	}
	else {
		cout << "ScoreSquares: A suitable candidate was found" << endl;
		mark_status = FOUND;
	}
	
	return idx;
}

//This function uses Camshift algorithm to track
//the best candidate
void TrackBestCandidate(vector<Point> best_candidate, Mat backproj, Mat *frame) {

	TermCriteria  term_crit(CV_TERMCRIT_EPS | CV_TERMCRIT_NUMBER, 10, 1);
	RotatedRect camshift_track_window;
	Point2f track_window_points[4];

	camshift_track_window = CamShift(backproj, boundingRect(best_candidate), term_crit);
	camshift_track_window.points(track_window_points);

	if (camshift_track_window.size.height > 0 && camshift_track_window.size.width > 0) {
		for (int j = 0; j < 4; j++) {
			line(*frame, track_window_points[j], track_window_points[(j + 1) % 4], Scalar(250, 200, 100));
		}
		cout << "TrackBestCandidate: Best candidate successfully tracked" << endl;
	}
	else
	{
		mark_status = NOT_FOUND;
		cout << "TrackBestCandidate: Best candidate lost. Reseting algorithm..." << endl;
	}
	

}

//This function takes the mean value of each of the squares found.
//A high mean scores more points than a low mean.
vector<double> ExtractRegionMean(vector<vector<Point>>candidates, Mat *frame) {
	vector<double> region_means;

	for (int i = 0; i < candidates.size(); i++) {

		//Create a mask to isolate each square
		Mat contour_region = ExtractRegion(candidates[i], frame);

		//Store the normalized value of mean in the output vector
		Scalar temp_value = mean(contour_region);
		region_means.push_back(temp_value[2]/255);
	}

	cout << "ExtractRegionMean: All candidates region means were evaluated" << endl;
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

			//coefficents = slope1/slope2;
			//take the best coefficent of the square found
			maxk1 = max(k1a, k1b); mink1 = min(k1a, k1b);
			maxk2 = max(k2a, k2b); mink2 = min(k2a, k2b);
			c1 = mink1 / maxk1; c2 = mink2 / maxk2;
			slope_coefficents.push_back(max(c1, c2));
	}

	cout << "ExtractSlopeCoefficents: All candidates slope coefficents were evaluated" << endl;
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

//This function draws the best candidate in red
// and the rest of them in blue
void DrawCandidates(vector<vector<Point>> candidates, Mat * frame, int idx) {

	if (!candidates.empty())
	{

		for (int i = 0; i < candidates.size(); i++)
		{
			Moments M; float cX, cY;
			M = moments(candidates[i]);
			cX = M.m10 / M.m00;
			cY = M.m01 / M.m00;
			Point center = Point(cX, cY);
			/////SPOT FOR PUTTING SCORE
			if (i != idx) DrawPoligon(candidates[i], frame, Scalar(255, 0, 0));
		}

		DrawPoligon(candidates[idx], frame, Scalar(0, 0, 255));
	}

}

//This function returns a Mat with
//the isolated region enclosed by poligon
Mat ExtractRegion(vector<Point> poligon, Mat * frame) {

	vector < vector<Point>> poligons;
	poligons.push_back(poligon);

	//Create a mask to isolate the contour
	Rect roi = boundingRect(poligon);
	Mat mask = Mat::zeros(frame->size(), CV_8UC1);
	drawContours(mask, poligons, -1, Scalar(255), CV_FILLED);

	//Isolate the contour
	Mat img_roi;
	Mat contour_region;
	frame->copyTo(img_roi, mask);
	contour_region = img_roi(roi);
	imshow("ROI used", img_roi);

	return contour_region;
}

//This function returns the back projection of the best candidate
Mat ExtractBackProjection(vector<Point> best_candidate, Mat *frame) {

	Mat roi;
	static Mat roi_hist;
	Rect rect_roi;
	int hbins = 30, vbins = 30;
	int ch[] = { 0, 1, 2 };
	int histSize[] = { hbins, vbins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 255 };
	float vranges[] = { 0, 255 };
	const float* ranges[] = { hranges, sranges, vranges };
	
	if (mark_status==FOUND) {
		roi = ExtractRegion(best_candidate, frame);
		cvtColor(roi, roi, CV_BGR2HSV);
		//normalize hue histogram
		calcHist(&roi, 1, ch, Mat(), roi_hist, 2, histSize, ranges, true, false);
		normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX, -1, noArray());
		mark_status = TRACKED;
	}
	
	//extract back projection
	Mat backproj;
	Mat frameHSV = frame->clone();
	cvtColor(frameHSV, frameHSV, CV_BGR2HSV);
	calcBackProject(&frameHSV, 1, ch, roi_hist, backproj, ranges);

	//process backprojection to get better results
	erode(backproj, backproj, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(backproj, backproj, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	addWeighted(backproj, 2, backproj, -0.5, 0, backproj);
	cout << "ExtractBackProjection: Back projection of best candidate evaluated" << endl;
	imshow("processed backprojection", backproj);

	return backproj;
}

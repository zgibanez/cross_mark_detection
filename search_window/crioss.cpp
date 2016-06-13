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
vector<vector<Point>> FindSquares(Mat * thresh, Mat * frame, Mat * frame_edges);
int				ScoreSquares(vector<vector<Point>> candidates, Mat* frame, double* mean, int * area, Mat * frame_edges);
Mat				ExtractRegion(vector<Point> poligon, Mat * frame, Mat * pmask = NULL);
vector<double>	ExtractRegionMean(vector<vector<Point>>candidates, Mat *frame);
vector<float>	ExtractSlopeCoefficent(vector<vector<Point>> candidates);
void			DetectCross(vector<vector<Point>> candidates, Mat * frame_edges);
Rect			CreateSearchWindow(vector<Point> best_candidate, Mat frame, Rect previous_search_window=Rect());
void			DrawPoligon(vector<Point> poligon, Mat *frame, Scalar color, Point offset = Point(0,0));
void			DrawCandidates(vector<vector<Point>> candidates, Mat * frame, int idx, Rect search_window = Rect());
vector<Point>	TrackBestCandidate(Mat thresh, int original_area);

//Global variables
enum Status { FOUND, NOT_FOUND, TRACKED };
Status mark_status = NOT_FOUND;
//time variables
int e1, e2, e3, e4;
float time;

int main(int argc, char** argv)
{

	//load video
	VideoCapture capture("C:/images/video6.mp4");
	if (!capture.isOpened())
		throw "Error: cannot read video";

	for (; ; )
	{
		Mat frame, frame_copy;
		static Rect search_window;
		static double candidate_mean;
		static int candidate_area;

		//measure start time
		e1 = getTickCount();

		//load frame
		capture >> frame;
		if (frame.empty()) break;
		frame_copy = frame.clone(); /*for displaying results without altering original frame*/

		//crop search window if applicable
		if (mark_status == TRACKED) {
			frame = frame(search_window);
		}

		//preproccess frame to highlight contours
		Mat frame_gray, frame_thresh;

		GaussianBlur(frame, frame, Size(3, 3), 4);
		addWeighted(frame, 1.5, frame, -0.5, 0, frame);
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

		//create an adaptative threshold if mark is to be found
		//if mark is found, put a standar threshold with maxmin-values around the mean of the previous best candidate
		if (mark_status == NOT_FOUND) adaptiveThreshold(frame_gray, frame_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 23, 3);
		else { threshold(frame_gray, frame_thresh, (1 - 0.1*candidate_mean) * 255, (1 + 0.1*candidate_mean) * 255, THRESH_BINARY); 
		}
		imshow("threshold", frame_thresh);

		vector<vector<Point>> candidates;
		static vector<Point> best_candidate;
		int idx;

		//search for candidates if mark is not found
		//evaluate them, and select the best of them
		if (mark_status == NOT_FOUND) {
			Mat frame_edges;
			candidates = FindSquares(&frame_thresh, &frame, &frame_edges);

			if (!candidates.empty()) {
				idx = ScoreSquares(candidates, &frame, &candidate_mean, &candidate_area, &frame_edges);
				if (idx != -1) {
					best_candidate = candidates[idx];
					DrawCandidates(candidates, &frame_copy, idx, search_window);
				}
			}
		}
		else {
			best_candidate = TrackBestCandidate(frame_thresh, candidate_area);
			if (!best_candidate.empty()) {
				DrawPoligon(best_candidate, &frame, Scalar(255, 0, 0));
			}
		}

		//calculate window search (green) around the best candidate
		// in case there is no candidate, search again in the whole frame
			if (mark_status == FOUND || mark_status == TRACKED) {
				search_window = CreateSearchWindow(best_candidate, frame_copy, search_window);
				rectangle(frame_copy, search_window, Scalar(0, 255, 0));
			}
			else {
				search_window = Rect(0, 0, frame_copy.cols, frame_copy.rows);
			}

		//display results and count time
		imshow("frame", frame);
		imshow("candidates", frame_copy);
		e2 = getTickCount();
		time = (e2 - e1) / getTickFrequency();
		cout << "Time:  " << time << endl;
		cout << "Mark Status: " << mark_status << endl;

		//exit with ESC key
		if (waitKey(0) == 27)
		{
			cout << "main: ESC key pressed by user" << endl;
			break;
		}
	}

	return 0;
}

//This function returns the 4 sided poligons found
// in the frame
vector<vector<Point>> FindSquares(Mat * thresh, Mat * frame, Mat * frame_edges)
{
	vector<vector<Point>> cnts;
	vector<Point> poligon;
	vector<vector<Point>> candidates;
	Mat frame_thresh_copy = thresh->clone();

	//preprocess image
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));
	erode(frame_thresh_copy, frame_thresh_copy, element);
	dilate(frame_thresh_copy, frame_thresh_copy, element);
	GaussianBlur(frame_thresh_copy, frame_thresh_copy, Size(3, 3), 4);

	//find contours on the frame
	Mat canny;
	cout << " FindSquares: Searching contours... ";
	Canny(frame_thresh_copy.clone(), canny, 500, 250 * 4);
	imshow("canny", canny);
	findContours(canny.clone(), cnts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
	cout << cnts.size() << "  contours detected" << endl;

	//we will use the canny output later
	canny.copyTo(*frame_edges);

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
	if (candidates.empty()) { 
		cout << "FindSquares: No candidates found" << endl;
		mark_status = NOT_FOUND; 
	}
	else					cout << "FindSquares: " << candidates.size() << "  candidates found" << endl;
	return candidates;
}

// This function returns the index of the candidate that
//scores more points.
// If none of the candidates have a minimun score it returns a -1.
int ScoreSquares(vector<vector<Point>> candidates, Mat * frame, double * mean, int * area, Mat * frame_edges) {

	//measure time
	e3 = getTickCount();
	
	vector<float> slope_coefficents;
	slope_coefficents = ExtractSlopeCoefficent(candidates);

	vector<double> region_means;
	region_means = ExtractRegionMean(candidates, frame);

	DetectCross(candidates, frame_edges);

	//take the candidate with the highest score and check if it reaches a minimum score
	float highest_score = 0, temp_score;
	int idx = -1;
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
	if (highest_score < 1.1) {
		idx = -1;
		mark_status = NOT_FOUND;
		cout << "ScoreSquares: None of the candidates passed the test" << endl;
	}
	else {
		cout << "ScoreSquares: A suitable candidate was found" << endl;
		//we take the mean sample and the area
		*mean = region_means[idx];
		*area = contourArea(candidates[idx]);
		mark_status = FOUND;
	}

	//measure time 2
	e4 = getTickCount();
	time = (e4 - e3) / getTickFrequency();
	cout << "ScoreSquares: The frame took " << time << " seconds to proccess" << endl;

	return idx;
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
		region_means.push_back(temp_value[2] / 255);
	}

	cout << "ExtractRegionMean: All candidates region means were evaluated" << endl;
	return region_means;
}

//This function returns for each square a coefficent <=1 describing how parallel
//are the lines of the squares found
vector<float> ExtractSlopeCoefficent(vector<vector<Point>> candidates) {

	vector<float> slope_coefficents;

	for (int i = 0; i < candidates.size(); i++) {
		//find slope between opposite sides
		// slope = (y2 - y1) / (x2 - x1)
		float k1a, k1b, k2a, k2b;
		float maxk1, mink1, maxk2, mink2;
		float c1, c2;
		k1a = abs((candidates[i][1].y - candidates[i][0].y) / (float)(candidates[i][1].x - candidates[i][0].x));
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
void DrawPoligon(vector<Point> poligon, Mat *frame, Scalar color, Point offset) {

	for (int i = 0; i < poligon.size(); i++) {
		putText(*frame, "P", Point(poligon[i].x + offset.x, poligon[i].y + offset.y), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0));
		if (i<poligon.size() - 1)
			line(*frame, poligon[i] + offset, poligon[i + 1] + offset, color);
		else
			line(*frame, poligon[poligon.size() - 1] + offset, poligon[0] + offset, color);
	}

}

//This function draws the best candidate in red
// and the rest of them in blue
void DrawCandidates(vector<vector<Point>> candidates, Mat * frame, int idx, Rect search_window) {
	
	Point offset = Point(search_window.x, search_window.y);

	if (!candidates.empty())
	{
		for (int i = 0; i < candidates.size(); i++)
		{
			if (i != idx) DrawPoligon(candidates[i], frame, Scalar(255, 0, 0), offset);
		}

		DrawPoligon(candidates[idx], frame, Scalar(0, 0, 255), offset);
	}

}

//This function returns a Mat with the isolated region enclosed by poligon
//The third argument is an optional pointer that returns the direction of the mask.
Mat ExtractRegion(vector<Point> poligon, Mat * frame, Mat * pmask) {

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

	//If pmask!=NULL copy mask to pmask
	if (pmask != NULL) {
		*pmask = mask(roi);
	}

	return contour_region;
}

//This function takes the poligon formed by the best candidates and returns
// a winder window enclosing the poligon.
Rect CreateSearchWindow(vector<Point> best_candidate, Mat frame, Rect previous_search_window) {

	Rect brect;			/*rect bounded to candidate*/
	Rect search_window; /*the output window*/
	brect = boundingRect(best_candidate);

	//configure search window dimentions
	search_window.x = (brect.x - 0.25*brect.width) + previous_search_window.x;
	search_window.y = (brect.y - 0.25*brect.height) + previous_search_window.y;
	search_window.width  = brect.width * 1.5;
	search_window.height = brect.height * 1.5;

	//check if search window exceeds the limits of the frame
	if (search_window.x < 0) search_window.x = 0;
	if (search_window.y < 0) search_window.y = 0;
	if ((search_window.x + search_window.width) > frame.cols) search_window.width = frame.cols - search_window.x;
	if ((search_window.y + search_window.height) > frame.rows) search_window.height = frame.rows - search_window.y;

	cout << "CreateSearchWindow: Search window set" << endl;
	mark_status = TRACKED;

	return search_window;
}


vector<Point> TrackBestCandidate(Mat thresh, int original_area) {

	vector<vector<Point>> cnts;
	vector<Point> biggest_contour;
	vector<Point> empty;

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));
	morphologyEx(thresh, thresh, MORPH_CLOSE, element);

	findContours(thresh, cnts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	drawContours(thresh, cnts, -1, Scalar(255), CV_FILLED);

	//find the contour with the biggest area
	int MinArea = 0;
	int idx = -1;
	for (int i = 0; i < cnts.size(); i++) {
		Moments M;
		M = moments(cnts[i], true);

			if (M.m00 > MinArea) {
				MinArea = M.m00;
				idx = i;
			}
		}

	//check if it still within the 
	if (MinArea < 0.5*original_area || idx == -1) {
		mark_status = NOT_FOUND;
		cout << "TrackBestCandidate: Mark lost. Algorithm will be reset..." << endl;
		return empty;
	}
	else {
		cout << "TrackBestCandidate: Mark succesfully tracked." << endl;
		return cnts[idx];
	}
	
}

//Apply hough transform to each candidate and check if there are lines close to the centroid
void DetectCross(vector<vector<Point>> candidates, Mat * frame_edges) {

	vector<int> results;

	for (int i = 0; i < candidates.size(); i++) {

		Mat mask;
		Mat frame_edges_cropped = ExtractRegion(candidates[i], frame_edges, &mask);
		imshow("framecrop", frame_edges_cropped);

		//find center of contour
		Moments M; float cX, cY;
		Point center;
		M = moments(mask);
		cX = M.m10 / M.m00;
		cY = M.m01 / M.m00;
		center = Point(cX, cY);

		vector<Vec2f> lines;
		HoughLines(frame_edges_cropped, lines, 0.8, (CV_PI / 180), 40);

		//Check if there are lines close to the centroid
		bool positive_slope = false;
		bool negative_slope = false;
		for (int u = 0; u < lines.size(); u++)
		{
			float rho = lines[u][0], theta = lines[u][1];
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			Point pt1, pt2;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));

			double hypotenuse = _hypot(abs(pt1.x - center.x), abs(pt1.y - center.y));

		}
	}

}



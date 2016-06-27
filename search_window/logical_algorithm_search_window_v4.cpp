#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

//for testing
#define MIN_SCORE 1.5

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

//Proccessing functions
Mat PreproccessFrame(Mat frame, double candidate_mean);

//Additional functions
vector<vector<Point>> FindSquares(Mat * thresh, Mat * frame);
struct tracked_mark FindMark(Mat frame_thresh, Mat frame);
vector<struct candidate_region> ExtractRegions(vector<vector<Point>> candidates, Mat * canny, Mat * original_frame);
vector<double> ExtractRegionMean(vector<struct candidate_region> candidate_regions);
vector<float>	ExtractSlopeCoefficent(vector<vector<Point>> candidates);
vector<int> DetectCross(vector<struct candidate_region> regions, vector<vector<Point>> candidates, Mat * frame_edges);
Rect			CreateSearchWindow(vector<Point> best_candidate, Mat frame, Rect previous_search_window = Rect(), Point * offset = NULL);
void			DrawPoligon(vector<Point> poligon, Mat *frame, Scalar color, Point offset = Point(0,0));
void			DrawCandidates(vector<vector<Point>> candidates, Mat * frame, int idx, Rect search_window = Rect());
double DistanceL2P(Point line_begin, Point line_end, Point point);
vector<Point>	TrackBestCandidate(Mat thresh, int original_area);

//Global variables
enum Status { FOUND, NOT_FOUND, TRACKED };
Status mark_status = NOT_FOUND;

struct candidate_region {
	Mat region;
	Mat mask;
};

struct tracked_mark {
	vector<Point> contour;
	int area;
	double mean;
	bool isvalid = false;
};

//time variables
int e1, e2, e3, e4;
float time;

//situational
bool resized = false;

int main(int argc, char** argv)
{

	//load video
	VideoCapture capture("C:/images/video9.mp4");
	int frame_num = 0;
	if (!capture.isOpened())
		throw "Error: cannot read video";

	for (; ; )
	{
		Mat frame, frame_copy;
		static Rect search_window;
		static struct tracked_mark best_candidate;

		//measure start time
		e1 = getTickCount();

		//load frame
		capture >> frame;
		if (frame.empty()) break;
		//resize(frame, frame, Size(),0.5,0.5,INTER_LINEAR);
		frame_copy = frame.clone(); /*for displaying results without altering original frame*/

		//crop search window if applicable
		if (mark_status == TRACKED) {
			frame = frame(search_window);
		}

		Mat frame_thresh;
		frame_thresh = PreproccessFrame(frame, best_candidate.mean);
		imshow("threshold", frame_thresh);

		//search for candidates if mark is not found
		//evaluate them, and select the best one
		if (mark_status == NOT_FOUND) {
			best_candidate = FindMark(frame_thresh, frame_copy);
			DrawPoligon(best_candidate.contour, &frame_copy, Scalar(0, 0, 255));
		}
		else {
			best_candidate.contour = TrackBestCandidate(frame_thresh, best_candidate.area);
		}

		//calculate window search (green) around the best candidate
		// in case there is no candidate, search again in the whole frame
			if (mark_status == FOUND || mark_status == TRACKED) {
				Point mark_shift;
				search_window = CreateSearchWindow(best_candidate.contour, frame_copy, search_window, &mark_shift);
				DrawPoligon(best_candidate.contour, &frame_copy, Scalar(255, 0, 0),Point((search_window.x - mark_shift.x),(search_window.y - mark_shift.y)));
				rectangle(frame_copy, search_window, Scalar(0, 255, 0));
				//cout << "main: search window size: " << search_window.height << " " << search_window.width << endl;
			}
			else {
				search_window = Rect(0, 0, frame_copy.cols, frame_copy.rows);
			}

		//display results and count time
		imshow("frame", frame_copy);
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

		//count frame (for test purposes)
		cout << " == MAIN: FRAME NUMBER " << frame_num << endl;
		frame_num++;
	}

	return 0;
}

//This function returns the 4 sided poligons found
// in the frame
vector<vector<Point>> FindSquares(Mat * thresh, Mat * frame_edges)
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
	//Canny(frame_thresh_copy.clone(), canny, 500, 250 * 4);
	Canny(frame_thresh_copy.clone(), canny, 300, 200*5);
	imshow("canny", canny);
	findContours(canny.clone(), cnts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
	cout << cnts.size() << "  contours detected" << endl;

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

	//return the edges image
	canny.copyTo(*frame_edges);

	//return the possible candidates
	if (candidates.empty()) { 
		cout << "FindSquares: No candidates found" << endl;
		mark_status = NOT_FOUND; 
	}
	else					cout << "FindSquares: " << candidates.size() << "  candidates found" << endl;
	return candidates;
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
		//take the mean value of the coefficents
		maxk1 = max(k1a, k1b); mink1 = min(k1a, k1b);
		maxk2 = max(k2a, k2b); mink2 = min(k2a, k2b);
		c1 = mink1 / maxk1; c2 = mink2 / maxk2;
		slope_coefficents.push_back((c1+c2)/2);

		//cout << "ExtractSlopeCoefficent: Candidate " << i << " scored " << (c1 + c2) / 2 << endl;
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
	

	if (!candidates.empty())
	{
		Point offset = Point(search_window.x, search_window.y);

		for (int i = 0; i < candidates.size(); i++)
		{
			if (i != idx) DrawPoligon(candidates[i], frame, Scalar(255, 0, 0), offset);
		}

		DrawPoligon(candidates[idx], frame, Scalar(0, 0, 255), offset);
	}

}



//This function takes the poligon formed by the best candidates and returns
// a winder window enclosing the poligon.
Rect CreateSearchWindow(vector<Point> best_candidate, Mat frame, Rect previous_search_window, Point * offset) {

	Rect brect;				   /*rect bounded to candidate*/
	static Rect previous_brect;
	Rect search_window;		   /*the output window*/
	brect = boundingRect(best_candidate);

	//set the upper-left corner of the search window
	Point mark_shift;
	mark_shift = Point(0, 0);
	if (mark_status != TRACKED) {
		search_window.x = (brect.x - 0.25*brect.width);
		search_window.y = (brect.y - 0.25*brect.height);
		previous_brect.x = brect.x - search_window.x;
		previous_brect.y = brect.y - search_window.y;
		mark_status = TRACKED;
	}
	else {
		mark_shift.x = brect.x - previous_brect.x;
		mark_shift.y = brect.y - previous_brect.y;
		if ((mark_shift.x < 0) && (previous_search_window.x == 0)) {
			mark_shift.x = 0;
		}
		if ((mark_shift.y < 0) && (previous_search_window.y == 0)) {
			mark_shift.y = 0;
		}

		search_window.x = previous_search_window.x + mark_shift.x;
		search_window.y = previous_search_window.y + mark_shift.y;

	}

	//set the search window dimentions
	search_window.width = brect.width * 1.5;
	search_window.height = brect.height * 1.5;

	//check if search window exceeds the limits of the frame
	if (search_window.x < 0) search_window.x = 0;
	if (search_window.y < 0) search_window.y = 0;
	if ((search_window.x + search_window.width) > frame.cols) search_window.width = frame.cols - search_window.x;
	if ((search_window.y + search_window.height) > frame.rows) search_window.height = frame.rows - search_window.y;

	//return offset for better results
	*offset = mark_shift;

	cout << "CreateSearchWindow: Search window set" << endl;

	return search_window;
}


vector<Point> TrackBestCandidate(Mat thresh, int original_area) {

	vector<vector<Point>> cnts;
	vector<Point> biggest_contour;
	vector<Point> empty;

	findContours(thresh, cnts, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	drawContours(thresh, cnts, -1, Scalar(255), CV_FILLED);

	//find the contour with the biggest area
	int idx = -1;
	for (int i = 0; i < cnts.size(); i++) {
		Moments M;
		M = moments(cnts[i], true);
			if (M.m00 <= 1.5*original_area && M.m00 >= 0.5*original_area) {
				idx = i;
			}
		}

	//check if it still within the range of the original area
	if (idx == -1) {
		mark_status = NOT_FOUND;
		cout << "TrackBestCandidate: Mark lost. Algorithm will be reset..." << endl;
		return empty;
	}
	else {
		cout << "TrackBestCandidate: Mark succesfully tracked." << endl;
		return cnts[idx];
	}
	
}

//cross detection
//candidates can score up to 2 points if they have a line with negative slope and a line with positive
//slope close to the centroid 
vector<int> DetectCross(vector<struct candidate_region> regions, vector<vector<Point>> candidates, Mat * frame_edges) {

	vector<Vec2f> lines;
	Rect brect;
	vector<int> scores;

	//these flags indicate if a line with positive or negative slope
	//passes near the centroid of the region
	bool neg_slope;
	bool pos_slope;

	for (int i = 0; i < candidates.size(); i++) {
		
		//crop the fragment of the canny image bounded to the candidate
		brect = boundingRect(candidates[i]);
		Mat copy, crop;
		frame_edges->copyTo(copy);
		copy = copy(brect);
		copy.copyTo(crop, regions[i].mask);

		//calculate centroid of the region
		Moments M = moments(regions[i].mask);
		Point centroid(M.m10 / M.m00, M.m01 / M.m00);
		//cout << "DetectCross: Centroid: ( " << centroid.x << " , " << centroid.y << " )" << endl;

		//take the fraction on the center
		Rect zoom;
		zoom.x = centroid.x - 0.2 * crop.cols;
		zoom.y = centroid.y - 0.2 * crop.rows;
		zoom.width = 0.4*crop.cols;
		zoom.height = 0.4*crop.rows;
		Mat test = crop(zoom);
		crop = crop(zoom);
		Point gcenter(crop.cols/2,crop.rows/2);
		int zeros = countNonZero(crop);
		//imshow("TEST", test);
	
		//find lines
		HoughLines(crop, lines, 1, (0.5*CV_PI / 180), 0.05*zeros);
		cvtColor(crop, crop, COLOR_GRAY2BGR);
		circle(crop, centroid, 5, Scalar(255, 0, 0), 2);
		circle(crop, gcenter, 5, Scalar(255, 0, 0), 2);
		cout << "DetectCross: " << lines.size() << "lines generated" << endl;

		//predict the slopes of the cross
		float slope, slope_neg = 0, slope_pos = 0;
		slope = (candidates[i][2].x - candidates[i][0].x) / (float)(candidates[i][2].y - candidates[i][0].y);
		if (slope < 0) {
			slope_neg = slope;
		}
		else {
			slope_pos = slope;
		}
		slope = (candidates[i][3].x - candidates[i][1].x) / (float)(candidates[i][3].y - candidates[i][1].y);
		if (slope < 0) {
			slope_neg = slope;
		}
		else {
			slope_pos = slope;
		}

		//these two flags indicate if the line detected is close to the centroid AND
		neg_slope = false;
		pos_slope = false;

		for (size_t u = 0; u < lines.size(); u++)
		{
			float rho = lines[u][0], theta = lines[u][1];
			//cout << "Line " << u << " : theta = " << theta << ", rho = " << rho << endl;
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);

			if ((!pos_slope && a > 0) || (!neg_slope && a < 0)) {
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000 * (-b)); pt1.y = cvRound(y0 + 1000 * (a));
				pt2.x = cvRound(x0 - 1000 * (-b)); pt2.y = cvRound(y0 - 1000 * (a));
				
				if (a < 0) line(crop, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
				else line(crop, pt1, pt2, Scalar(0, 255, 0), 1, CV_AA);

				//Check if the line is close to the centroid
				double dist = DistanceL2P(pt1, pt2, gcenter);
				//cout << "Distance: " << dist << endl;
				if (dist < 0.05*crop.cols) {
					if (a > 0) {
						pos_slope = true;
					}
					if (a < 0) {
						neg_slope = true;
					}
					line(crop, pt1, pt2, Scalar(0, 255, 255), 4, CV_AA);
				}
				line(*frame_edges, candidates[i][1], candidates[i][3], Scalar(200, 200, 200), 4, CV_AA);
				line(*frame_edges, candidates[i][0], candidates[i][2], Scalar(200, 200, 200), 4, CV_AA);
			}
		}
		imshow("cross detection lines", crop);
		//imshow("cross localizations", *frame_edges);

		//score +1 point for each line close to the centroid detected
		int temp_score = 0;
		if (pos_slope) temp_score++;
		if (neg_slope) temp_score++;
		scores.push_back(temp_score);
	}

	return scores;
}

//This function calculates the distance between a line (given 2 points) and an external point
double DistanceL2P(Point line_begin, Point line_end, Point point) {

	vector<Point> triangle;
	triangle.push_back(line_begin);
	triangle.push_back(line_end);
	triangle.push_back(point);
	triangle.push_back(line_begin);

	float area = contourArea(triangle);
	double base = norm(line_end - line_begin);
	double distance = 2 * area / base; /*From the triangle area formula*/

	triangle.clear();
	return distance;
}

//Preproccessing function to make contours clear
Mat PreproccessFrame(Mat frame, double candidate_mean) {
	
	Mat frame_gray, frame_thresh;

	GaussianBlur(frame, frame, Size(3, 3), 4);
	addWeighted(frame, 1.5, frame, -0.5, 0, frame);
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	//create an adaptative threshold if mark is to be found
	//if mark is found, put a standar threshold with maxmin-values around the mean of the previous best candidate
	if (mark_status == NOT_FOUND) adaptiveThreshold(frame_gray, frame_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 23, 3);
	else {
		threshold(frame_gray, frame_thresh, (1 - 0.1*candidate_mean) * 255, (1 + 0.1*candidate_mean) * 255, THRESH_BINARY);
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
		morphologyEx(frame_thresh, frame_thresh, MORPH_CLOSE, element);
		GaussianBlur(frame_thresh, frame_thresh, Size(3, 3), 4);
	}

	return frame_thresh;
}

struct tracked_mark FindMark(Mat frame_thresh, Mat frame) {

	struct tracked_mark best_candidate;

	//Search for 4-sided poligons (candidates)
	vector<vector<Point>> candidates;
	Mat frame_edges;
	e3 = getTickCount();
	candidates = FindSquares(&frame_thresh, &frame_edges);
	for (int i = 0; i < candidates.size(); i++) {
		DrawPoligon(candidates[i], &frame, Scalar(255, 0, 0));
	}
	e4 = getTickCount();
	time = (e4 - e3) / getTickFrequency();
	cout << "FindMark: Finding squares took " << time << " seconds" << endl;

	//Extract slope coefficents
	vector<float> slope_coefficents;
	slope_coefficents = ExtractSlopeCoefficent(candidates);
	//eliminate candidates with low slope coefficents
	for (int i = 0; i < slope_coefficents.size(); i++) {
		if (slope_coefficents[i] < 0.3) {
			slope_coefficents.erase(slope_coefficents.begin()+i);
			candidates.erase(candidates.begin() + i);
			i--;
		}
	}

	//Extract regions and masks
	vector<struct candidate_region> regions;
	e3 = getTickCount();
	regions = ExtractRegions(candidates, &frame_edges, &frame);
	e4 = getTickCount();
	time = (e4 - e3) / getTickFrequency();
	cout << "FindMark: Extracting regions took " << time << " seconds" << endl;

	//Score means
	vector<double> region_means;
	e3 = getTickCount();
	region_means = ExtractRegionMean(regions);
	//eliminate candidates with low slope coefficents
	for (int i = 0; i < region_means.size(); i++) {
		if (region_means[i] < 0.5) {
			region_means.erase(region_means.begin() + i);
			regions.erase(regions.begin() + i);
			candidates.erase(candidates.begin() + i);
			i--;
		}
	}
	e4 = getTickCount();
	time = (e4 - e3) / getTickFrequency();
	cout << "FindMark: Calculating means took " << time << " seconds" << endl;
	
	//Find crosses
	e3 = getTickCount();
	vector<int> crosses;
	crosses = DetectCross(regions, candidates, &frame_edges);
	e4 = getTickCount();
	time = (e4 - e3) / getTickFrequency();
	cout << "FindMark: Detecting crosses took " << time << " seconds" << endl;

	//Evaluate each candidate
	//take the candidate with the highest score and check if it reaches a minimum score
	float highest_score = 0, temp_score;
	int idx = -1;
	for (int i = 0; i < candidates.size(); i++) {
		//draw on frame individual scores
		char text[50];
		sprintf_s(text, "Slope: %f", slope_coefficents[i]);
		putText(frame, text, candidates[i][1], CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		sprintf_s(text, "Mean: %f", region_means[i]);
		putText(frame, text, Point(candidates[i][1].x, candidates[i][1].y-20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

		temp_score = slope_coefficents[i] + region_means[i] + crosses[i];
		if (temp_score > highest_score) {
			highest_score = temp_score;
			idx = i;
		}
		cout << "FindMark: Score for square number " << i << ": " << temp_score << endl;
	}

	//if none of the candidates passes the minimun score
	//then we suppose the mark is not found
	if (highest_score < 1.5) {
		cout << "FindMark: None of the candidates passed the test" << endl;
		best_candidate.isvalid = false;
		mark_status = NOT_FOUND;
	}
	else {
		//If a good candidate is found, return its data
		cout << "FindMark: A suitable candidate was found" << endl;
		best_candidate.contour = candidates[idx];
		best_candidate.mean  = region_means[idx];
		best_candidate.area  = contourArea(candidates[idx]);
		best_candidate.isvalid = true;
		mark_status = FOUND;
	}

	return best_candidate;
}

//This function returns a set of images with the isolated region enclosed by each of the candidates
//and their respective masks
vector<struct candidate_region> ExtractRegions(vector<vector<Point>> candidates, Mat * canny, Mat * original_frame) {

	vector<struct candidate_region> regions;
	struct candidate_region temp_region;

	for (int i = 0; i < candidates.size(); i++)
	{
		vector < vector<Point>> candidate_v;
		candidate_v.push_back(candidates[i]);

		//Create and store mask to isolate the candidate
		Rect roi = boundingRect(candidates[i]);
		Mat mask = Mat::zeros(canny->size(), CV_8UC1);
		drawContours(mask, candidate_v, -1, Scalar(255), CV_FILLED);

		//Crop the isolated region
		Mat contour_region;
		original_frame->copyTo(contour_region,mask);
		contour_region = contour_region(roi);
		
		//Add it to the vector of regions
		temp_region.region = contour_region;
		temp_region.mask   = mask(roi);
		regions.push_back(temp_region);
		
		candidate_v.clear();
	}
	
	cout << "ExtractRegions: All regions extracted" << endl;
	return regions;
}

//This function takes the mean value of each of the squares found.
//A high mean scores more points than a low mean.
vector<double> ExtractRegionMean(vector<struct candidate_region> candidate_regions) {
	vector<double> region_means;

	for (int i = 0; i < candidate_regions.size(); i++) {

		//Store the normalized value of mean in the output vector
		Scalar temp_value = mean(candidate_regions[i].region, candidate_regions[i].mask);
		region_means.push_back(temp_value[2] / 255);
	}

	cout << "ExtractRegionMean: All candidates region means were evaluated" << endl;
	return region_means;
}


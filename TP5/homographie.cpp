#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
using namespace std;
using namespace cv;


const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main()
{
	Mat I1 = imread("../IMG_0045.JPG", CV_LOAD_IMAGE_GRAYSCALE);
	Mat I2 = imread("../IMG_0046.JPG", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat I2 = imread("../IMG_0046r.JPG", CV_LOAD_IMAGE_GRAYSCALE);
	/*
	namedWindow("I1", 1);
	namedWindow("I2", 1);
	
	imshow("I1", I1);
	imshow("I2", I2);
	*/

	Ptr<AKAZE> D = AKAZE::create();
	Mat desc1, desc2;
	vector<KeyPoint> m1, m2;

	D->detectAndCompute(I1, noArray(), m1, desc1);
	D->detectAndCompute(I2, noArray(), m2, desc2);
	
	Mat J;
	drawKeypoints(I1, m1, J);
	/*
	imshow("J", J);
	*/
	
	Mat homography, J2;

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);
	drawMatches(I1, m1, I2, m2, nn_matches, J2);
	imshow("All matches", J2);
	
	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
	vector<Point2f> obj1, obj2, gobj1, gobj2;
	vector<DMatch> good_matches;
	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if (dist1 < nn_match_ratio * dist2) {
			matched1.push_back(m1[first.queryIdx]);
			matched2.push_back(m2[first.trainIdx]);
		}
	}

	for (size_t i = 0; i < matched1.size(); i ++) {
		obj1.push_back(matched1[i].pt);
		obj2.push_back(matched2[i].pt);
	}

	homography = findHomography(obj1, obj2, CV_RANSAC);
	cout << homography;

	for (unsigned i = 0; i < matched1.size(); i++) {
		Mat col = Mat::ones(3, 1, CV_64F);
		col.at<double>(0) = matched1[i].pt.x;
		col.at<double>(1) = matched1[i].pt.y;

		col = homography * col;
		col /= col.at<double>(2);
		double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
			pow(col.at<double>(1) - matched2[i].pt.y, 2));

		if (dist < inlier_threshold) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);

			gobj1.push_back(matched1[i].pt);
			gobj2.push_back(matched2[i].pt);
			good_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}

	homography = findHomography(gobj1, gobj2, CV_RANSAC);
	cout << homography;
	Mat J3;
	drawMatches(I1, matched1, I2, matched2, good_matches, J3);


	Mat res;
	drawMatches(I1, inliers1, I2, inliers2, good_matches, res);
	imshow("Good Matches", res);
	
	imwrite("res.png", res);

	double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
	cout << "A-KAZE Matching Results" << endl;
	cout << "*******************************" << endl;
	cout << "# Keypoints 1:                        \t" << m1.size() << endl;
	cout << "# Keypoints 2:                        \t" << m2.size() << endl;
	cout << "# Matches:                            \t" << matched1.size() << endl;
	cout << "# Inliers:                            \t" << inliers1.size() << endl;
	cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
	cout << endl;

	Mat IP1 = imread("../IMG_0045.JPG"), IP2 = imread("../IMG_0046.JPG");
	/*
	Mat K(2 * I2.cols, I2.rows, CV_8U);
	Mat L;
	warpPerspective(I2, L, homography, I2.size());
	hconcat(I1, L, K);
	imshow("Reconstruction", K);
	*/

	Mat KP(2 * IP2.cols, IP2.rows, CV_8UC3);
	Mat LP;
	warpPerspective(IP2, LP, homography, IP2.size());
	hconcat(IP1, LP, KP);
	imshow("Reconstruction", KP);

	// ...
	
	//Mat J;
	//drawKeypoints(...
	

	//BFMatcher M ...

	// drawMatches ...
	
	// Mat H = findHomography(...
	
	
	// Mat K(2 * I1.cols, I1.rows, CV_8U);
	// warpPerspective( ...
	
	waitKey(0);
	return 0;
}

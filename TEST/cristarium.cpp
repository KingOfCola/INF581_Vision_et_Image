#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
using namespace std;
using namespace cv;


Mat black(int n, int p) {
	Mat I(n, p, CV_64FC3);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			for (int k = 0; k < 3; k++) {
				I.at<Vec3d>(i, j)[k] = 0.;
			}
		}
	}
	return I;
}

template
void drawLine(Mat I, Point a, Point b, )

void thunder(Mat I, int nb_points = 50, int size = 10) {
	int n = I.rows, p = I.cols;
	int j = n / 2;
	for (int i = 0; i < p; i++) {
		
	}
}

int main() {
	Mat I;
	I = black(50, 100);
	imshow("I", I);
	waitKey(0); 
	return 0;
}

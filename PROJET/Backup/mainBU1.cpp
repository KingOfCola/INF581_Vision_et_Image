#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;
using namespace cv;

inline bool isInInterval(const Vec3b a, const Vec3b b, const int p) {
	return ((int)a.val[0] > (int)b.val[0] - p) && ((int)a.val[0] < (int)b.val[0] + p) &&
		((int)a.val[1] > (int)b.val[1] - p) && ((int)a.val[1] < (int)b.val[1] + p) &&
		((int)a.val[2] > (int)b.val[2] - p) && ((int)a.val[2] < (int)b.val[2] + p);
}

template <typename T>
Mat integral(Mat a) {
	int n = a.rows, p = a.cols;
	Mat I(a);
	I.at<T>(0, 0) = a.at<T>(0, 0);
	for (int i = 1; i < n; i++) {
		I.at<T>(i, 0) = I.at<T>(i - 1, 0) + a.at<T>(i, 0);
	}
	for (int j = 1; j < p; j++) {
		I.at<T>(0, j) = I.at<T>(0, j - 1) + a.at<T>(0, j);
	}
	for (int i = 1; i < n; i++) {
		for (int j = 1; j < p; j++) {
			I.at<T>(i, j) = I.at<T>(i - 1, j) - I.at<T>(i - 1, j - 1) + I.at<T>(i, j- 1) + a.at<T>(i, j);
		}
	}
	return I;
}

template <typename T>
int sumValues(Mat I) {
	int s = 0;
	for (int m = 0; m < I.rows; m++) {
		for (int n = 0; n < I.cols; n++) {
			s += (int)I.at<T>(m, n);
		}
	}
	return s;
}


template <typename T>
Mat reconstruct_image(vector<Mat> goodPixels, vector<Mat> imgs) {
	int m = imgs[0].rows, n = imgs[0].cols;
	vector<int> relevancy, indices;
	Mat I(m, n, CV_8UC3), Choose(m, n, CV_8UC3);
	vector<Vec3b> Colors;
	Colors.push_back(Vec3b(0, 0, 0));
	Colors.push_back(Vec3b(255, 0, 0));
	Colors.push_back(Vec3b(0, 255, 0));
	Colors.push_back(Vec3b(0, 0, 255));
	Colors.push_back(Vec3b(255, 255, 0));
	Colors.push_back(Vec3b(0, 255, 255));
	Colors.push_back(Vec3b(255, 0, 255));
	Colors.push_back(Vec3b(255, 255, 255));

	int j;
	for (int i = 0; i < goodPixels.size(); i++) {
		int s = (int)sumValues<T>(goodPixels[i]);
		relevancy.push_back(s);
		indices.push_back(i);
		for (j = indices.size() - 2; j >= 0; j--) {
			if (s > relevancy[j]) { 
				relevancy[j + 1] = relevancy[j]; 
				relevancy[j] = s;
				indices[j + 1] = indices[j];
				indices[j] = i;
			}
			else {
				break;
			}
		}
	}

	cout << I.rows << "          ---    " << I.cols << endl;
	for (int m = 0; m < I.rows; m++) {
		for (int n = 0; n < I.cols; n++) {
			I.at<Vec3b>(m, n)[0] = 0;
			I.at<Vec3b>(m, n)[1] = 0;
			I.at<Vec3b>(m, n)[2] = 0;
		}
	}
	imshow("TEst", I);


	// Choix de l'image par pixel
	for (int i = 0; i < goodPixels.size(); i++) {
		cout << relevancy[i] << "    " << indices[i] << endl;
		cout << I.rows << "  " << I.cols << endl;
		for (int m = 0; m < I.rows; m++) {
			for (int n = 0; n < I.cols; n++) {
				if ((I.at<Vec3b>(m, n)[0] == 0) && goodPixels[indices[i]].at<T>(m, n) != 0) {
					I.at<Vec3b>(m, n) = imgs[indices[i]].at<Vec3b>(m, n);
					(Choose.at<Vec3b>(m, n)) = Colors[indices[i]];
				}
			}
		}
	}
	imshow("Choose", Choose);
	imwrite("../resultatChoose.jpg", Choose);
	return I;
}



int main()
{// importation des images
	String nom_image = "../photo";
	String extension = ".jpg";
	int nb_images = 8;
	vector<Mat> I;
	for (int i = 1; i <= nb_images; i++) {
		I.push_back(imread(nom_image + to_string(i) + extension));
	}

	// initialisation de l'image resultat
	Mat res = imread(nom_image + to_string(1) + extension);
	int seuil = 15;

	// compter les occurences par pixel et choisir max
	Vec3b rgb;
	cout << I[0].rows << "  -  " << I[0].cols << endl;
	for (int m = 1; m < I[0].rows; m++) {
		for (int n = 1; n < I[0].cols; n++) {
			vector<Vec3b> couleurs;
			vector<int> occurences;
			for (int i = 0; i < nb_images; i++) {
				rgb = I[i].at<Vec3b>(m, n);
				bool trouve = false;
				for (int j = 0; j < couleurs.size(); j++) {
					if (norm(couleurs[j], rgb, CV_L2) < seuil) {
						occurences[j] ++;
						trouve = true;
						break;
					}
				}
				if (!trouve) {
					couleurs.push_back(rgb);
					occurences.push_back(1);
				}
			}
			int argmax = distance(occurences.begin(), max_element(occurences.begin(), occurences.end()));
			res.at<Vec3b>(m, n) = couleurs[argmax];
		}
	}
	vector<Mat> goodPixels;
	for (int i = 0; i < nb_images; i++) {
		Mat temp = Mat(I[0].size(), CV_8U);
		for (int m = 0; m < I[0].rows; m++) {
			for (int n = 0; n < I[0].cols; n++) {
				Vec3b v1 = I[i].at<Vec3b>(m, n);
				Vec3b v2 = res.at<Vec3b>(m, n);
				float dist = norm(v1, v2, CV_L2);
				if (dist < seuil) {
					temp.at<unsigned char>(m, n) = 255;
				}
				else {
					temp.at<unsigned char>(m, n) = 0;
				}
			}
		}
		goodPixels.push_back(temp);
	}

	Mat resul = reconstruct_image<unsigned char>(goodPixels, I);
	imshow("AIAIAIA", resul);

	imwrite("../resultat.jpg", res);
	imwrite("../resultatMaxIm.jpg", resul);
	/*
	for (int i = 0; i < nb_images; i++) {
		namedWindow("Resultat" + to_string(i), i + 1);
		imshow("Resultat" + to_string(i), goodPixels[i]);
	}
	*/
	waitKey(0);
	return 0;
}

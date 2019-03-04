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




Mat add_rectangles(Mat I, int nb_rectangles, int dx = 0, int dy = 0) {
	// Initialization : dx, dy, max dims of rectangles.
	int n1, n2, m1, m2, c;
	Vec3b color;
	Mat J = I.clone();
	if (dx < 1) {dx = I.rows; }
	if (dy < 1) {dy = I.cols; }

	for (int i = 0; i < nb_rectangles; i++) {
		/*
		Random locations : (m1, n1) 
		Random sizes : (m2, n2)
		Random Color
		*/

		m1 = rand() % (I.rows + dx) - dx;
		m2 = m1 + rand() % dx;
		cout << m1 << " " << m2 << " ";
		if (m2 >= I.rows) { m2 = I.rows - 1; }
		if (m2 < 0) { m2 = 0; }
		if (m1 < 0) { m1 = 0; }

		n1 = rand() % (I.cols + dy) - dy;
		n2 = n1 + rand() % dy;
		cout << n1 << " " << n2 << endl;
		if (n2 >= I.cols) { n2 = I.cols - 1; }
		if (n2 < 0) { n2 = 0; }
		if (n1 < 0) { n1 = 0; }

		color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
		
		cout << m1 << " " << m2 << " " << n1 << " " << n2 << " " << color << endl << endl;
		// Filling it

		for (int x = m1; x < m2; x++) {
			for (int y = n1; y < n2; y++) {
				J.at<Vec3b>(x, y) = color;
			}
		}
	}
	return J;
}

Vec3b at(Mat I, int i, int j) {
	// returns I[i, j] if it exists, 0 otherwise
	if (i < 0 || i >= I.rows || j < 0 || j >= I.cols) {
		return Vec3b(0, 0, 0);
	}
	return I.at<Vec3b>(i, j);
}

Mat convolate(Mat I, Mat X) {
	int m = X.rows / 2;
	int n = X.cols / 2;
	Mat J = I.clone();
	int Xsum = 0;
	for (int x = 0; x < X.rows; x++) {
		for (int y = 0; y < X.cols; y++) {
			Xsum += int(X.at<unsigned char>(x, y));
		}
	}

	for (int i = 0; i < I.rows; i++) {
		for (int j = 0; j < I.cols; j++) {
			Vec3i s(0, 0, 0);
			for (int x = 0; x < X.rows; x++) {
				for (int y = 0; y < X.cols; y++) {
					s += Vec3i(at(I, i - m + x, j - n + y)) * int(X.at<unsigned char>(x, y));
				}
			}
			J.at<Vec3b>(i, j) = Vec3b(s / Xsum);
		}
	}
	return J;
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
Mat integral(Mat I) {
	Mat J;
	J = I.clone();
	for (int n = 1; n < I.cols; n++) {
		J.at<T>(0, n) += J.at<T>(0, n - 1);
	}
	for (int m = 1; m < I.rows; m++) {
		J.at<T>(m, 0) += J.at<T>(m - 1, 0);
	}
	for (int m = 1; m < I.rows; m++) {
		for (int n = 1; n < I.cols; n++) {
			J.at<T>(m, n) += J.at<T>(m - 1, n) + J.at<T>(m, n - 1) - J.at<T>(m - 1, n - 1);
		}
	}
	return J;
}

Mat reconstruct_image_mean(vector<Mat> goodPixels, vector<Mat> imgs) {
	int m = imgs[0].rows; 
	int n = imgs[0].cols; 
	int s;
	int nbGoodPixel;
	Vec3i currentPixel;
	Mat I(m, n, CV_8UC3);
	Mat NB(m, n, CV_8U);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			s = 0;
			currentPixel = Vec3i(0, 0, 0);
			for (int k = 0; k < goodPixels.size(); k++) {
				if (goodPixels[k].at<unsigned char>(i, j) != 0) {
					currentPixel += Vec3i(imgs[k].at<Vec3b>(i, j));
					s += 1;
				}
			}
			I.at<Vec3b>(i, j) = Vec3b(currentPixel / s);
			NB.at<unsigned char>(i, j) = s * 30;
		}
	}
	imshow("NB", NB);
	imwrite("nbGoodPixels.png", NB);
	return I;
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
{

	// importation des images
	String nom_image = "../photo";
	String extension = ".jpg";
	int nb_images = 8;
	vector<double> sums;
	vector<Mat> I;
	for (int i = 1; i <= nb_images; i++) {
		I.push_back(imread(nom_image + to_string(i) + extension));
		sums.push_back(norm(sum(I[i - 1])));
	}

	Mat X(15, 15, CV_8U);
	for (int x = 0; x < X.rows; x++) {
		for (int y = 0; y < X.cols; y++) {
			X.at<unsigned char>(x, y) = 1;
		}
	}
	Mat TEST = convolate(I[0], X);
	imshow("test", TEST);
	/*
	imshow("Weird", add_rectangles(I[0], 30, 150, 300));
	waitKey(0);
	return 0;
	*/
	/*
	double moy = sum(sums)[0] / nb_images;
	for (int i = 0; i < nb_images; i++) {
		I[i] = I[i] * moy / sums[i];
		cout << i << " " << moy << " " << sums[i] << endl;
	}
	*/


	// initialisation de l'image resultat
	Mat res = imread(nom_image + to_string(1) + extension);
	int seuil = 15;

	// compter les occurences par pixel et choisir max
	Vec3b rgb;
	cout << I[0].rows << "  -  " << I[0].cols << endl;
	for (int m = 1; m < I[0].rows-1; m++) {
		for (int n = 1; n < I[0].cols-1; n++) {
			vector<Vec3b> couleurs;
			vector<int> occurences;
			for (int i = 0; i < nb_images; i++) {
				rgb = I[i].at<Vec3b>(m,n)/5+ I[i].at<Vec3b>(m+1, n)/5+ I[i].at<Vec3b>(m-1, n)/5+ I[i].at<Vec3b>(m, n+1)/5+ I[i].at<Vec3b>(m, n-1)/5;
				rgb = rgb;
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
			res.at<Vec3b>(m,n) = couleurs[argmax];
		}
	}
	vector<Mat> goodPixels;
	for (int i = 0; i < nb_images; i++) {
		Mat temp = Mat(I[0].size(), CV_8U);
		for (int m = 1; m < I[0].rows-1; m++) {
			for (int n = 1; n < I[0].cols-1; n++) {
				Vec3b v1 = I[i].at<Vec3b>(m, n)/5+I[i].at<Vec3b>(m + 1, n)/5 + I[i].at<Vec3b>(m - 1, n)/5 + I[i].at<Vec3b>(m, n + 1)/5 + I[i].at<Vec3b>(m, n - 1)/5;
				Vec3b v2 = res.at<Vec3b>(m, n);
				float dist = norm(v1,v2, CV_L2);
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
	//Mat resul = reconstruct_image<unsigned char>(goodPixels, I);
	Mat resul = reconstruct_image_mean(goodPixels, I);
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

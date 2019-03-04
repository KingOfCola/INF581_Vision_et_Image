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



/*
Miscellaneous functions to alter the images

*/
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

/* 
Miscellaneous functions to make some small elementary operations on the images.

*/
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


Mat integral(Mat I) {
	// Creates the integral image of the pixel.
	Mat J(I.rows, I.cols, CV_32SC3);
	J.at<Vec3i>(0, 0) = Vec3i(I.at<Vec3b>(0, 0));
	for (int n = 1; n < I.cols; n++) {
		J.at<Vec3i>(0, n) = Vec3i(I.at<Vec3b>(0, n)) + J.at<Vec3i>(0, n - 1);
	}
	for (int m = 1; m < I.rows; m++) {
		J.at<Vec3i>(m, 0) = Vec3i(I.at<Vec3b>(m, 0)) + J.at<Vec3i>(m - 1, 0);
	}
	for (int m = 1; m < I.rows; m++) {
		for (int n = 1; n < I.cols; n++) {
			J.at<Vec3i>(m, n) = Vec3i(I.at<Vec3b>(m, n)) + J.at<Vec3i>(m - 1, n) + J.at<Vec3i>(m, n - 1) - J.at<Vec3i>(m - 1, n - 1);
		}
	}
	return J;
}

template <typename T>
inline void add(Mat I, Mat J) {
	// In place add J to I.
	for (int m = 1; m < I.rows; m++) {
		for (int n = 1; n < I.cols; n++) {
			I.at<T>(m, n) += J.at<T>(m, n);
		}
	}
}


inline void div(Mat I, int x) {
	// In place add J to I.
	for (int m = 1; m < I.rows; m++) {
		for (int n = 1; n < I.cols; n++) {
			I.at<Vec3i>(m, n) /= x;
		}
	}
}

template <typename T>
inline T zoneSum(Mat I, int i1, int i2, int j1, int j2) {
	// 
	return I.at<T>(i2, j2) - I.at<T>(i1, j2) - I.at<T>(i2, j1) + I.at<T>(i1, j1);
}

Vec3i div(Vec3i a, Vec3i b) {
	return Vec3i(a[0] / b[0], a[1] / b[1], a[2] / b[2]);
}


/*
Rescaling algorithms.

*/
Mat imageRescaled(Mat ImgIntegral, Mat ImgMean, Mat originIm,  int dx, int dy) {
	/* 
	Inputs:
	- imgIntegral: the integral image of the current image to rescale
	- imgMean: the integral image of the reconstructed image.
	- originIm: the original image of the current image to rescale.
	- dx, dy: the sizes of the rectangles.
	
	Rescales an image luminosity : 
	- each dx dy rectangle is fitted to the mean luminosity of the supposed reconstructed image.
	*/
	int m = ImgIntegral.rows;
	int n = ImgIntegral.cols; 
	int i1, i2, j1, j2;
	Mat I(m, n, CV_8UC3);
	for (int i = 0; i < m; i++) {
		i1 = i - dx;
		i2 = i + dx;
		if (i1 < 0) { i1 = 0; }
		if (i2 >= m) { i2 = m - 1; }

		for (int j = 0; j < n; j++) {
			j1 = j - dy;
			j2 = j + dy;
			if (j1 < 0) { j1 = 0; }
			if (j2 >= n) { j2 = n - 1; }

			I.at<Vec3b>(i, j) = originIm.at<Vec3b>(i, j) * (norm(zoneSum<Vec3i>(ImgMean, i1, i2, j1, j2))  / norm(zoneSum<Vec3i>(ImgIntegral, i1, i2, j1, j2)));
		}
	}
	return I;
}

vector<Mat> rescaleLuminsoity(vector<Mat> Imgs, int dx, int dy) {
	/*
	Inputs:
	- Imgs: the vector containing all the images which luminosity is to rescale.
	- dx, dy: the sizes of the rectangles.

	Rescales an image luminosity :
	- each dx dy rectangle is fitted to the mean luminosity of the images.
	*/

	int m = Imgs[0].rows; // the size of the images
	int n = Imgs[0].cols; 
	Mat ImgsMean(m, n, CV_32SC3);	// the mean image pixel values
	vector<Mat> ImgsIntegral;		// the integral images of each image
	vector<Mat> ImgsRescale;		// the rescaled images

	// Integrating the images
	for (int i = 0; i < Imgs.size(); i++) {
		ImgsIntegral.push_back(integral(Imgs[i]));
	}

	// Computing the mean image
	ImgsMean = ImgsIntegral[0].clone();
	for (int i = 1; i < Imgs.size(); i++) {
		add<Vec3i>(ImgsMean, ImgsIntegral[i]);
		cout << i;
	}

	div(ImgsMean, Imgs.size());

	// Rescaling
	for (int i = 0; i < Imgs.size(); i++) {
		ImgsRescale.push_back(imageRescaled(ImgsIntegral[i], ImgsMean, Imgs[i], dx, dy));
		imshow("Image " + to_string(i), ImgsRescale[i]);
	}

	return ImgsRescale;
}


vector<Mat> rescaleLuminsoity(vector<Mat> Imgs, Mat refImg, int dx, int dy) {
	// Rescale Luminosity on the basis of a naive reconstructed image
	int m = Imgs[0].rows;
	int n = Imgs[0].cols;
	Mat PixelSums(m, n, CV_32SC3);
	vector<Mat> ImgsIntegral;
	vector<Mat> ImgsRescale;
	for (int i = 0; i < Imgs.size(); i++) {
		ImgsIntegral.push_back(integral(Imgs[i]));
	}

	for (int m = 1; m < refImg.rows; m++) {
		for (int n = 1; n < refImg.cols; n++) {
			PixelSums.at<Vec3i>(m, n) = Vec3i(refImg.at<Vec3b>(m, n));
		}
	}


	for (int i = 0; i < Imgs.size(); i++) {
		ImgsRescale.push_back(imageRescaled(ImgsIntegral[i], PixelSums, Imgs[i], dx, dy));
		imshow(to_string(i), ImgsRescale[i]);
	}

	return ImgsRescale;
}


/*
Finding more relevant pixels on the occurences

*/
Mat findMaxOccurence(vector<Mat> Imgs, int threshhold = 15) {

	// compter les occurences par pixel et choisir max
	Vec3b rgb;
	int nb_images = Imgs.size();
	Mat res = Imgs[0].clone();

	cout << "Computing the max occurence image on a set of " << to_string(Imgs.size()) << " images of size : " << Imgs[0].rows << "  -  " << Imgs[0].cols << endl;

	for (int m = 1; m < Imgs[0].rows - 1; m++) {
		for (int n = 1; n < Imgs[0].cols - 1; n++) {
			vector<Vec3b> couleurs;		// The vector containing the 'color boxes' for the current pixel to count. A box is a range around the color enabling some variataions around the color
			vector<int> occurences;		// The vector containing the number of pixels in a specified box 

			for (int i = 0; i < nb_images; i++) {
				// Meaning the color on a specific range
				rgb = Imgs[i].at<Vec3b>(m, n) / 5 + Imgs[i].at<Vec3b>(m + 1, n) / 5 + Imgs[i].at<Vec3b>(m - 1, n) / 5 + Imgs[i].at<Vec3b>(m, n + 1) / 5 + Imgs[i].at<Vec3b>(m, n - 1) / 5;
				rgb = rgb;

				// Adding the color in the correct box
				bool trouve = false;
				for (int j = 0; j < couleurs.size(); j++) {
					if (norm(couleurs[j], rgb, CV_L2) < threshhold) {	// If a box correponding to the pixel is found, then its number of occurences is increased by 1.
						occurences[j] ++;
						trouve = true;
						break;
					}
				}

				if (!trouve) {										// Else, a new box is created.
					couleurs.push_back(rgb);
					occurences.push_back(1);
				}
			}

			// Finding the most filled box for the pixel
			int argmax = distance(occurences.begin(), max_element(occurences.begin(), occurences.end()));
			res.at<Vec3b>(m, n) = couleurs[argmax];
		}
	}
	return res;
}

vector<Mat> findGoodPixels(vector<Mat> Imgs, Mat reference, int threshhold = 15) {
	// Compute the map of the ccorrect pixels on all images of the list. 255 holds for a good pixel, 0 for a wrong one.
	vector<Mat> goodPixels;
	
	for (int i = 0; i < Imgs.size(); i++) {
		Mat temp = Mat(Imgs[0].size(), CV_8U);		// good pixels map
		for (int m = 1; m < Imgs[0].rows - 1; m++) {
			for (int n = 1; n < Imgs[0].cols - 1; n++) {
				// Comparing the current pixel to the reference one.
				Vec3b v1 = Imgs[i].at<Vec3b>(m, n) / 5 + Imgs[i].at<Vec3b>(m + 1, n) / 5 + Imgs[i].at<Vec3b>(m - 1, n) / 5 + Imgs[i].at<Vec3b>(m, n + 1) / 5 + Imgs[i].at<Vec3b>(m, n - 1) / 5;
				Vec3b v2 = reference.at<Vec3b>(m, n);
				float dist = norm(v1, v2, CV_L2);
				if (dist < threshhold) {
					temp.at<unsigned char>(m, n) = 255;
				}
				else {
					temp.at<unsigned char>(m, n) = 0;
				}
			}
		}
		goodPixels.push_back(temp);
	}
	return goodPixels;
}

/*
Reconstructing algorithms

*/
Mat reconstruct_image_mean(vector<Mat> goodPixels, vector<Mat> imgs) {
	int m = imgs[0].rows; 
	int n = imgs[0].cols; 
	int s;
	Vec3i currentPixel;
	Mat I(m, n, CV_8UC3);	// Reconstructed image
	Mat NB(m, n, CV_8U);	// Number  of good images for a given pixel

	// Finding the correct value of the pixel
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
	imwrite("../nbGoodPixels_LumProblem2.jpg", NB);
	return I;
}

template <typename T>
Mat reconstruct_image(vector<Mat> goodPixels, vector<Mat> imgs) {
	int m = imgs[0].rows, n = imgs[0].cols;
	vector<int> relevancy, indices;
	Mat I(m, n, CV_8UC3), Choose(m, n, CV_8UC3);
	
	// Indexing colors for labeling images
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
	vector<Mat> Imgs;
	for (int i = 1; i <= nb_images; i++) {
		Imgs.push_back(imread(nom_image + to_string(i) + extension));
		sums.push_back(norm(sum(Imgs[i - 1])));
	}

	
	
	
	double moy = sum(sums)[0] / nb_images;
	for (int i = 0; i < nb_images; i++) {
		Imgs[i] = Imgs[i] * moy / sums[i];
		cout << i << " " << moy << " " << sums[i] << endl;
	}
	




	vector<Mat> ImgsL = rescaleLuminsoity(Imgs, 40, 40);
	Imgs = ImgsL;
	//waitKey(0);

	// initialisation de l'image resultat
	int threshhold = 15;
	Mat reference = findMaxOccurence(Imgs, threshhold);


	vector<Mat> goodPixels = findGoodPixels(Imgs, reference, threshhold);

	//Mat resul = reconstruct_image<unsigned char>(goodPixels, Imgs);
	Mat resul = reconstruct_image_mean(goodPixels, Imgs);
	imshow("Reconstructed image", resul);

	//vector<Mat> ImgsR = rescaleLuminsoity(Imgs, resul, 40, 40);
	//waitKey(0);
	imwrite("../resultatLuminosity.jpg", reference);
	imwrite("../resultatReconstructedLuminosity.jpg", resul);
	/*
	for (int i = 0; i < nb_images; i++) {
	namedWindow("Resultat" + to_string(i), i + 1);
	imshow("Resultat" + to_string(i), goodPixels[i]);
	}
	*/
	waitKey(0);
	return 0;

}

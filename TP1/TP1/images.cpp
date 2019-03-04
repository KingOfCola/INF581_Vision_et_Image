#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>

#include <iostream>

using namespace cv;
using namespace std;

Mat float2byte(const Mat& If) {
	double minVal, maxVal;
	minMaxLoc(If,&minVal,&maxVal);
	Mat Ib;
	If.convertTo(Ib, CV_8U, 255.0/(maxVal - minVal),-minVal * 255.0/(maxVal - minVal));
	return Ib;
}

void onTrackbar(int sigma, void* p)
{
	Mat A=*(Mat*)p;
	Mat B;
	if (sigma) {
		GaussianBlur(A,B,Size(0,0),sigma);
		imshow("images",B);
	} else
		imshow("images",A);
}

Mat convol(const Mat& If, const Mat& p) {
	/*
	Calcule la 'convolution' de la matrice If par la matrice p
	Fonction pour simplifier les calculs de dérivées.
	*/
	int dm = p.rows, dn = p.cols;
	int m = If.rows, n = If.cols;
	Mat Grad(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float g = 0;
			if (i <= dm / 2 || i >= m - dm / 2 || j <= dn / 2 || j >= n - dn / 2)
				g = 0;
			else {
				for (int di = 0; di < dm; di++) {
					for (int dj = 0; dj < dn; dj++) {
						int a = i + di - dm / 2;
						int b = j + dj - dn / 2;
						g = g + float(If.at<uchar>(i + di - dm / 2, j + dj - dn / 2)) * p.at<float>(di, dj);
					}
				}
			}
			Grad.at<float>(i, j) = g;
		}
	}
	return Grad;
}

Mat normale(const Mat& x, const Mat& y) {
	/*
	Calcule la norme de la matrice dont les composantes sont contenues dans x et y
	*/
	int m = x.rows, n = y.cols;
	float ix, iy;
	Mat G(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			ix = x.at<float>(i, j);
			iy = y.at<float>(i, j);
			G.at<float>(i, j) = sqrt(ix*ix + iy*iy);
		}
	}
	return G;
}

Mat gradX(const Mat& If) {
	/*
	Calcule la dérivée sur x lissée selon y.
	*/
	Mat Cx(3, 3, CV_32F);
	Cx.at<float>(0, 0) = -1;
	Cx.at<float>(0, 1) = -2;
	Cx.at<float>(0, 2) = -1;
	Cx.at<float>(1, 0) = 0;
	Cx.at<float>(1, 1) = 0;
	Cx.at<float>(1, 2) = 0;
	Cx.at<float>(2, 0) = 1;
	Cx.at<float>(2, 1) = 2;
	Cx.at<float>(2, 2) = 1;

	return convol(If, Cx);
}

Mat gradY(const Mat& If) {
	/*
	Calcule la dérivée sur y lissée selon x.
	*/
	Mat Cy(3, 3, CV_32F);
	Cy.at<float>(0, 0) = -1;
	Cy.at<float>(0, 1) = 0;
	Cy.at<float>(0, 2) = 1;
	Cy.at<float>(1, 0) = -2;
	Cy.at<float>(1, 1) = 0;
	Cy.at<float>(1, 2) = 2;
	Cy.at<float>(2, 0) = -1;
	Cy.at<float>(2, 1) = 0;
	Cy.at<float>(2, 2) = 1;

	return convol(If, Cy);
}

Mat gradient(const Mat& If) {
	/*
	Calcule la norme du gradient.
	*/
	Mat Gx, Gy, G;
	Gx = gradX(If);
	Gy = gradY(If);

	return normale(Gx, Gy);
	
}

Mat canny(const Mat& If, float s1, float s2) {
	int m = If.rows, n = If.cols;
	float sq1 = s1 * s1, sq2 = s2 * s2;
	Mat Gx, Gy, G, M;
	queue<Point> Q;
	Gx = gradX(If);
	Gy = gradY(If);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			G.at<float>(i, j) = Gx.at<float>(i, j) * Gx.at<float>(i, j) + Gy.at<float>(i, j) * Gy.at<float>(i, j);
		}
	}

	float ix, iy, a, b;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			if (G.at<float>(i, j) < s1) { M.at<float>(i, j) = 0; continue; }
			ix = Gx.at<float>(i, j);
			iy = Gy.at<float>(i, j);
			if (iy < 0) {
				ix = -ix; iy = -iy;
			}
			if (ix > -0.4 * iy) {
				if (ix > 2.4 * iy) {
					a = G.at<float>(i + 1, j);
					b = G.at<float>(i - 1, j);
				}
				else if (ix > 0.4 * iy) {
					a = G.at<float>(i + 1, j + 1);
					b = G.at<float>(i - 1, j - 1);
				}
				else {
					a = G.at<float>(i, j + 1);
					b = G.at<float>(i, j - 1);
				}
			}
			else {
				if (ix > -2.4 * iy) {
					a = G.at<float>(i + 1, j - 1);
					b = G.at<float>(i - 1, j + 1);
				}
				else {
					a = G.at<float>(i + 1, j);
					b = G.at<float>(i - 1, j);
				}
			}
			if (G.at<float>(i, j) > s2) { 
				M.at<float>(i, j) = 2; 
				Q.push(Point(i, j)); 
			} 
			else if (G.at<float>(i, j) > s2) {
				M.at<float>(i, j) = 1;
			}
			else {
				M.at<float>(i, j) = 0;
			}
		}
	}
}

Mat shrink(const Mat& If) {
	/*
	Renvoie une matrice correspondant à la matrice d'entrée en réduisant ses coefficients entre 0. et 1.
	*/
	int m = If.rows, n = If.cols;
	float mini, maxi, x;
	mini = If.at<float>(0, 0);
	maxi = If.at<float>(0, 0);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			x = If.at<float>(i, j);
			if (x < mini)
				mini = x;
			if (x > maxi)
				maxi = x;
		}
	}

	float dm = maxi - mini;
	if (dm == 0)
		dm = 1;

	Mat I(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			I.at<float>(i, j) = (If.at<float>(i, j) - mini) / dm;
		}
	}
	return I;
}

bool isLocalMax(const Mat& Gx, const Mat& Gy, const Mat& G, int i, int j) {
	/*
	Détermine si le point (i, j) de l'image est un point de crête sur l'image.

	Inputs :
		- Gx, Gy : dérivées partielles selon x et y (doivent avoir la même taille que l'image d'entrée)
		- i, j : coordonnées du pixel à tester.
	*/
	float x, y;
	int m = Gx.rows, n = Gx.cols;
	x = Gx.at<float>(i, j);
	y = Gy.at<float>(i, j);
	if (x < 0)
		x = -x; y = -y;

	// Direction du gradient.
	int di, dj;
	if (x > 2 * y) {di = 1; dj = 0;}
	else if (x > y / 2) {di = 1; dj = 1;}
	else if (x > -y / 2) { di = 0; dj = 1; }
	else if (x > -y * 2) { di = -1; dj = 1; }
	else { di = 1; dj = 0; }

	// Test du maximum local
	if (i + di >= 0 && i + di < m && i - di >= 0 && i - di < m && j + dj >= 0 && j + dj < n && j - dj >= 0 && j - dj < n) {
		float A, B, C;
		A = G.at<float>(i, j);
		B = G.at<float>(i + di, j + dj);
		C = G.at<float>(i - di, j - dj);
		return (B < A && C < A);
	}
	else
		return false;
}

Mat isContour(const Mat& If) {
	/*
	Calcule les contours affinés de l'image d'entrée.

	Inputs :
	- If : image en niveaux de gris.

	Outputs :
	Matrice contenant 1 à la position des contours, 0 sinon.
	*/
	Mat Gx, Gy, G, I;
	Gx = gradX(If);
	Gy = gradY(If);
	G = normale(Gx, Gy);
	int m = If.rows, n = If.cols;
	I = Mat(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (isLocalMax(Gx, Gy, G, i, j) && G.at<float>(i, j) > 0) { I.at<float>(i, j) = 1; }
			else { I.at<float>(i, j) = 0; }
		}
	}
	return I;
}

Mat findContoursAll(const Mat& G, const Mat& allContours, const float s1, const float s2) {
	/*
	Calcule les contours affinés, seuillés et prolongés de l'image d'entrée.
	
	Inputs :
		- G : Gradient de l'image.
		- allContours : les contours affinés de l'image.
		- s1 : Seuil primaire de sélection des crêtes.
		- s2 : Seuil secondaire de continuation des crêtes.

	Outputs :
		Matrice contenant 1 à la position des contours, 0 sinon.
	*/
	queue<Point> contours = queue<Point>();
	int m = G.rows, n = G.cols;
	Mat C = Mat(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (allContours.at<float>(i, j) == 1. && G.at<float>(i, j) > s1) {
				contours.push(Point(i, j));
				C.at<float>(i, j) = 1.;
			}
			else {
				C.at<float>(i, j) = 0.;
			}
		}
	}

	int i, j;
	while (!(contours.empty())) {
		i = contours.front().x;
		j = contours.front().y;
		contours.pop();
		if (allContours.at<float>(i, j) == 1. && C.at<float>(i, j) == 0. && G.at<float>(i, j) > s2) {
			C.at<float>(i, j) = 1.;
			for (int di = -1; di <= 1; di++) {
				for (int dj = -1; dj <= 1; dj++) {
					if ((di != 0 || dj != 0) && i + di >= 0 && i + di < n &&  j + dj >= 0 && j + dj < m) {
						contours.push(Point(i + di, j + dj));
					}
				}
			}
		}
	}

	return C;
}


int main()
{
    Mat A=imread("C:/Users/Urvan/Documents/X/3A/INF/TP1/TP1/fruits.jpg");
	//namedWindow("images");
	//imshow("images",A);	waitKey();

	
	Mat I;
	cvtColor(A,I,CV_BGR2GRAY);
	Mat G, J, K, L;
	G = gradient(I);
	J = shrink(G);
	K = isContour(I);
	string s = "y";
	float s1, s2;
	while (s != "n" && s != "") {
		cout << "Seuil primaire : "; cin >> s1;
		cout << "Seuil secondaire : "; cin >> s2;

		L = findContoursAll(G, K, s1, s2);
		//imshow("image couleurs réelles", A); 
		imshow("image niveau de gris", I);
		//imshow("Gradient corrigé (foncé : faible, clair, fort)", float2byte(J)); 
		//imshow("Contours affinés", float2byte(K)); 
		imshow("Contours sélectionnés", float2byte(L)); waitKey();
		cout << "Nouveau test y/[n] : "; cin >> s;
	}
	/*
	int m=I.rows,n=I.cols;
	Mat Ix(m,n,CV_32F),Iy(m,n,CV_32F),G(m,n,CV_32F);
	for (int i=0;i<m;i++) {
		for (int j=0;j<n;j++){
			float ix,iy;
			if (i==0 || i==m-1)
				iy=0;
			else
				iy=(float(I.at<uchar>(i+1,j))-float(I.at<uchar>(i-1,j)))/2;
			if (j==0 || j==n-1)
				ix=0;
			else
				ix=(float(I.at<uchar>(i,j+1))-float(I.at<uchar>(i,j-1)))/2;
			Ix.at<float>(i,j)=ix;
			Iy.at<float>(i,j)=iy;
			G.at<float>(i,j)=sqrt(ix*ix+iy*iy);
		}
	}
	imshow("images",float2byte(Ix));waitKey();
	imshow("images",float2byte(Iy));waitKey();
	imshow("images",float2byte(G));waitKey();

	Mat C;
	threshold(G,C,10,1,THRESH_BINARY);
	imshow("images",float2byte(C));waitKey();


	createTrackbar("sigma","images",0,20,onTrackbar,&A);
	imshow("images",A);	waitKey();
	*/
	return 0;
}

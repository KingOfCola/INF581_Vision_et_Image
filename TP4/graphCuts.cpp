#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

#include "maxflow/graph.h"

using namespace std;
using namespace cv;

// This section shows how to use the library to compute a minimum cut on the following graph:
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      4    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////

void testGCuts()
{
	Graph<int,int,int> g(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1); 
	g.add_node(2); 
	g.add_tweights( 0,   /* capacities */  1, 5 );
	g.add_tweights( 1,   /* capacities */  6, 1 );
	g.add_edge( 0, 1,    /* capacities */  4, 3 );
	int flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i=0;i<2;i++)
		if (g.what_segment(i) == Graph<int,int,int>::SOURCE)
			cout << i << " is in the SOURCE set" << endl;
		else
			cout << i << " is in the SINK set" << endl;
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

struct Data {
	Mat I1;
	bool fin = false, fout = false;
	Vec3b cin, cout;
};

inline void presskey() {
	cout << "Press any key to continue..." << endl;
}

inline float g_f(float x) {
	return 1.f / (1 + x * x);
}

void onMouse1(int event, int x, int y, int foo, void* p)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	Point m1(x, y);

	Data* D = (Data*)p;
	D->cin = D->I1.at<Vec3b>(y, x);
	if (D->fin) {
		cout << "La couleur du poisson a ete mise a jour a : " << D->cin << endl;
	}
	else {
		D->fin = true;
		cout << "La couleur du poisson a ete initialisee a : " << D->cin << endl;
		presskey();
	}
}

void onMouse2(int event, int x, int y, int foo, void* p)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	Point m1(x, y);

	Data* D = (Data*)p;
	D->cout = D->I1.at<Vec3b>(y, x);
	if (D->fout) {
		cout << "La couleur du fond a ete mise a jour a : " << D->cout << endl;
	}
	else {
		D->fout = true;
		cout << "La couleur du fond a ete initialisee a : " << D->cout << endl;
		presskey();
	}
}


int main() {
	testGCuts();

	float alpha, beta;

	Mat I=imread("../fishes.jpg");
	int m = I.rows, n = I.cols;
	Mat A;
	cvtColor(I, A, CV_BGR2GRAY);
	Graph<float, float, float> g(m * n, 1);
	g.add_node(m * n);
	Mat G = gradient(I);

	cout << "Entrez un coefficient pour l'attache aux donnees : ";
	cin >> alpha;
	cout << "Entrez un coefficient pour la regularite : ";
	cin >> beta;

	imshow("I",I);
	Data D;
	D.I1 = I;
	setMouseCallback("I", onMouse1, &D);

	cout << "Cliquez sur un point du poisson : " << endl;
	waitKey(0);
	setMouseCallback("I", onMouse2, &D);
	cout << "Cliquez sur un point du fond : " << endl;
	waitKey(0);




	Scalar Iin(D.cin), Iext(D.cout);
	float din, dext;
	
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			din = (float)(norm(Scalar(I.at<Vec3b>(i, j)) - Iin));
			dext = (float)(norm(Scalar(I.at<Vec3b>(i, j)) - Iext));
			g.add_tweights(i * n + j, alpha * din, alpha * dext);
			if (i < m - 1) {
				float a = g_f(((G.at<float>(i, j) + G.at<float>(i + 1, j)) / 2.));
				g.add_edge(i * n + j, (i + 1) * n + j, beta * a, beta * a);
				if (i == 50 && j == 50)
					cout << din * alpha << endl << dext * alpha << endl << a * beta;
			}
			if (j < n - 1) {
				float a = g_f((G.at<float>(i, j) + G.at<float>(i, j + 1)) / 2.);
				g.add_edge(i * n + j, i * n + j + 1, beta * a, beta * a);
			}
		}
	}

	Mat C(m, n, CV_32F);
	g.maxflow();

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (g.what_segment(i * n + j) == Graph<float, float, float>::SOURCE) {
				C.at<float>(i, j) = 1.;
			}
			else {
				C.at<float>(i, j) = 0.;
			}	
		}
	}


	imshow("I2", C);

	presskey();
	waitKey(0);
	return 0;
}

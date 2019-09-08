#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int compute_weight(Mat& img, Mat& tmap, int win_size, float eps){

	int num_win = (win_size*2 + 1)*(win_size*2 + 1); //number of pixels in window
	int num_win_sq = num_win*num_win;
	int N = img.rows*img.cols;

	int i, j; 
	for(i = win_size; i < img.rows-win_size; i++){
		for(j = win_size; j < img.cols-win_size; j++){

			// extract the window out of image
			Mat win = img.rowRange(i-win_size, i+win_size+1);
			win = win.colRange(j-win_size, j+win_size+1);
			Mat win_ravel = Mat::zeros(9, 3, CV_64F); //doubt ??
			double sum1 = 0;
			double sum2 = 0;
			double sum3 = 0; 

			int c = 0;
			for(int p = 0; p < win_size*2+1; p++){
				for(int q = 0; q < win_size*2+1; q++){
					win_ravel.at<double>(c,0) = win.at<cv::Vec3b>(p,q)[0]/255.0;
					win_ravel.at<double>(c,1) = win.at<cv::Vec3b>(p,q)[1]/255.0;
					win_ravel.at<double>(c,2) = win.at<cv::Vec3b>(p,q)[2]/255.0;
					// cout<<double(win.at<cv::Vec3b>(p,q)[0])<<endl;
					// exit(0);
					sum1 += win.at<cv::Vec3b>(p,q)[0]/255.0;
					sum2 += win.at<cv::Vec3b>(p,q)[1]/255.0;
					sum3 += win.at<cv::Vec3b>(p,q)[2]/255.0;
					c++;
				}
			}
			win = win_ravel;

			Mat win_mean = Mat::zeros(1, 3, CV_64F);
			win_mean.at<double>(0,0) = sum1/num_win; 
			win_mean.at<double>(0,1) = sum2/num_win; 
			win_mean.at<double>(0,2) = sum3/num_win; 

			// calculate the covariance matrix 
      		Mat covariance = (win.t() * win / num_win) - (win_mean.t() * win_mean);
      		Mat I = Mat::eye(img.channels(), img.channels(), CV_64F);
      		Mat inv = (covariance + eps / num_win * I).inv();

      		Mat X = win - repeat(win_mean, num_win, 1);
      		Mat vals = (1 + X * inv * X.t()) / num_win;
      		vals = vals.reshape(0, num_win_sq);
		}
	}


}


int main(){

	Mat image,tmap;
	// my_vector_of_vectors_t samples, indm, Euu;
	string img_path = "../data/input_lowres/plasticbag.png";
	image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

    string tmap_path = "../data/trimap_lowres/Trimap1/plasticbag.png";
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);
 //    int i, j;
 //    unordered_set<int> unk;
 //    for(i = 0; i < tmap.rows; i++)
 //    	for(j = 0; j < tmap.cols; j++){
 //    		float pix = tmap.at<cv::Vec3b>(i,j)[0];
 //    		if(pix == 128)
 //    			unk.insert(i*tmap.cols+j);
 //    	}

 //    cout<<"UNK: "<<unk.size()<<endl;

 //    kdtree_CM(image, indm, samples, unk);
	// cout<<"KD Tree done"<<endl;
	cout<<image.rows<<" "<<image.cols<<endl;
	// cout<<image.at<cv::Vec3b>(0,0)[0]<<"    ";

	float eps = 0.001;
	compute_weight(image, tmap, 1, eps);

}
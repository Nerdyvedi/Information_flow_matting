#include <iostream>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include <ctime>
#include <cstdlib>
using namespace nanoflann;
using namespace std;
using namespace cv;

const int dim = 5;

typedef vector<vector<double>> my_vector_of_vectors_t;
typedef vector<set<int, greater<int>>> my_vector_of_set_t;

void generateFVectorIntraU(my_vector_of_vectors_t &samples, Mat &img)
{
	// CV_Assert(img.depth() == CV_8U);

	int channels = img.channels();
	int nRows = img.rows;
	int nCols = img.cols;

	samples.resize(nRows*nCols);
	
	int i,j,k;	
	for( i = 0; i < nRows; ++i)
		for ( j = 0; j < nCols; ++j){
			samples[i*nCols+j].resize(dim);		
			samples[i*nCols+j][0] = img.at<cv::Vec3b>(i,j)[0];
			samples[i*nCols+j][1] = img.at<cv::Vec3b>(i,j)[1];
			samples[i*nCols+j][2] = img.at<cv::Vec3b>(i,j)[2];
			samples[i*nCols+j][3] = double(i/20);
			samples[i*nCols+j][4] = double(j/20);
		}

	cout << "feature vectors done"<<endl;
}


void kdtree_intraU(Mat &img, my_vector_of_vectors_t& indm, my_vector_of_set_t& ind, my_vector_of_vectors_t& samples)
{
	const double max_range = 20;

	// Generate feature vectors for intra U:
	generateFVectorIntraU(samples, img);	

	// Query point: same as samples from which KD tree is generated

	// construct a kd-tree index:
	// Dimensionality set at run-time (default: L2)
	// ------------------------------------------------------------
	typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
	my_kd_tree_t mat_index(dim /*dim*/, samples, 10 /* max leaf */ );
	mat_index.index->buildIndex();

	// do a knn search with ku  = 5
	const size_t num_results = 5; 

	int i,j;
	int N = img.rows*img.cols;

	// just for testing purpose ...delete this later!
	int c = 0; 

	vector<size_t> ret_indexes(num_results);
	vector<double> out_dists_sqr(num_results);
	nanoflann::KNNResultSet<double> resultSet(num_results);

	indm.resize(N);
	ind.resize(N);
	for(i = 0; i < N; i++){
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
		mat_index.index->findNeighbors(resultSet, &samples[i][0], nanoflann::SearchParams(10));	

		// cout << "knnSearch(nn="<<num_results<<"): \n";
		indm[i].resize(num_results);
		for (j = 0; j < num_results; j++){
			// cout << "ret_index["<<j<<"]=" << ret_indexes[j] << " out_dist_sqr=" << out_dists_sqr[j] << endl;
			ind[i].insert(ret_indexes[j]);
			indm[i][j] = ret_indexes[j];
		}
		c++;
		// if(c == 5)
		// 	return;
	}
}

double l1norm(vector<double>& x, vector<double>& y){
	double sum = 0;
	for(int i = 0; i < dim; i++)
		sum += abs(x[i]-y[i]);
	return sum;
}

void intraU(my_vector_of_vectors_t& indm, my_vector_of_set_t& inds, my_vector_of_vectors_t& samples, my_vector_of_vectors_t& Euu){

	// input: indm, inds, samples
	int num_nbr = 5;
	int N = indm.size();
	

	int i,j, curr_ind, nbr_ind;
	for(i = 0; i < N; i++){
		for(j = 0; j < num_nbr; j++){
			// lookup indm[i][j] in jth entry
			curr_ind = i;
			nbr_ind = indm[i][j];
			if(inds[nbr_ind].find(curr_ind) == inds[nbr_ind].end())
				indm[nbr_ind].push_back(curr_ind);
		}
	}

	my_vector_of_vectors_t weights;
	weights.resize(N);
	for(i = 0; i < N; i++){
		weights[i].resize(indm[i].size());
		for(j = 0; j < indm[i].size(); j++)
			weights[i][j] = max(l1norm(samples[i], samples[j]), 0.0);
	}

	cout<<"intraU weights computed"<<endl;
	// Euu
	
}

int main()
{
	Mat image,tmap;
	my_vector_of_vectors_t samples, indm, Euu;
	my_vector_of_set_t inds;
	string img_path = "../data/input_lowres/plasticbag.png";
	image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file
	kdtree_intraU(image, indm, inds, samples);
	cout<<"KD Tree done"<<endl;
	intraU(indm, inds, samples, Euu);
}

const int dim = 5; //dimension of feature vectors 

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "KtoU.cpp"
#include "intraU.cpp"
#include "cm.cpp"
#include "local_info.cpp"
#include "Eigen/IterativeLinearSolvers"
#include "trimming.cpp"

using namespace std;
using namespace cv;
// String output; 


void show(Mat& image){
    namedWindow( "Display window", WINDOW_AUTOSIZE );    // Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
}

int check_image(Mat& image){
    if(!image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    return 0;
}

void type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');
  cout<<r<<endl;

}

void solve(SparseMatrix<double> Wcm,SparseMatrix<double> Wuu,SparseMatrix<double> Wl,SparseMatrix<double> Dcm,
    SparseMatrix<double> Duu,SparseMatrix<double> Dl,SparseMatrix<double> H,SparseMatrix<double> T,
    Mat &ak,Mat &wf, bool useKU, Mat &alpha){

    float sku = 0.05, suu = 0.01, sl = 1, lamd = 100;

    SparseMatrix<double> Lifm = ((Dcm-Wcm).transpose())*(Dcm-Wcm) + suu*(Duu-Wuu) + sl*(Dl-Wl);
    // # Lifm = suu*(Duu-Wuu) + sl*(Dl-Wl)
    SparseMatrix<double> A;
    int n = ak.cols; 
    VectorXd b(n), wf_(n), x(n);

 
    for(int i = 0; i < n; i++)
        wf_(i) = wf.at<uchar>(i,0);


    if(useKU){
        A = Lifm + lamd*T + sku*H;
        b = (lamd*T + sku*H)*(wf_);        
    }
    else{
        A = Lifm + lamd*T;
        b = (lamd*T)*(wf_);        
    }
    // # print(csr.sum(b))
    // M = diags(A.diagonal())
    // # print(A.shape)
    // # print(b.shape)

    ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
    cg.setMaxIterations(500);
    cg.compute(A);
    x = cg.solve(b);

    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;
    // alpha = cg(A, b, x0=wf, tol=1e-10, maxiter=700, M=None, callback=None, atol=None)
    // # alpha = spsolve(A, b)
    // # print(alpha)
    // # print(type(alpha[0]))
    // return alpha[0]
    // ###solve
    // # A = Lifm + lamd*T
    // # b = (lamd*T).dot(ak)
    // ###solve

    int nRows = alpha.rows; 
    int nCols = alpha.cols;
    for(int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j){
            // cout<<x(i*nCols+j)<<endl;
            alpha.at<uchar>(i,j) = x(i*nCols+j)*255;   
        }
    // show(alpha);
    cout<<"Done"<<endl;
}



int main(int argc, char** argv){
  

    clock_t begin = clock();
    Mat image,tmap;
    // const char* img_path = "../../data/input_lowres/elephant.png";
    char* img_path = argv[1];
    cout<<img_path<<endl;
    image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

    check_image(image);
    // show(image);
    cout<<image.size<<endl;
    cout<<image.channels()<<endl;

    // const char* tmap_path = "../../data/trimap_lowres/Trimap1/elephant.png";
    char* tmap_path = argv[2];
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);
    check_image(tmap);
    // show(tmap);


    int nRows = image.rows;
    int nCols = image.cols; 
    int N = nRows*nCols;

    Mat ak, wf;
    SparseMatrix<double> T(N,N);
    typedef Triplet<double> Tr;
    vector<Tr> triplets;
    // triplets.reserve(N*N);

    ak.create(1, nRows*nCols, CV_8U); 
    wf.create(nRows*nCols, 1, CV_8U); 
    for(int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j){
            float pix = tmap.at<uchar>(i,j);
            if(pix != 128)                         //collection of known pixels samples
                triplets.push_back(Tr(i*nCols+j, i*nCols+j, 1));
            else triplets.push_back(Tr(i*nCols+j, i*nCols+j, 0));
            if(pix > 200)                                   //foreground pixel
                ak.at<uchar>(0,i*nCols+j) = 1;
            else ak.at<uchar>(0,i*nCols+j) = 0;
            wf.at<uchar>(i*nCols+j,0) = ak.at<uchar>(0,i*nCols+j);
        }


    SparseMatrix<double> Wl(N, N), Dl(N, N); 
    local_info(image,tmap,Wl,Dl);

    SparseMatrix<double> Wcm(N, N), Dcm(N, N); 
    cm(image, tmap, Wcm, Dcm);
    // # Wcm = csr((h*w,h*w))
    // wk,H = ku(img,tmap,X)

    Mat new_tmap = tmap.clone(); //after pre-processing
    // trimming(image, tmap, new_tmap, tmap, true);
    trimming(image, tmap, new_tmap, false);
    SparseMatrix<double> H = KtoU(image, new_tmap, wf);
    // # H = csr((h*w,h*w))
    // # wk = csr((wk.shape))
    // Wuu = intra_u(img,tmap,X)
    SparseMatrix<double> Wuu(N, N), Duu(N, N); 
    UU(image, tmap, Wuu, Duu);
    // # Wuu = csr((h*w,h*w)) 


    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"time for flow calc: "<<elapsed_secs<<endl;

    T.setFromTriplets(triplets.begin(), triplets.end());
    // Mat calc_alpha = solve(Wcm,Wuu,Wl,Dcm,Duu,Dl,H,T,ak,wf,true);

    Mat alpha; 
    alpha.create(nRows, nCols, CV_8UC1);
    solve(Wcm,Wuu,Wl,Dcm,Duu,Dl,H,T,ak,wf,false,alpha);

    Mat trim_alpha = alpha.clone();
    cout<<"Solved"<<endl;
    // trimming(image, tmap, new_tmap, tmap, true);
    int i, j;
    for(i = 0; i < image.rows; i++){
        for(j = 0; j < image.cols; j++){
            float pix = new_tmap.at<uchar>(i,j);
            if(pix != 128){
                // cout<<"in "<<float(trim_alpha.at<uchar>(i, j))<<endl;
                trim_alpha.at<uchar>(i, j) = pix; 
            }
        }
    }

    cout<<"Trimmed"<<endl;
    char* res_path = argv[3];
    // imwrite("elephant_alpha_trim.png", trim_alpha);
    imwrite(res_path, trim_alpha);

    /*
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"total time: "<<elapsed_secs<<endl;
    */
    
    return 0;
} 


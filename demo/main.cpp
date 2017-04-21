//
//  main.cpp
//  demo
//
//  Created by morpheus on 21/04/2017.
//  Copyright © 2017 morpheus. All rights reserved.
//

#include <iostream>
#include <unistd.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

/**
 * sobelOperate : take Sobel operation on srcImage for edge detection and return the matrix of magnitude of gradient
 *                  the return matrix can be used to calculate Non-maximun-suppression
 */
Mat sobelOperate(const Mat &srcImage, Mat &resultImage, float minThresh, float maxThresh) {
    
    int nRows = srcImage.rows - 2;
    int nCols = srcImage.cols - 2;
    
    // Define Sobel kernel for X and Y direction
    Mat sobelx = (Mat_<double>(3,3) << 1, 0, -1, 2, 0, -2,  1,  0, -1);
    Mat sobely = (Mat_<double>(3,3) << 1, 2,  1, 0, 0,  0, -1, -2, -1);
    
    // Define Matrix of gradient value for Non-Maximum Suppression operation
    Mat gradMagMat = Mat::zeros(nRows, nCols, CV_32F);
    
    // Define Variables for detected edge value for X and Y direction
    double edgeX = 0;
    double edgeY = 0;
    // Define Variable for Gradient value
    double gradMag = 0;
    
    // perform Sobel Operation
    for (int k = 1; k < srcImage.rows - 1; ++k) {
        for (int n = 1; n < srcImage.cols - 1; ++n) {
            edgeX = 0;
            edgeY = 0;
            
            // Calculate gradient for both X and Y direction
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    edgeX += srcImage.at<uchar>(k+i, n+j) * sobelx.at<double>(1+i, 1+j);
                    edgeY += srcImage.at<uchar>(k+i, n+j) * sobely.at<double>(1+i, 1+j);
                }
            }
            
            // Calculate magnitude of gradient
            gradMag = sqrt(pow(edgeY, 2) + pow(edgeX, 2));
            // convert to binary image
            resultImage.at<uchar>(k-1, n-1) = ((gradMag > minThresh) && (gradMag < maxThresh) ? 255 : 0);
            // store gradient magnitude for calculate nms operation
            gradMagMat.at<float>(k-1, n-1) = gradMag;
        }
    }
    
    return gradMagMat;
}

/**
 * nmsOperate: take Non-Maximum-Suppression operation of gradMagMat for eliminate false edge and return the result image
 */
Mat nmsOperate(Mat gradMagMat, float minThresh) {
    
    const int nRows = gradMagMat.rows;
    const int nCols = gradMagMat.cols;
    
    Mat tempImageMat = Mat::zeros(nRows, nCols, gradMagMat.type());
    
    float *pDataMag = (float *)gradMagMat.data;
    float *pDataTmp = (float *)tempImageMat.data;
    
    printf("nRows: %d\n", nRows);
    printf("nCols: %d\n", nCols);
    
    // NMS operation: Compare with adjacent pixel for checking the pixel is Edge or not
    for (int i=1; i != nRows - 1; ++i) {
        for (int j=1; j != nCols - 1; ++j) {
            
            bool b1 = (pDataMag[i * nCols + j] > pDataMag[i * nCols + j - 1]);
            bool b2 = (pDataMag[i * nCols + j] > pDataMag[i * nCols + j + 1]);
            bool b3 = (pDataMag[i * nCols + j] > pDataMag[(i-1) * nCols + j]);
            bool b4 = (pDataMag[i * nCols + j] > pDataMag[(i+1) * nCols + j]);
            pDataTmp[i * nCols + j] = 255 * ((pDataMag[i * nCols + j] > minThresh) && ((b1 && b2) || (b3 && b4) ));
        }
    }
    
    tempImageMat.convertTo(tempImageMat, CV_8UC1);
    return tempImageMat.clone();

}


int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n";
    
    char *dir = getcwd(NULL, 0);
    if (!dir) {
        return 2;
    }
    
    string sourceFile = "/test6.jpg";
    string strSobelFile = "/sobel_image.jpg";
    string strNMSFile = "/nms_image.jpg";

    string strDir = string(dir);
    string strSourcePath = strDir + sourceFile;
    Mat srcImage = cv::imread(strSourcePath);
    
    if (srcImage.empty()) {
        return 1;
    }
    
    Mat srcGray;
    cvtColor(srcImage, srcGray, CV_RGB2GRAY);
    
    Mat sobelImage = Mat::zeros(srcGray.rows-2, srcGray.cols-2, srcGray.type());
    Mat gradMagMat = sobelOperate(srcGray, sobelImage, 50, 200);
    Mat nmsImage = nmsOperate(gradMagMat, 50);

    string target_path = string(dir);
    target_path = strDir + strSobelFile;
    printf("Current dir: %s\n", target_path.c_str());
    imwrite(target_path, sobelImage);
    
    target_path = strDir + strNMSFile;
    printf("Current dir: %s\n", target_path.c_str());
    imwrite(target_path, nmsImage);
    
    
//    namedWindow("test");
//    imshow("Gray", srcGray);
//    imshow("Sobel", sobelImage);
//    imshow("nms", nmsImage);
//    waitKey(0);
    return 0;
}

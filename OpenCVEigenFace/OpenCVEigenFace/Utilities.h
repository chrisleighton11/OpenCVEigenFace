#ifndef UTILITIES_H
#define UTILITIES_H

/*
   Utilties.h
   Description:   defines useful functions
   Author:        Chris Leighton
   Date:          Feb 10th 2010

*/

#include <string>
#include <iostream>
#include <sstream>

#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>


typedef std::vector<CvRect*>        RectVec;
typedef std::vector<IplImage*>      ImageVec;

typedef int personIDType;

typedef unsigned char uchar;

// convert image to grey scale
void ConvertToGreyScale(const IplImage* input, IplImage* output);

// convert float image to grey scale
IplImage* ConvertFloatToGreyScale( const IplImage* image );


// perform histogram equalization on image
void HistogramEqualization(const IplImage* input, IplImage* output);


// resize image
void Resize(const IplImage* input, IplImage* output, int cols, int rows);

// ImageToMatrixf copies float image to matrix
void ImageToMatrixf( const IplImage* input, float* output, int output_width);

// input is not float image
void ImageToMatrix( const IplImage*  input, float* output, int output_width);


template <typename T>
double Avg( const T* src, int nEle)
{
   double sum(0.0);

   for ( int i = 0; i < nEle; i++ )
   {
      sum += src[i];
   }

   return (double)sum/(double)nEle;
}


template <typename T>
double sDev( const T* src, int nEle, double mean )
{
   double sum(0.0);

   for ( int i = 0; i < nEle; i++ )
   {
      sum += ( ( src[i] - mean )*(src[i] - mean) );
   }

   return sqrt(sum / (double)(nEle - 1));
}









/*

how to get the time is takes to do somthing in ms
int ms
t = (double)cvGetTickCount();
t = (double)cvGetTickCount() - t;
ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
*/



#endif
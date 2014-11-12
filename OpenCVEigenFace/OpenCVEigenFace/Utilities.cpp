
#include "Utilities.h"



/* 
   Function:   ConvertToGrayScale
   Purpose:    Converts color image to grey scale 
   Notes:      
   Throws      
*/
void ConvertToGreyScale(const IplImage* input, IplImage* output)
{
   if ( !input )
      throw std::string("ConvertToGreyScale received null image as argument");
   
   if ( input->nChannels > 1 )
   {
      cvCvtColor( input, output, CV_BGR2GRAY );
   }
   else
   {
      output = cvCloneImage(input);
   }
}


/* 
   Function:   HistogramEqualization
   Purpose:    Performs Histogram Equalization
   Notes:      Not Needed
   Throws      
*/
void HistogramEqualization(const IplImage* input, IplImage* output)
{
   cvEqualizeHist(input, output);
}



/* 
   Function:   Resize
   Purpose:    Resizes image
   Notes:      Does not keep apect ratio the same
   Throws      
*/
void Resize(const IplImage* input, IplImage* output, int cols, int rows)
{
   int flag = 0;

   if (cols < input->width && rows < input->height )
      flag |= CV_INTER_AREA; // shrinking
   else
      flag |= CV_INTER_LINEAR;
   
   cvResize(input, output, flag);
}


/*
   Function: ConvertFloatToGreyScale
   Purpose:  Given a float image, convert it to grey scale
   Notes:    users should release returned image themselves
   Throws:   std::string if it can't create the new image
   Returns:  Grey Scale Image
*/
IplImage* ConvertFloatToGreyScale( const IplImage* image )
{
	IplImage* result = NULL;

	double min, max;
	
	// get the min and max values with help from openCV
	cvMinMaxLoc(image, &min, &max);

	// deal with NaN
	if (cvIsNaN(min) || min < -1e30)
	 	min = -1e30;
	if (cvIsNaN(max) || max > 1e30)
		max = 1e30;
	if (max-min == 0.0f)
		max = min + 0.001;

	result = cvCreateImage(cvSize(image->width, image->height), 8, 1);
	if ( !result )
		throw std::string("ConvertFloatToGreyScale could not create image.");

	// thankyou openCV
	cvConvertScale(image, result, 255.0 / (max - min), - min * 255.0 / (max-min));

	return result;
}


/*
   Function: ImageToMatrixf
   Purpose:  Given a float image, store in row of matrix, type is float
   Notes:    image is assumed to be grey scale
   Throws:   std::sting is size is wrong, or image is not grey scat
   Returns:  
*/
void ImageToMatrixf( const IplImage* input, float* output, int output_width)
{
   if ( input->nChannels != 1 )
      throw std::string("ImageToMatrixf - input not grey scale image" );

   for ( int row = 0; row < input->height; row++ )
   {
      float* therow = (float*)(input->imageData + (row*input->widthStep));

      for ( int col = 0; col < input->width; col++ )
      {
         output[row*input->width+col] = (float)therow[col];
      }
   }
}


/*
   Function: ImageToMatrix
   Purpose:  Given an image, store in row of matrix, type is float
   Notes:    image is assumed to be grey scale
   Throws:   std::sting is size is wrong, or image is not grey scat
   Returns:  
*/
void ImageToMatrix( const IplImage* input, float* output, int output_width)
{
   if ( input->nChannels != 1 )
      throw std::string("ImageToMatrixf - input not grey scale image" );

   for ( int row = 0; row < input->height; row++ )
   {
      unsigned char* therow = (unsigned char*)(input->imageData + (row*input->widthStep));

      for ( int col = 0; col < input->width; col++ )
      {
         output[row*input->width+col] = (float)therow[col];
      }
   }
}



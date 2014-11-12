#ifndef STANDARDIZE_H
#define STANDARDIZE_H

/* 
   Standardize.h
   Chris Leighton
   5/9/2011

   Provides methods to standardize matrices

*/


#include "Utilities.h"


/*
   Standardize using function Zij = ( Xij - mean(X) ) / Sdev

   Note the uses the column data - ie the mean is the mean for all elements in a row.
   this only works for a floating point matrix
*/
void stdMatSDEV( CvMat* src, CvMat* dst );








#endif
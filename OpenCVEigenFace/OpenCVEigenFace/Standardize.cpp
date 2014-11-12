#include "Standardize.h"


void stdMatSDEV( CvMat* src, CvMat* dst )
{
   int nrows = src->rows;
   int ncols = src->cols;

   if ( nrows != dst->rows || ncols != dst->cols )
      throw std::string("stdMatSDEV - src and dst matrices are not same size");

   for ( int r = 0; r < src->rows; r++ )
   {
      double avg = Avg(src->data.fl + (r*ncols), ncols);
      double sdev = sDev(src->data.fl + (r*ncols), ncols, avg);

      for ( int c = 0; c < ncols; c++ )
      {
         int index = r*ncols+c;
         dst->data.fl[index] = ( (src->data.fl[index] - avg) / sdev );
      }
   }
}
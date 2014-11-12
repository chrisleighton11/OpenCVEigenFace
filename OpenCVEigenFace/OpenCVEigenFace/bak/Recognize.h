#ifndef RECOGNIZE_H
#define RECOGNIZE_H

/*
   Recognize.h
   Description:   defines all of the classes and methods for the recognition part
   Author:        Chris Leighton
   Date:          Feb 20th 2010

*/


#include "Utilities.h"
#include <vector>


std::string Recognize(const char* image, const char* database, double& distance, std::string& resultsdir);



class Recognizer
{
public:
   Recognizer(const char* image, const char* database);
   ~Recognizer();


   bool        LoadTrainingDatabase();
   std::string FindFace( int faceNum, double& distance );
   int         EuclideanDistance( float* projectedTestFace, double& distance );
   int         MahalanobisDistance( float* projectedTestFace, double& distance );

   void	       GenResults(std::string& resultsdir);

private:

   /// Not implemented - might need
   void BetweenClassThreshold( int personID, double& e_threshold, double& m_threshold  );
   
   // training member variables
   const char*             m_DatabaseName;
   int                     m_nImages;           // number of images in database
   int                     m_nPeople;           // number of people in database
   std::vector<std::string> m_Names;            // names of people in database
   std::vector<std::string> m_OriginalImages;   // names of original images in database that were used for training

   int                     m_nEigenVals;        // the number of eigen values stored in database
   CvMat*                  m_PersonIDMatrix;    // matrix to store person ids
   CvMat*                  m_EigenValueMatrix;  // matrix to store Eigen values
   CvMat*                  m_ProjectedFaceMatrix; // matrix to store projected faces
   IplImage*               m_AverageImage;      // the average trained face
   IplImage**              m_EigenVectorArray;  // array to store eigen vectors

   double                  m_EuclideanThreshold;
   double                  m_MahalanobisThreshold;

   const char*             m_SearchImageName;   // name if search image
   IplImage*               m_FaceImage;         // original image with faces to find, could be more than one
   IplImage**              m_FacesToFind;       // array of Faces to find
   int                     m_nFacesToFind;

   // results
   int 			   m_IDFound;
   double                  m_DistanceFound;
   std::string             m_PersonFound;     // if we din't find a person it remains ""

};




#endif


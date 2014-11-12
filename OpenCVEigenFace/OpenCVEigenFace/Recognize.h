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

   void	      GenResults(std::string& resultsdir);

private:

   /// Not implemented
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
   
   const char*             m_SearchImageName;   // name if search image
   IplImage*               m_FaceImage;         // original image with faces to find, could be more than one
   IplImage**              m_FacesToFind;       // array of Faces to find
   int                     m_nFacesToFind;

   ///////// LDA
   int                               m_nClasses;
   int                               m_nFisherFaces;
   std::map<personIDType, CvMat*>    m_ClassToImageMap;         // each classes projected image array
   std::map<personIDType, int>       m_ClassCountMap;           // number of images in each class
   std::multimap<personIDType, int>  m_ClassToIndexMap;         // stores each class's index into m_EigenVectorArray

   // averages
   std::map<personIDType, CvMat*>    m_ClassToAverageImageMap;  // each classes average projected image
   CvMat*                            m_AverageProjectedImage; 
   
   // EigenVectors and EigenValues for the LDA subspace
   CvMat*                             m_LDAEigenVectors;
   CvMat*                             m_LDAEigenValues;


   // results
   int 			            m_IDFound;
   double                  m_DistanceFound;
   std::string             m_PersonFound;     // if we din't find a person it remains ""

};




#endif


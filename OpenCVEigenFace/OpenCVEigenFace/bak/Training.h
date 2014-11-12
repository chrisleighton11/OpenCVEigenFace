#ifndef TRAINING_H
#define TRAINING_H

/*
   Training.h
   Description:   defines all of the training classes and methods
   Author:        Chris Leighton
   Date:          Feb 16th 2010

*/

#include "Utilities.h"
#include <fstream>
#include <vector>

// structure to store ImageFile contents
struct Image
{
   int            m_ID;
   std::string    m_PersonName;
   std::string    m_ImageName;
   IplImage*      m_Image;

   Image(const char* buffer)
   {
      std::string stuff(buffer);

      // LOOK!!!!! this assumes that the line is space delimited
      // needs improvement if to become production code like some error checking
      
      
      // get the id
      int pos = stuff.find_first_of(' ',0);
      if ( pos == std::string::npos )
         throw std::string("Image: Bad line");
      m_ID = atoi(stuff.substr(0,pos).c_str());

      // get persons name
      int pos2 = stuff.find_first_of(' ',pos+1);
      if ( pos2 == std::string::npos )
         throw std::string("Image: Bad line");
      m_PersonName = stuff.substr(pos+1,pos2-pos-1);

      // get the image name
      m_ImageName = stuff.substr(pos2+1);
   }
};



void Train(const char* imagelist, const char* database, std::string& resultdir);




class Trainer
{
public:
   Trainer(const char* imagelist, const char* database);
   ~Trainer();

   int LoadImages();
   void CreateSubspace();
   void ProjectOntoSubSpace();
   void StoreData();
   void GenResults(std::string& resultsdir);
   void CalculateThresholds();

private:
   std::string             m_ImageFile;      // list of images of faces and thier names
   std::string             m_DatabaseFile;   // where to put the results
      

   int                     m_nImages;        // number of images(faces)
   int                     m_nEigenVals;     // number of eigenvalues and eigenvecors used in subspace
   std::vector<Image>      m_ImageVec;       // pointers to each face
   std::vector<std::string> m_Names;         // names of people

   IplImage**              m_ImageArray;      // array to store images of faces
   IplImage**              m_EigenVectorArray; // array to store eigen vectors
   CvMat*                  m_PersonIDMatrix;   // matrix to store person ids
   CvMat*                  m_EigenValueMatrix; // matrix to store Eigen values
   CvMat*                  m_ProjectedFaceMatrix; // matrix to store projected faces 

   double                  m_EuclideanThreshold;   		// 1/2 the largest euclidean distance for each projected face
   double                  m_MahalanobisThreshold;	// 1/2 the largest Mahalanobis distance for each projected face

   IplImage*               m_AverageImage;     // Average image of all training images
};





#endif



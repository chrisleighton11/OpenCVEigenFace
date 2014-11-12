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
#include <map>


// structure to store ImageFile contents
struct Image
{
   personIDType            m_ID;
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
   void DoLDA();
   void StoreData();
   void GenResults(std::string& resultsdir);
   void CalculateThresholds();

private:
   void CalcClassAverageImage();
   void CalcAverageImage(CvMat* images, int nImages, CvMat* avgImage);
   void CalcWithinScatterMat();
   void CalcBetweenScatterMat();
   void ProjectOntoLDASubspace();
   
   std::string             m_ImageFile;      // list of images of faces and thier names
   std::string             m_DatabaseFile;   // where to put the results
      

   int                     m_nImages;        // number of images(faces)
   int                     m_Width;          // width of training images, all images should be same size
   int                     m_Height;       
   int                     m_nEigenVals;     // number of eigenvalues and eigenvecors used in subspace
   std::vector<Image>      m_ImageVec;       // pointers to each face
   std::vector<std::string> m_Names;         // names of people

   IplImage**              m_ImageArray;      // array to store images of faces
   IplImage**              m_EigenVectorArray; // array to store eigen vectors
   CvMat*                  m_PersonIDMatrix;   // matrix to store person ids
   CvMat*                  m_EigenValueMatrix; // matrix to store Eigen values
   CvMat*                  m_ProjectedFaceMatrix; // matrix to store projected faces 

   double                  m_EuclideanThreshold;   		// 1/2 the largest euclidean distance for each projected face
   
   IplImage*               m_AverageImage;     // Average image of all training images
   
   
   ///////// LDA additions
   int                               m_nClasses;
   int                               m_nLDAEigens;             // number of eigenvalues and eigenvectors used for LDA
   int                               m_nFisherFaces;           // number of fisherfaces to use
   std::map<personIDType, CvMat*>    m_ClassToImageMap;        // each classes image array
   std::map<personIDType, int>       m_ClassCountMap;          // number of images in each class
   std::multimap<personIDType, int>  m_ClassToImageIndexMap;   // stores each class's index into m_EigenVectorArray

   
   // averages
   std::map<personIDType, CvMat*>    m_ClassToAverageImageMap;  // each classes average image
   CvMat*                            m_AverageProjectedImage;
   
   // scatter matrices
   CvMat*                             m_WithinScatterMat;
   CvMat*                             m_InverseWScatterMat;
   CvMat*                             m_BetweenScatterMat;

   // EigenVectors and EigenValues for the LDA subspace
   CvMat*                             m_LDAEigenVectors;
   CvMat*                             m_LDAEigenValues;

   CvMat*                             m_ProjectedLDAFaceMat; // projected LDA face matrix

   
   
};





#endif



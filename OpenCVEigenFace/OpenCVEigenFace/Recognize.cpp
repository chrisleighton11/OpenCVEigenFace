#include "Recognize.h"
#include "FaceDetector.h"
#include "PreProcess.h"
#include <fstream>
#include "HTMLHelper.h"

/* 
Function:   Recognize
Purpose:    Recognize a face 
Arguments:  1) the image with the face to recognize 2) the trained database
Notes:      Function will return empty string if we don't find the person
Returns:    std::string with persons name we found
Throws:     std::string if it can't open file or create memory
*/
std::string Recognize( const char* image, const char* database, double& distance, std::string& resultsDir )
{
   std::string personFound = "";
   try
   {
      Recognizer r(image, database);
      r.LoadTrainingDatabase();

      // find the person
      std::string foundPerson = "";
      personFound = r.FindFace(0, distance);
   }
   catch (...)
   {
      throw;
   }

   return personFound;
}



/* 
Function:   Recognizer class constructor
Purpose:     
Arguments:  1) the image with the face to recognize 2) the trained database
Notes:      
Throws:     std::string if it can't open file or create memory
*/
Recognizer::Recognizer( const char* image, const char* database ) : m_DatabaseName(database), m_nPeople(0), m_nEigenVals(0), m_SearchImageName(image), m_FaceImage(NULL), m_nFacesToFind(0), m_EuclideanThreshold(0.0),
   m_IDFound(0), m_DistanceFound(0.0), m_PersonFound("")
{
   m_FaceImage = cvLoadImage(image,0);  // give face should be pre-processed
   if ( m_FaceImage )
   {
      m_nFacesToFind = 1;
      m_FacesToFind = (IplImage**)cvAlloc(m_nFacesToFind*sizeof(IplImage*));
      m_FacesToFind[0] = m_FaceImage;
   }
   else
   {
      std::string err;
      err = "Recognizer could not load image: ";
      err += image;
      throw err;
   }
}




/* 
Function:   Recognizer class destructor
Purpose:    clean up memory that Recognizer uses     
Notes:      
*/
Recognizer::~Recognizer()
{
   // release the image with all of the faces
   cvReleaseImage(&m_FaceImage);


   // release eigen vectors
   for ( int i = 0; i < m_nEigenVals; i++ )
   {
      if ( m_EigenVectorArray[i] )
         cvReleaseImage(&m_EigenVectorArray[i]);
   }
}




/* 
Function:   LoadTrainingDatabase
Purpose:    loads to training database that was creating during the training session
Notes:      
Returns:    true if success
throws:     std::string if can't open training data or somthing else goes wrong
*/
bool Recognizer::LoadTrainingDatabase()
{
   bool bRet = false;
   CvFileStorage* database = NULL;

   database = cvOpenFileStorage( m_DatabaseName, 0, CV_STORAGE_READ );

   if ( !database )
   {
      std::string err = "Recognizer::LoadTrainingDatabase could not open database ";
      err += m_DatabaseName;
      throw err;
   }

   m_nImages = cvReadIntByName( database, 0, "nImages", 0 );
   m_nPeople = cvReadIntByName( database, 0, "nPeople", 0 );

   // read person names and original image names
   for ( int i = 0; i < m_nPeople; i++ )
   {
      std::string tempName;
      char varname[256];
      sprintf(varname,"PersonID_%d",i+1);
      tempName = cvReadStringByName( database, 0, varname, 0 );
      m_Names.push_back(tempName);

      sprintf(varname, "ImageID_%d", i);
      tempName = cvReadStringByName( database, 0, varname, 0 );
      m_OriginalImages.push_back(tempName);
   }


   m_nEigenVals = cvReadIntByName( database, 0, "nLDAEigens", 0 );
   m_PersonIDMatrix = (CvMat*)cvReadByName( database, 0, "PersonIDMatrix", 0 );
   m_nClasses = cvReadIntByName( database, 0, "nClasses", 0 );
   m_nFisherFaces = cvReadIntByName( database, 0, "nFisherFaces", 0 );

   for ( int i = 0; i < m_nImages; i++ )
   {
      std::map<personIDType, int>::iterator it;
      personIDType id = m_PersonIDMatrix->data.i[i];
      it = m_ClassCountMap.find(id);

      if ( it == m_ClassCountMap.end() )
      {
         m_ClassCountMap[id] = 1;
      }
      else
      {
         m_ClassCountMap[id]++;
      }

      m_ClassToIndexMap.insert( pair<personIDType, int>(id, i) );
   }


   m_EigenValueMatrix = (CvMat*)cvReadByName( database, 0, "PCAEigenValues", 0 );
   m_ProjectedFaceMatrix = (CvMat*)cvReadByName( database, 0, "ProjectedLDAFaceMat", 0 );


   m_LDAEigenVectors = (CvMat*)cvReadByName( database, 0, "LDAEigenVectors", 0 );
   m_LDAEigenValues = (CvMat*)cvReadByName( database, 0, "LDAEigenValues", 0 );

   m_AverageImage = (IplImage*)cvReadByName( database, 0, "AverageImage", 0 );
   m_AverageProjectedImage = (CvMat*)cvReadByName( database, 0, "AverageProjectedImage", 0 );

   m_EigenVectorArray = (IplImage**)cvAlloc(m_nImages*sizeof(IplImage*));
   for ( int i = 0; i < m_nEigenVals; i++ )
   {
      char var[256];
      sprintf(var ,"PCAEigenVector_%d",i);
      m_EigenVectorArray[i] = (IplImage*)cvReadByName(database, 0, var, 0);
   }


   m_EuclideanThreshold = cvReadRealByName( database, 0, "EuclideanThreshold", 0 );
 
   cvReleaseFileStorage(&database);

   return bRet;
}




/* 
Function:   FindFace
Purpose:    attempts to find a face in the database
Notes:      this function uses distance to determine how 
confident we are with the closest face, the threshold value can be adjusted
to try to prevent false positive results
Returns:    name of person, if it finds it, empty if it does not find the face
throws:     
*/
std::string Recognizer::FindFace( int faceNum, double& distance )
{
   std::string personName = "";

   // project the test face onto
   // the PCA subspace so try to find a match
   if ( faceNum < 0 || faceNum >= m_nFacesToFind )
      throw std::string("Recognizer::FindFace - Invalid face number argument");

   // first project the face to find onto the PCA subspace
   float * tempFaceToFind = NULL;
   tempFaceToFind = (float *)cvAlloc( (m_nImages-1)*sizeof(float) );
   cvEigenDecomposite(m_FacesToFind[0], m_nEigenVals, m_EigenVectorArray, 0, 0, m_AverageImage, tempFaceToFind);
   


   // multiply Fisherfaces and Eigenfaces
   // m_LDAEigenVectors is nClass rows and m_nEigenVals cols
   // m_PCAEigenVectors is m_nEigenVal rows and m*n (image size) cols

   // store m_PCAEigenVectors in a matrix
   int size = m_EigenVectorArray[0]->width * m_EigenVectorArray[0]->height;
   CvMat* PCAEigenVectors = cvCreateMat( m_nEigenVals, size, CV_32FC1);
   for ( int row = 0; row < m_nEigenVals; row++ )
   {
      float* temp = new float[size];
      ImageToMatrixf(m_EigenVectorArray[row],  temp, size);
      for ( int col = 0; col < size; col++ )
      {
         PCAEigenVectors->data.fl[row*m_EigenVectorArray[0]->width + col] = temp[col];
      }
      delete [] temp;
   }

  
   // multiply m_LDAEigenVectors by PCAEigenVecotors with 
   CvMat* res = cvCreateMat(m_nClasses-1, size, CV_32FC1);
   cvMatMul( m_LDAEigenVectors, PCAEigenVectors, res);
   
   // res is now m_nClasses-1 rows and size cols
   // multiply it by the probe image face which is 1 row and size cols
   // actually transpose the probe image to be size rows and 1 col
   // the resulting matrix will be m_nClasses-1 rows and 1 col
   // the transpose of this matrix is the projected probe image we will use to compare distances

   // Store original image as matrix with size rows and 1 col
   CvMat* probe = cvCreateMat(size, 1, CV_32FC1);
   ImageToMatrix(m_FacesToFind[faceNum], probe->data.fl, size);

   /*
   for ( int i = 0; i < size; i++ )
      cout << probe->data.fl[i] << " ";

   */

   // Store m_AverageImage in matrix
   CvMat* avgImg = cvCreateMat(size, 1, CV_32FC1);
   ImageToMatrixf(m_AverageImage, avgImg->data.fl, size);

   // center probe
   cvSub(probe, avgImg, probe);

   // multiply res by the probe
   CvMat* ProjectedProbe = cvCreateMat(1, m_nClasses-1, CV_32FC1);
   CvMat* ProjectedProbe_t = cvCreateMat(m_nClasses-1, 1, CV_32FC1);
   cvMatMul(res,probe,ProjectedProbe_t);
   cvTranspose(ProjectedProbe_t, ProjectedProbe);


   // now we can find the least Euclidean Distance comparing the ProjectedProbe 
   // to each m_ProjectedFaceMatrix
   double bestChoiceDiff = DBL_MAX;
   int bestClass = -1;

   int row = 0;
   for ( row = 0 ; row < m_nClasses; row++ )
   {
      double distance = 0.0;

      for ( int col = 0; col < m_nClasses-1; col++ )
      {
         // cout << ProjectedProbe->data.fl[col] << "    " << m_ProjectedFaceMatrix->data.fl[row*m_nClasses+col] << endl;
         float d = ProjectedProbe->data.fl[col] - m_ProjectedFaceMatrix->data.fl[row*m_nClasses+col];
         distance += d*d;
      }

      if ( distance < bestChoiceDiff )
      {
         bestChoiceDiff = distance;
         bestClass = row;
      }
   }


   // find the index from classToIndexMap
   pair<multimap<personIDType, int>::iterator, multimap<personIDType, int>::iterator> eq;
   eq = m_ClassToIndexMap.equal_range(bestClass);

   if ( bestClass != -1 && bestClass <= m_nClasses )
   {
      std::multimap<personIDType,int>::iterator it;
      it = eq.first;
      personName = m_Names[it->second];
   }

   return personName;

   
}





/*
function:	GenResults
Purpose:	Generate html and image results for face search
Notes: 
*/
void Recognizer::GenResults(std::string& resultsDir)
{
   // lets name the results after the image we searched for
   // first I need to get just the image name and remove the directories
   std::string searchImageName(m_SearchImageName);

   // get rid of directory
   int pos = 0;
   pos = searchImageName.find_last_of('/');
   if ( pos != std::string::npos )
      searchImageName = searchImageName.substr(pos+1, searchImageName.size()-pos);

   // now change the extension from .xxx to _xxx
   std::string extension = searchImageName.substr(searchImageName.size()-3, 3);
   std::string saveImageName = resultsDir + searchImageName;   // to write image to new directory
   searchImageName = searchImageName.substr(0, searchImageName.size()-4);
   searchImageName += "_";
   searchImageName += extension;

   std::string resultsname = resultsDir + searchImageName;

   std::string htmlname = resultsname + ".html";

   std::ofstream html(htmlname.c_str());

   if ( !html.is_open() )
   {
      std::string err;
      err = "GenResults can not open results file ";
      err += htmlname;
      throw err;
   }

   html << GetHeader();
   html << GetTitle(searchImageName.c_str());
   std::string title;
   title = "Search Results for ";
   title += searchImageName;
   html << GetText(title.c_str(),"h1", "    ");


   // IE and firefox will not show some image formats.. e.g .pgm
   // I'll make it .jpg.  openCV automatically saves in the format based on
   // file extension.
   saveImageName = saveImageName.substr(0, saveImageName.size()-4); // chop extension
   saveImageName += ".jpg"; // add new extension

   cvSaveImage(saveImageName.c_str(), m_FaceImage);
   html << GetText("Search Face", "h2", "    ");
   std::string tempname = ""; // cut out directory, assume all files in same location
   pos = saveImageName.find_last_of('/');
   if ( pos != std::string::npos )
      tempname = saveImageName.substr(pos+1, saveImageName.size()-pos);
   else
      tempname = saveImageName;

   html << GetImageTag(tempname.c_str(), "200", "200", "    ");

   if ( m_IDFound )
   {
      std::stringstream s1;
      s1 << "Search Found Person ID: " << m_IDFound;
      html << GetText(s1.str().c_str(), "h3", "    ");

      std::stringstream s2;
      s2 << "PersonFound: " << m_PersonFound;
      html << GetText(s2.str().c_str(), "h3", "    ");

      std::stringstream s3;
      s3 << "Distance: " << m_DistanceFound;
      html << GetText(s3.str().c_str(), "h3", "    ");

      html << GetText("Images in database with this persons face:", "h3", "    ");

      // save original images to load into html
      for ( int i = 0; i < m_nImages; i++ )
      {
         if ( m_PersonIDMatrix->data.i[i] == m_IDFound )
         {
            std::string original = m_OriginalImages[i];
            // load the original
            IplImage* image = cvLoadImage(original.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            if ( image )
            {
               // create new name
               pos = 0;
               pos = original.find_last_of('/');  // cut off directory
               if ( pos != std::string::npos )
                  original = original.substr(pos+1, original.size()-pos);

               original = original.substr(0, original.size()-4);
               original += ".jpg"; // change type

               // insert personfound name - if there is one
               if ( !m_PersonFound.empty() )
               {
                  original.insert(0, "_");
                  original.insert(0, m_PersonFound);
               }

               std::string saveimage = resultsDir + original;

               // save to disk
               cvSaveImage(saveimage.c_str(), image);
               cvReleaseImage(&image);

               std::stringstream htmlname;
               htmlname << original << " database index: " << i;
               html << GetText(htmlname.str().c_str(), "h3", "    ");
               html << GetImageTag(original.c_str(), "200", "200", "    " );
            }
         }
      }

   }
   else
   {
      html << GetText("Search failed to find match", "h3", "    ");

      std::stringstream s1;
      s1 << "Distance: " << m_DistanceFound;
      html << GetText(s1.str().c_str(), "h3", "    ");
   }


   html << GetClosingTags();
   html.close();
   
}



/*   
function:	BetweenClassThreshold
Purpose:	calculate threshold based on distance for each projected image in a specific class
Notes:          populates e_threshold and m_threshold with values
returns:        
*/
void Recognizer::BetweenClassThreshold( int personID, double& e_threshold, double& m_threshold )
{
   int nImagesInClass = 0;
   int *class_indexes = NULL;
   class_indexes = new int [m_nImages];

   // find all indexes with personID
   for ( int col = 0; col < m_nImages; col++ )
   {
      if ( personID == m_PersonIDMatrix->data.i[col] )
      {
         class_indexes[nImagesInClass++] = col;
      }
   }

   if ( !nImagesInClass )
      throw std::string("BetweenClassThreshold could not find any images with class");

   // what is the max distance between any image in this class
   double maxE = 0.0;
   double maxM = 0.0;
   int index = 0;
   while ( index <= nImagesInClass )
   {
      for ( int row = class_indexes[index]; row < nImagesInClass; row++ )
      {
         double e_distance = 0.0;
         double m_distance = 0.0;

         for ( int col = 0; col < m_nEigenVals; col++ )
         {
            double d = m_ProjectedFaceMatrix->data.fl[index*m_nEigenVals + col]  ;
            // TODO
         }
      }
   }

   e_threshold = maxE;
   m_threshold = maxM;

   delete class_indexes;
}

#include "Training.h"
#include "PreProcess.h"
#include <fstream>
#include "HTMLHelper.h"



/* 
Function:   Train
Purpose:    Calls into the Trainer to train the system and load the database
Notes:      
Throws      
*/
void Train(const char* imagelist, const char* database, std::string& resultdir)
{
   try
   {
      Trainer trn(imagelist,database);
      trn.LoadImages();
      trn.CreateSubspace();
      trn.ProjectOntoSubSpace();
      trn.DoLDA();
      trn.CalculateThresholds();
      trn.StoreData();

   }
   catch (...)
   {
      throw;
   }


}




////////////////////////////////////////////
//           Trainer class                //
////////////////////////////////////////////


/* 
Function:   Trainer constructor
Purpose:    Constructor for the trainer
Notes:      
Throws      
*/
Trainer::Trainer(const char* imagelist, const char* database) : m_nImages(0), m_Width(0), m_Height(0), m_nEigenVals(0), m_AverageImage(NULL), m_EuclideanThreshold(0.0),
   m_nLDAEigens(0), m_nClasses(0)
{
   m_ImageFile = imagelist;
   m_DatabaseFile = database;

}



/* 
Function:   Trainer destructor
Purpose:    destructor for the trainer, cleans up all of the memory
Notes:      
Throws      
*/
Trainer::~Trainer()
{
   for ( int i = 0; i < m_ImageVec.size(); i++ )
   {
      cvReleaseImage(&m_ImageVec[i].m_Image);
   }
}



/* 
Function:   LoadImages
Purpose:    reads in m_ImageFile and loads the images
Notes:      LoadImages will pre-process the images
Throws      std::string if file can not be opened, or if image can not be found
returns:    Number if images processed
*/
int Trainer::LoadImages()
{
   m_nImages = 0;

   // open the iamges file
   std::ifstream in(m_ImageFile.c_str());

   if ( !in.is_open() )
   {
      std::string err;
      err = "Trainer could not open images file ";
      err += m_ImageFile;
      throw err;
   }

   // buffer to read each line of file
   char buffer[512];

   while ( in.getline(buffer,512) )
   {
      try
      {
         std::string line(buffer);
         if ( line.empty() )
            break;

         Image img(buffer);

         if ( img.m_ID == 0 )
            throw std::string("Trainer::LoadImages - Training person ids should start with 1");

         m_Names.push_back(img.m_PersonName);

         // load image
         IplImage* temp = NULL;
         temp = cvLoadImage(img.m_ImageName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);  // assume image is greyscale since it has been preprocessed 

         if ( !temp )
         {
            std::string err;
            err = "Trainer::LoadImages could not create image for ";
            err += img.m_ImageName;
            throw err;
         }

         IplImage* newtemp = NULL;

         if ( m_Width == 0 )
         {
            m_Width = temp->width;
            m_Height = temp->height;
         }
         else
         {
            if ( m_Width != temp->width || m_Height != temp->height )
               throw std::string("Trainer::LoadImages: Images should be same size");
         }

         img.m_Image = temp;

         m_ImageVec.push_back(img);

         ////////////////////
         // 
         std::map<personIDType, int>::iterator it;
         it = m_ClassCountMap.find(img.m_ID);

         if ( it == m_ClassCountMap.end() )
         {
            m_nClasses++;
            m_ClassCountMap[img.m_ID] = 1;
         }
         else
         {
            m_ClassCountMap[img.m_ID]++;
         }


      }
      catch (...)
      {
         throw;
      }

      // success
      m_nImages++;
   }

   in.close();


   // now store images and person id's in array to pass to eigen functions
   m_ImageArray = (IplImage**)cvAlloc(m_nImages*sizeof(IplImage*));
   m_PersonIDMatrix = cvCreateMat(1,m_nImages,CV_32SC1);   
   for ( int i = 0; i < m_nImages; i++ )
   {
      m_PersonIDMatrix->data.i[i] = m_ImageVec[i].m_ID;

      m_ImageArray[i] = m_ImageVec[i].m_Image;      

      // insert index of this classes image
      m_ClassToImageIndexMap.insert( pair<personIDType, int>(m_ImageVec[i].m_ID, i) );
   }

   return m_nImages;
}



/* 
Function:   CreateSubspace
Purpose:    finds average image, centers each image around mean, finds covariance matrix, then finds 
Eigenvectors (Principal components) and Eigenvalues
Notes:      
Throws      std::string if it can't allocate memory
returns:    
*/
void Trainer::CreateSubspace()
{
   // we can only find m_nImages - m_nClasses for LDA
   m_nEigenVals = m_nImages - m_nClasses;

   CvSize size;
   size.width = m_ImageVec[0].m_Image->width;
   size.height = m_ImageVec[0].m_Image->height;

   // allocate space for the eigen vectors
   m_EigenVectorArray    = (IplImage**)cvAlloc(sizeof(IplImage*) * m_nEigenVals);
   for ( int i = 0; i < m_nEigenVals; i++ )
   {
      m_EigenVectorArray[i] = cvCreateImage(size, IPL_DEPTH_32F, 1);   // floating point image
      
      if ( !m_EigenVectorArray[i] )
         throw std::string("Trainer::DoPCA could not allocate EigenVector");
   }

   m_AverageImage = cvCreateImage(size, IPL_DEPTH_32F, 1 );
   if ( !m_AverageImage )
      throw std::string("Trainer::DoPCA could not allocate m_AverageImage");

   // This is how many eigen vectors we will use out of all of them
   // i.e in the subspace, each image will have nEigenVals 
   CvTermCriteria limit = cvTermCriteria(CV_TERMCRIT_ITER, m_nEigenVals, 1);

   // these will be the actuall eigen values
   m_EigenValueMatrix = cvCreateMat(1, m_nEigenVals, CV_32FC1);

   // ask openCv to do the work
   cvCalcEigenObjects( m_nImages, (void*)m_ImageArray, (void*)m_EigenVectorArray, CV_EIGOBJ_NO_CALLBACK, 0, 0, &limit,
      m_AverageImage, m_EigenValueMatrix->data.fl );

   // now we have the averge image, eigenvectors of the covariance matrix, and eigen values
   cvNormalize(m_EigenValueMatrix, m_EigenValueMatrix, 1, 0, CV_L1, 0);

}




/* 
Function:   ProjectOntoSubspace
Purpose:    projects the faces onto the PCA subspace
Notes:      
Throws      
returns:    
*/
void Trainer::ProjectOntoSubSpace()
{
   m_ProjectedFaceMatrix = cvCreateMat(m_nImages, m_nEigenVals, CV_32FC1);

   // calculate each images projection onto the eigen subspace 
   // m_ProjectedFaceMatrix->data.fl + (row*nEigenVals) gets us to the row containing the images
   // projection
   // m_ProjectedFaceMatrix is a [m_nImages][nEigenVals] matrix

   // to avoid getting a bunch of NaN values, I normalize the Eigenvalues to be between 0 and 1
   cvNormalize(m_EigenValueMatrix, m_EigenValueMatrix, 1, 0, CV_L1, 0);

   for ( int row = 0; row < m_nImages; row++ )
   {
      cvEigenDecomposite(m_ImageArray[row], m_nEigenVals, m_EigenVectorArray, 0, 0, m_AverageImage, 
         m_ProjectedFaceMatrix->data.fl + (row*m_nEigenVals));  
   }

   // now the training projection is completed, Each row of m_ProjectedFaceMatrix represents
   // each image's values projected onto the new subspace.
   // in other words, where each image used to be a NxM matrix, it is now only nEigenVals long
   // Thus, we have reduced dimentionality... Yeah

}



/* 
Function:   DoLDA
Purpose:    Does all LDA related functions
Notes:      
Throws      std::string if somthing goes wrong  
returns:    
*/
void Trainer::DoLDA()
{
   m_nLDAEigens = m_nEigenVals;
   m_nFisherFaces = m_nClasses - 1;

   if ( m_nFisherFaces < 2 )
      throw std::string("Trainer::DoLDA - number of classes needs to be more than 2" );

      // 1. get average of m_ProjectedFaceMatrix
   m_AverageProjectedImage = cvCreateMat(1, m_nEigenVals, CV_32FC1);
   CalcAverageImage( m_ProjectedFaceMatrix, m_nImages, m_AverageProjectedImage );

   // 2. break m_ProjectedFaceMatrix into m_ClassToImageMap ( index of row of m_ProjectedFaceMatrix = m_ClassToIndexMap )
   
   std::map<personIDType, int>::iterator classIt;
   for ( classIt = m_ClassCountMap.begin(); classIt != m_ClassCountMap.end(); classIt++ )
   {
      pair<multimap<personIDType, int>::iterator, multimap<personIDType, int>::iterator> eq;
      eq = m_ClassToImageIndexMap.equal_range(classIt->first);

      m_ClassToImageMap[classIt->first] = cvCreateMat( m_ClassCountMap[classIt->first], m_nEigenVals, CV_32FC1 ) ;

      std::multimap<personIDType,int>::iterator it;
      
      int row_c = 0;
      for ( it = eq.first; it != eq.second; it++ )
      {
         int cls = (*it).first;
         int index = (*it).second;
         for ( int col = 0; col < m_nEigenVals; col++ )
         {
            m_ClassToImageMap[cls]->data.fl[row_c*m_nEigenVals + col] = m_ProjectedFaceMatrix->data.fl[index*m_nEigenVals + col];
         }
         row_c++;
      }
   }

   CalcClassAverageImage();
   CalcWithinScatterMat();
   CalcBetweenScatterMat();
   ProjectOntoLDASubspace();
 
}


/* 
Function:   CalcClassAverageImage
Purpose:    calculates average image for each class
Notes:      
Throws      
returns:    void
*/
void Trainer::CalcClassAverageImage()
{
   // foreach class, calculate average image
   std::map<personIDType, int>::iterator it;
   for ( it = m_ClassCountMap.begin(); it != m_ClassCountMap.end(); it++ )
   {
      int cls = it->first;
      int nImages = it->second;
      
      m_ClassToAverageImageMap[cls] = cvCreateMat(1, m_nEigenVals, CV_32FC1); 

      CalcAverageImage(m_ClassToImageMap[cls], nImages, m_ClassToAverageImageMap[cls]);
   }

}


/* 
Function:   ClassAverageImage
Purpose:    calculates average image for given 
Notes:      
Throws      
returns:    void
*/
void Trainer::CalcAverageImage(CvMat* images, int nImages, CvMat* avgImage)
{
   // first clear out avgImage
  for ( int col = 0; col < avgImage->width; col++ )
    avgImage->data.fl[col] = 0;
  
   for ( int row = 0; row < nImages; row++ )
   {
      for ( int col = 0; col < images->width; col++ )
      {
         avgImage->data.fl[col] += images->data.fl[row*images->width+col];
      }
   }

   for ( int i = 0; i < avgImage->width; i++ )
      avgImage->data.fl[i] /= nImages;

}



/* 
Function:   ClassWithinScatterMat
Purpose:    For LDA implementation
Notes:      
Throws      
returns:    void
*/

void Trainer::CalcWithinScatterMat()
{
   m_WithinScatterMat = cvCreateMat( m_nLDAEigens, m_nLDAEigens, CV_32FC1 );
   m_InverseWScatterMat = cvCreateMat( m_nLDAEigens, m_nLDAEigens, CV_32FC1 );

   for ( int row = 0; row < m_nLDAEigens; row++ )
   {
      for ( int col = 0; col < m_nLDAEigens; col++ )
      {
         m_WithinScatterMat->data.fl[row*m_nLDAEigens+col] = 0.0;
      }
   }

   // for each class, loop through and calculate scatter matrix
   // LOOK: class ID's start with one and don't nessasarily have to be continuous, 
   // I am going to store each classes scatter in on matrix
   // So I will just keep track of it.
   // the class id in m_ClassToAverageImageMap maps to the matrix

   std::map<personIDType, CvMat*>::iterator classNumIt;
   for ( classNumIt = m_ClassToImageMap.begin(); classNumIt != m_ClassToImageMap.end(); classNumIt++ )
   {
      CvMat* temp             = cvCreateMat( 1, m_nLDAEigens, CV_32FC1 );
      CvMat* temp_transposed  = cvCreateMat( m_nLDAEigens, 1, CV_32FC1 );
      CvMat* res              = cvCreateMat( m_nLDAEigens, m_nLDAEigens, CV_32FC1 );
      CvMat* tempclass        = cvCreateMat( m_nLDAEigens, m_nLDAEigens, CV_32FC1 );

      for ( int row = 0; row < m_nLDAEigens; row++ )
      {
         temp->data.fl[row] = temp_transposed->data.fl[row] = 0.0;
         
         for ( int col = 0; col < m_nLDAEigens; col++ )
         {
            res->data.fl[row*m_nLDAEigens+col] = 0.0;
            tempclass->data.fl[row*m_nLDAEigens+col] = 0.0;
         }
      }

      // use m_ClassToImageIndex to find indices of this classes projected images in m_ProjectedFaceMatrix      
      pair<multimap<personIDType, int>::iterator, multimap<personIDType, int>::iterator> eq;
      eq = m_ClassToImageIndexMap.equal_range(classNumIt->first);
      std::multimap<personIDType,int>::iterator it;
      
      for ( it = eq.first; it != eq.second; it++ )
      {
         // subtract the class mean from the projected values for this class's image
         // store values in temp
         CvMat* projectedImage = cvCreateMat( 1, m_nEigenVals, CV_32FC1 );
         for ( int col = 0; col < m_nEigenVals; col++ )
            projectedImage->data.fl[col] = m_ProjectedFaceMatrix->data.fl[it->second*m_nEigenVals+col];
         
         cvSub( projectedImage, m_ClassToAverageImageMap[it->first], temp );

         cvTranspose(temp, temp_transposed);

         // multiply temp_transposed * temp_transposed 
         // res=(temp_transposed)*(temp_transposed)^T
         cvMulTransposed(temp_transposed, res, 0);

         // add this classes result to the tempclass matrix, store result in tempclass
         cvAdd(res, tempclass, tempclass);
      }
      
      // divide all values by the number of images in this class
      double nImages = (double)m_ClassCountMap[classNumIt->first];
      cvDiv( NULL, tempclass, tempclass, nImages );

      cvAdd(m_WithinScatterMat, tempclass, m_WithinScatterMat);
   }

   // find inverse of m_WithinScatterMat, store in m_InverseWScatterMat
   cvInvert( m_WithinScatterMat, m_InverseWScatterMat );
}


/* 
Function:   ClassBetweenScatterMat
Purpose:    For LDA implementation
Notes:      
Throws      
returns:    void
*/
void Trainer::CalcBetweenScatterMat()
{
   m_BetweenScatterMat = cvCreateMat( m_nLDAEigens, m_nLDAEigens, CV_32FC1 );

   for ( int row = 0; row < m_nLDAEigens; row++ )
   {
      for ( int col = 0; col < m_nLDAEigens; col++ )
      {
         m_BetweenScatterMat->data.fl[row*m_nLDAEigens+col] = 0.0;
      }
   }

   std::map<personIDType, CvMat*>::iterator classNumIt;
   for ( classNumIt = m_ClassToImageMap.begin(); classNumIt != m_ClassToImageMap.end(); classNumIt++ )
   {
      CvMat *temp = cvCreateMat( 1, m_nLDAEigens, CV_32FC1 );
      CvMat *res = cvCreateMat( m_nLDAEigens, m_nLDAEigens, CV_32FC1 );

      for ( int row = 0; row < m_nLDAEigens; row++ )
      {
         temp->data.fl[row] = 0.0;

         for ( int col = 0; col < m_nLDAEigens; col++ )
         {
            res->data.fl[row*m_nLDAEigens+col]=0.0;
         }
      }

      // subtract class mean by total mean, put in temp
      cvSub( m_ClassToAverageImageMap[classNumIt->first], m_AverageProjectedImage, temp );

      // multiply temp * temp^t and put into res
      cvMulTransposed( temp, res, 1 );

      // divide res by class number scalar
      cvDiv( NULL, res, res, m_ClassCountMap[classNumIt->first] );

      // sum all of it up
      cvAdd( res, m_BetweenScatterMat, m_BetweenScatterMat );

   }
}



/* 
Function:   ProjectOntoLDASubspace
Purpose:    For LDA implementation
Notes:      
Throws      
returns:    void
*/
void Trainer::ProjectOntoLDASubspace()
{
   m_LDAEigenVectors = cvCreateMat( m_nFisherFaces, m_nLDAEigens, CV_32FC1 );
   m_LDAEigenValues = cvCreateMat( 1, m_nFisherFaces,  CV_32FC1 );

   // temp matrices to get all eigenVectors and eigenValeus for m_nLDAEigens
   CvMat* tempVec = cvCreateMat( m_nLDAEigens, m_nLDAEigens, CV_32FC1 );
   CvMat* tempVal = cvCreateMat( 1, m_nLDAEigens, CV_32FC1 );

   for ( int row = 0; row < m_nLDAEigens; row++ )
   {
      tempVal->data.fl[row] = 0.0;  // really column
      for ( int col = 0; col < m_nLDAEigens; col++ )
      {
         tempVec->data.fl[row*m_nLDAEigens+col] = 0.0;
      }
   }

   for ( int row = 0; row < m_nFisherFaces; row ++ )
   {
      m_LDAEigenValues->data.fl[row] = 0.0;
      for ( int col = 0; col < m_nFisherFaces; col++ )
      {
         m_LDAEigenVectors->data.fl[row*m_nFisherFaces+col] = 0.0;
      }
   }

   // multiply inverse within scatter mat * within scatter mat, store in temp
   CvMat* temp = cvCreateMat( m_nLDAEigens, m_nLDAEigens, CV_32FC1 );
   cvMul( m_InverseWScatterMat, m_WithinScatterMat, temp );

   // get the eigenvectors and eigenvalues
   // CV_SVD_U_T - transposed tempVec will be returned
   // CV_SVD_MODIFY_A - optimization, means function can modify temp during calculation
   cvSVD( temp, tempVal, tempVec, NULL, CV_SVD_U_T + CV_SVD_MODIFY_A );

   // only keep up to the number m_nFisherFaces
   for ( int row = 0; row < m_nFisherFaces; row++ )
   {
      m_LDAEigenValues->data.fl[row] = tempVal->data.fl[row];

      for ( int col = 0; col < m_nLDAEigens; col++ )
      {
         m_LDAEigenVectors->data.fl[row*m_nLDAEigens+col] = tempVec->data.fl[row*m_nLDAEigens+col];
      }
   }

   // now calculate projected images
   // each class has a projection
   m_ProjectedLDAFaceMat = cvCreateMat( m_nClasses, m_nFisherFaces, CV_32FC1 );

   std::map<personIDType, CvMat*>::iterator classNumIt;
   int row = 0;
   for ( classNumIt = m_ClassToImageMap.begin(); classNumIt != m_ClassToImageMap.end(); classNumIt++ )
   {
      // temp matrices for calculations
      CvMat* PCATemp = cvCreateMat( m_nLDAEigens, 1, CV_32FC1 );
      CvMat* LDATemp = cvCreateMat( m_nFisherFaces, 1, CV_32FC1 );
      CvMat* projection = cvCreateMat( 1, m_nFisherFaces, CV_32FC1 );

      // transpose this classes mean projection
      cvTranspose( m_ClassToAverageImageMap[classNumIt->first], PCATemp );

      // multiply the eigenvectors to the transposed mean, store in LDATemp
      cvMatMul( m_LDAEigenVectors, PCATemp, LDATemp );

      // tranpose and store in projection
      cvTranspose( LDATemp, projection );

      // copy projection to m_ProjectedLDAFaceMat for perminiant storage
      for ( int col = 0; col < m_nFisherFaces; col++ )
      {
         m_ProjectedLDAFaceMat->data.fl[row*m_nFisherFaces+col] = projection->data.fl[col];
      }
      row++;
   }
}



/* 
Function:   CalculateThresholds
Purpose:    calculates thresholds of database used for comparing a probe image
Notes:      
Throws      
returns:    void
*/
void Trainer::CalculateThresholds()
{
   double maxE = 0.0;
   int index = 0;

   while ( index <= m_nClasses )
   {

      for ( int row = index; row < m_nClasses; row++ )
      {
         double e_distance = 0.0;
         double m_distance = 0.0;

         for ( int col = 0; col < m_nFisherFaces; col++ )
         {
            double d = m_ProjectedLDAFaceMat->data.fl[index*m_nClasses +col] - m_ProjectedLDAFaceMat->data.fl[row*m_nClasses + col];

            double dd = d*d; 
            e_distance += dd;
         }

         if ( e_distance > maxE )
            maxE = e_distance;
      }

      index++;
   }
   m_EuclideanThreshold = maxE * .5;
 
}



/* 
Function:   StoreData
Purpose:    writes data from training session to disk
Notes:      
Throws      std::string if it can't open training database
returns:    void
*/

void Trainer::StoreData()
{

   CvFileStorage* database;
   database = cvOpenFileStorage(m_DatabaseFile.c_str(), 0, CV_STORAGE_WRITE);

   if ( !database )
      throw std::string("Trainer::StorData could not open database");

   cvWriteInt( database, "nImages", m_nImages );
   cvWriteInt( database, "nPeople", m_Names.size() );

   // store names
   for ( int i = 0; i < m_Names.size(); i++ )
   {
      char var[256];
      sprintf(var,"PersonID_%d", (i+1));
      cvWriteString( database, var, m_Names[i].c_str(), 0 );
   }

   // store database images (in order)
   for ( int i = 0; i < m_ImageVec.size(); i++ )
   {
      char var[256];
      sprintf(var, "ImageID_%d", i);
      cvWriteString( database, var, m_ImageVec[i].m_ImageName.c_str(), 0 );
   }

   cvWriteInt( database, "nLDAEigens", m_nLDAEigens );
   cvWriteInt( database, "nClasses", m_nClasses );
   cvWriteInt( database, "nFisherFaces" , m_nFisherFaces );
   cvWrite( database, "PersonIDMatrix", m_PersonIDMatrix, cvAttrList(0,0) );
   
   for ( int i = 0; i < m_nLDAEigens; i++ )
   {
      char var[256];
      sprintf(var, "PCAEigenVector_%d", i);
      cvWrite( database, var, m_EigenVectorArray[i], cvAttrList(0,0) );
   }
   
   cvWrite( database, "PCAEigenValues", m_EigenValueMatrix, cvAttrList(0,0) );
   cvWrite( database, "LDAEigenVectors", m_LDAEigenVectors, cvAttrList(0,0) );
   cvWrite( database, "LDAEigenValues", m_LDAEigenValues, cvAttrList(0,0) );
   cvWrite( database, "ProjectedLDAFaceMat", m_ProjectedLDAFaceMat, cvAttrList(0,0) );
   cvWrite( database, "AverageImage", m_AverageImage, cvAttrList(0,0) );
   cvWrite( database, "AverageProjectedImage", m_AverageProjectedImage, cvAttrList(0,0) );

   // write each class ID out
   std::map<personIDType, int>::iterator it;
   int i = 0;
   for ( it = m_ClassCountMap.begin(); it != m_ClassCountMap.end(); it++ )
   {
      char var[256];
      sprintf(var, "Class_%d", i++ );
      cvWriteInt( database, var, it->first );
   }

   // write each classes average image
   std::map<personIDType, CvMat*>::iterator ImageIt;
   for ( ImageIt = m_ClassToAverageImageMap.begin(); ImageIt != m_ClassToAverageImageMap.end(); ImageIt++ )
   {
      char var[256];
      sprintf(var, "ClassAverageImage_ID%d", ImageIt->first);
      cvWrite( database, var, ImageIt->second);
   }

   // store threshold values
   cvWriteReal( database, "EuclideanThreshold", m_EuclideanThreshold );

   cvReleaseFileStorage( &database );

}



/* 
Function:   GenResults
Purpose:    create images and html representation of results of training session
Notes:      
Throws      std::string if it can't create directory
returns:    void
*/
void Trainer::GenResults(std::string& resultsdir)
{
   // first get the name of the results.  Name it the database name
   std::string resultsname = m_DatabaseFile.substr(0, m_DatabaseFile.size()-4); // chop of .xml

   // get rid of prefix directory since we are storing results in fiven resultsdir
   size_t dirpos = resultsname.find_last_of('/');
   if ( dirpos != std::string::npos )
      resultsname = resultsname.substr(dirpos+1, resultsname.size() - dirpos );

   std::string htmlfile = resultsdir + resultsname + ".html"; // name the reults after the database created
   std::string imageprefix = resultsdir + resultsname;


   std::ofstream html(htmlfile.c_str());
   if ( !html.is_open() )
   {
      std::string err;
      err = "Trainer::GenResults could not open html file ";
      err += htmlfile;
      throw err;
   }

   html << GetHeader();  
   html << GetTitle(resultsname.c_str());

   html << GetText("Training Results", "h1", "    ");

   std::stringstream s1;
   s1 << "Number of Images " << m_nImages;
   html << GetText(s1.str().c_str(), "h2", "    ");

   std::vector<std::string> people;
   for ( int i = 0; i < m_Names.size(); i++ )
   {
      std::vector<std::string>::iterator it;
      if ( (it = find(people.begin(), people.end(), m_Names[i])) == people.end() )
      {
         people.push_back(m_Names[i]);
      }
   }

   std::stringstream s2;
   s2 << "Number of people " << people.size();
   html << GetText(s2.str().c_str(), "h2", "    ");

   html << GetText("Names:", "h2", "    ");
   for ( int i = 0; i < people.size(); i++ )
   {
      html << GetText(people[i].c_str(), "h4", "    ");
      for ( int j = 0; j < m_nImages; j++ )
      {
         if ( m_ImageVec[j].m_PersonName == people[i] )  //hack
         {
            std::string savename = m_ImageVec[j].m_ImageName;
            size_t p = savename.find_last_of('/');
            if ( p != std::string::npos )
               savename = savename.substr(p+1, savename.size()-p);
            p = savename.find_last_of('.');
            if ( p != std::string::npos )
               savename = savename.substr(0, savename.size()-4);
            std::stringstream hname;
            hname << people[i] << "_" << savename << ".jpg";
            std::string sname = resultsdir;
            sname += hname.str();
            cvSaveImage(sname.c_str(), m_ImageVec[j].m_Image);
            html << GetImageTag(hname.str().c_str(), "200", "200", "    ");
         }
      }
   }

   std::stringstream s3;
   s3 << "Number of Eigen Values " << m_nEigenVals;
   html << GetText(s3.str().c_str(), "h2", "    ");

   // create images of average and some of the eigenfaces
   std::string avgname = imageprefix + "_average.jpg";
   cvSaveImage(avgname.c_str(), m_AverageImage);

   html << GetText("Average Image", "h2", "   ");
   html << GetBr("   ");

   // only output name, not entire directory
   // to make this simple I am going to put everything in the same directory
   std::string tempname = "";
   int pos = 0;
   pos = avgname.find_last_of('/');
   if ( pos != std::string::npos )
      tempname = avgname.substr(pos+1, avgname.size()-pos);
   else
      tempname = avgname;
   html << GetImageTag(tempname.c_str(), "200", "200", "   ");

   html << GetBr("    ");

   // now create some of the eigen vectors
   // for now I'll just do five
   html << GetText("Some EigenFaces", "h2", "   ");
   int n = m_nEigenVals;
   for ( int i = 0; i < n && i < 5; i++ )
   {
      // make name of image to save to disk
      std::stringstream s5;
      s5 << imageprefix << "_EigenFace_" << i << ".jpg";
      std::string name = s5.str();
      pos = name.find_last_of('/');
      if ( pos != std::string::npos )
         tempname = name.substr(pos+1, name.size()-pos);

      IplImage* greyimage = ConvertFloatToGreyScale(m_EigenVectorArray[i]);	
      cvSaveImage(name.c_str(), greyimage);
      cvReleaseImage(&greyimage);

      html << GetImageTag(tempname.c_str(), "200", "200", "   ");

      html << GetBr("    ");
   }

   html << GetClosingTags();

   html.close();
}

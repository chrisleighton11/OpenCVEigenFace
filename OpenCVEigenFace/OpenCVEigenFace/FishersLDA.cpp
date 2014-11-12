/*
FishersLDA.cpp
Description:   Driver for detecting faces using Fishers Linear Discriminant Analysis
	            provides comand line user interface to interact with prgram

Author:        Chris Leighton
Date:          Mar 15th 2010

*/

#include <algorithm>
#include <cctype>
#include <string>
#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>
#include "FaceDetector.h"
#include "Utilities.h"
#include "PreProcess.h"
#include "Training.h"
#include "TrainingFile.h"
#include "Recognize.h"
#include "Standardize.h"

void PrintUsage();

int main( int argc, char** argv )
{
	system("clear");
	std::string command = "";

	while (1)
	{
		PrintUsage();
		
		cin >> command;
	
		std::transform( command.begin(), command.end(), command.begin(),(int(*)(int)) std::toupper );

	        try 
		{	

			if ( command == "PREPROCESS" )
			{
				std::string input;
				std::string output;
				cout << "Enter Input image name: ";
				cin >> input;
				cout << "Enter new name of preprocessed image: ";
				cin >> output;
         			if ( DetectAndPreProcess(input.c_str(), output.c_str()) )
            				cout << "Detected face in " << input << ". PreProcessed face saved as " << output << endl;
         			else
     			        	cout << "An error occured attemting to detect a face and PreProcess " << input  << endl;
			}
			else if ( command == "GENFILE" )
			{
				std::string trainingfile;
				std::string basedir;
				cout << "Enter Training File Name:";
				cin >> trainingfile;

				cout << "Enter base directory:";
				cin >> basedir;

				TrainingFile tf( trainingfile );
				tf.SetBaseDir(basedir);
				
				std::string name = "";
				std::string filename = "";

				while (1)
				{
					// get lines from user
					cout << "Enter Person Name (q to quit):";
					cin >> name;
					if ( name == "q" || name == "Q" )
						break;

					cout << "Enter File Name:";
					cin >> filename;
					
					tf.addEntry(name, filename);
					
					name = filename = "";
				}
	
				if ( tf.GenFile() )
				{
					cout << trainingfile << " created." << endl;
				}
				else
				{
					cout << "An error occured" << endl;
				}
			}
			else if ( command == "TRAIN" )
			{
				std::string trainingfile = "";
				std::string outputfile = "";
				std::string resultsdir = "";
				cout << "Enter Training File:";
				cin >> trainingfile;
				cout << "Enter database name:";
				cin >> outputfile;
				cout << "Enter results directory:";
				cin >> resultsdir; 
				
				// make sure results dir ends with '/'
				if ( !resultsdir.empty() &&  resultsdir[resultsdir.size()-1] != '//' ) 
					resultsdir.append("//"); 

				Train( trainingfile.c_str(), outputfile.c_str(), resultsdir );

				cout << "Database created: " << outputfile << endl;

			}
			else if ( command == "SEARCH" )
			{
				std::string imagename = "";
				std::string database = "";
				std::string resultsdir = "";
				cout << "Enter image to search for. Note: Image should be preprocessed:";
				cin >> imagename;
				cout << "Enter trained database file name: ";
				cin >> database;
				cout << "Enter results directory:";
				cin >> resultsdir;

				// make sure results dir ends with '/'
				if ( !resultsdir.empty() &&  resultsdir[resultsdir.size()-1] != '//' )
					resultsdir.append("//");

         			double distance = DBL_MAX;
         			std::string result = Recognize(imagename.c_str(),database.c_str(), distance, resultsdir );
         			if ( result.empty() )
         			{
            				cout << "Could not find person" << endl;
            				cout << "Distance: " << distance << endl;
        			 }
         			else
        		 	{
            				cout << "Found: " << result << endl;
            				cout << "Distance: " << distance << endl;
         			}
			}
         else if ( command == "TEST" )
         {
            int nrows = 3;
            int ncols = 3;
            CvMat* mat = cvCreateMat(nrows, ncols, CV_32SC1);
            int i = 0;
            float* temp = mat->data.fl;
            temp[i++] = 200.0;
            temp[i++] = 123.0;
            temp[i++] = 21.0;
            temp[i++] = 19.0;
            temp[i++] = 23.0;
            temp[i++] = 19.0;
            temp[i++] = 6.0;
            temp[i++] = 12.0;
            temp[i++] = 24.0;
            

            for ( int r = 0; r < nrows; r++ )
            {
               for ( int c = 0; c < ncols; c++ )
               {
                  cout << mat->data.fl[r*ncols+c] << " ";
               }
               cout << endl;
            }

            stdMatSDEV(mat, mat);

            for ( int r = 0; r < nrows; r++ )
            {
               for ( int c = 0; c < ncols; c++ )
               {
                  cout << mat->data.fl[r*ncols+c] << " ";
               }
               cout << endl;
            }

            i = 0;
         }
			else if ( command == "EXIT" )
			{
				cout << "good by" << endl;
				break;
			}

			command = "";
		}
		catch ( std::string err )
		{
			cout << "Error: " << err << endl;
			return 1;
		}
	}

   return 0;
}




void PrintUsage()
{
   cout << "Please select a command:" << endl << endl;
   cout << "preprocess - detect a face and preprocess the image, then store face on disk" << endl;
   cout << "genfile    - create a training file" << endl;
   cout << "train      - train the system" << endl;
   cout << "search     - search the database for a face in an image" << endl;
   cout << "test       - run a test" << endl;
   cout << "exit" << endl << ":";
}


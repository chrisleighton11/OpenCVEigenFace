CC           =  g++
CFLAGS       = -Wall -g

CXXFLAGS    = `pkg-config opencv --cflags`
LDFLAGS     = `pkg-config opencv --libs`

TARGET1 = EigenFace
OBJS1   = FaceDetector.o OpenCVEigenFace.o PreProcess.o Recognize.o Training.o Utilities.o TrainingFile.o HTMLHelper.o

all:	$(TARGET1)

.SUFFIXES: .cpp .o .hpp

$(TARGET1): $(OBJS1)
	$(CC) -o $@ $(OBJS1) $(CXXFLAGS) $(LDFLAGS)

clean:
	/bin/rm -f *.o core $(TARGET1)








#gcc `pkg-config --cflags opencv` `pkg-config --libs opencv` -o test opencvTest.cpp



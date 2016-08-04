CC=g++
CFLAGS=-g -O3 --std=c++1z -Wall $(shell pkg-config opencv --cflags) -DLSD_NODRAW=true -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare
LIBS=$(shell pkg-config opencv --libs) -lboost_system -lboost_filesystem -lyaml-cpp
CERES_LIBS=-DCXX11=ON -lceres -lgflags -lglog -llapack -lcamd -lamd -lccolamd -lcolamd -lcholmod -lcxsparse -fopenmp -lpthread -lgomp -lm
SOURCES=src/tracker/depth.cpp src/loader/tum.cpp src/main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
DEPS=$(SOURCES:.cpp=.d)
EXECUTABLE=lsdslam.out

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LIBS) $(CERES_LIBS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) -MMD -MP $< -o $@

clean:
	$(RM) $(OBJECTS) -f
	$(RM) $(DEPS) -f
	$(RM) $(EXECUTABLE) -f

-include $(DEPS)
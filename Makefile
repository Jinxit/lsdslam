CC=g++
CFLAGS=-g -Og --std=c++1z -Wall $(shell pkg-config opencv --cflags) -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare
LIBS=$(shell pkg-config opencv --libs) -lboost_system -lboost_filesystem -lyaml-cpp
CERES_LIBS=-lceres -lgflags -lglog -llapack -lcamd -lamd -lccolamd -lcolamd -lcholmod -lcxsparse -fopenmp -lpthread -lgomp -lm
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
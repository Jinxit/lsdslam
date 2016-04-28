CC=g++
CFLAGS=-g -Og --std=c++1z -Wall $(shell pkg-config opencv --cflags)
LIBS=$(shell pkg-config opencv --libs) -lboost_system -lboost_filesystem -lyaml-cpp
SOURCES=src/tracker.cpp src/loader/euroc.cpp src/loader/tum.cpp src/main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
DEPS=$(SOURCES:.cpp=.d)
EXECUTABLE=lsdslam.out

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LIBS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) -MMD -MP $< -o $@

clean:
	$(RM) $(OBJECTS) -f
	$(RM) $(DEPS) -f
	$(RM) $(EXECUTABLE) -f

-include $(DEPS)
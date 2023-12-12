CFLAG = -Wall -g -fopenmp -O2 -march=native -fno-strict-aliasing
LIBS = -lrt -ldnnl 
CC = g++
BIN = test_l2sqr
CFILES = l2sqr.cpp 

all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm $(BIN)

.PHONY: clean
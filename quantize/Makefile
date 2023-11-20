CFLAG = -Wall -g -fopenmp -O2 -march=native -fno-strict-aliasing
LIBS = -lrt -ldnnl
CC = g++
BIN = test_quantize 
CFILES =quantize.cpp

all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm -rf $(BIN)

.PHONY: clean
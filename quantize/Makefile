CFLAG = -Wall -g -lrt -ldnnl -fopenmp -O2 -march=native -fno-strict-aliasing
CC = g++
BIN = test_quantize 
CFILES =quantize.cpp

all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm $(BIN)

.PHONY: clean
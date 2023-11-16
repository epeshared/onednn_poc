CFLAG = -Wall -g -lrt -ldnnl -fopenmp -O2 -march=native -fno-strict-aliasing
CC = g++
BIN = test_inner_product
CFILES =inner_product.cpp 

all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm $(BIN)

.PHONY: clean
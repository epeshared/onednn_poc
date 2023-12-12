CFLAG = -Wall -g -fopenmp -O2 -march=native -fno-strict-aliasing
LIBS = -lrt -ldnnl 
CC = g++
BIN = test_cacl_sum_for_row.cpp
CFILES =cacl_sum_for_row.cpp 

all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm $(BIN)

.PHONY: clean
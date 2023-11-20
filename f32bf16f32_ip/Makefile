CFLAG = -Wall -g -fopenmp -O2 -march=native -fno-strict-aliasing
LIBS = -lrt -ldnnl 
CC = g++
BIN = test_f32b16f32
CFILES = f32bf16f32_ip.cpp

all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm $(BIN)

.PHONY: clean
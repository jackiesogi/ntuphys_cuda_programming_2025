CUDIR     := /usr/local/cuda

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVFLAGS := -O3 -I$(CUDIR)/include -arch=compute_89 -code=sm_89 -Xptxas -v

BIN     := vecAdd
LIB     :=

all: $(BIN)

$(BIN): matAdd.cu
	nvcc -o $(BIN) $(NVFLAGS) vecAdd.cu $(LIB)

clean:
	rm -f $(BIN)

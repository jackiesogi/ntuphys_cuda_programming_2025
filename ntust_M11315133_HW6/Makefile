CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v

BIN1    := hist_1_global
BIN2    := hist_1_shared
LIB     :=

all: $(BIN)

$(BIN1): hist_1_global.cu;
	nvcc -o $(BIN1) $(NVFLAGS) hist_1_global.cu $(LIB)

$(BIN2): hist_1_shared.cu;
	nvcc -o $(BIN2) $(NVFLAGS) hist_1_shared.cu $(LIB)

clean:
	rm -f $(BIN)

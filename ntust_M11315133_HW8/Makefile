CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v

BIN1    := cpu
BIN2    := ngpu
LIB     :=

all: $(BIN)

$(BIN1): cpu.cu;
	nvcc -o $(BIN1) $(NVFLAGS) cpu.cu $(LIB)

$(BIN2): ngpu.cu;
	nvcc -o $(BIN2) $(NVFLAGS) ngpu.cu $(LIB)

clean:
	rm -f $(BIN)

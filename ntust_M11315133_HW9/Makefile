CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v

BIN1    := poisson_solver3d
LIB     :=

all: $(BIN)

$(BIN1): poisson_solver3d.cu;
	nvcc -o $(BIN1) $(NVFLAGS) poisson_solver3d.cu $(LIB)

clean:
	rm -f $(BIN)

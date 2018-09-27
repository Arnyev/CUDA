NVCC=/usr/local/cuda/bin/nvcc
BIN_DIR=bin
INCLUDES=-I/usr/local/cuda/include -Iinc
CC=g++
CCFLAGS= -L/usr/local/cuda/lib64 $(INCLUDES) -lpthread -lcuda -lcudart -lcublas -O3 -Wextra -std=c++11
NVCC_FLAGS= -gencode arch=compute_60,code=sm_60 -std=c++11 -O3 $(INCLUDES)


all: $(BIN_DIR)/tmp/device.o  $(BIN_DIR)/prog.o

$(BIN_DIR)/tmp/device.o: DeviceFunctions.cu
	$(NVCC) $(NVCC_FLAGS) -c DeviceFunctions.cu -o $(BIN_DIR)/tmp/device.o

$(BIN_DIR)/prog.o: Main.cpp
	$(CC) -o prog Main.cpp $(BIN_DIR)/tmp/device.o $(CCFLAGS)

clean:
	rm -r $(BIN_DIR)
	mkdir $(BIN_DIR)
	mkdir $(BIN_DIR)/tmp


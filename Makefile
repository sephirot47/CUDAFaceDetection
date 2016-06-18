CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=sm_20 -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

default: bin/sequential.exe bin/one-device.exe bin/four-devices.exe bin/four-devices-pinned.exe

bin/stbi.o: src/stbi.cpp
	$(NVCC) -std=c++11 -c src/stbi.cpp -o bin/stbi.o

bin/sequential.o: src/sequential.cu
	$(NVCC) -std=c++11 -c -o $@ src/sequential.cu $(NVCC_FLAGS)
bin/one-device.o: src/one-device.cu
	$(NVCC) -std=c++11 -c -o $@ src/one-device.cu $(NVCC_FLAGS)
bin/four-devices.o: src/four-devices.cu
	$(NVCC) -std=c++11 -c -o $@ src/four-devices.cu $(NVCC_FLAGS)
bin/four-devices-pinned.o: src/four-devices-pinned.cu
	$(NVCC) -std=c++11 -c -o $@ src/four-devices-pinned.cu $(NVCC_FLAGS)

bin/sequential.exe: bin/stbi.o bin/sequential.o
	$(NVCC) bin/stbi.o bin/sequential.o -o bin/sequential.exe $(LD_FLAGS)
bin/one-device.exe: bin/stbi.o bin/one-device.o
	$(NVCC) bin/stbi.o bin/one-device.o -o bin/one-device.exe $(LD_FLAGS)
bin/four-devices.exe: bin/stbi.o bin/four-devices.o
	$(NVCC) bin/stbi.o bin/four-devices.o -o bin/four-devices.exe $(LD_FLAGS)
bin/four-devices.exe: bin/stbi.o bin/four-devices-pinned.o
	$(NVCC) bin/stbi.o bin/four-devices.o -o bin/four-devices-pinned.exe $(LD_FLAGS)

clean:
	rm -rf *.o bin/*
	
ultraclean:
	rm -rf *.o* *.e* bin/*

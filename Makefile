CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O2 -I$(CUDA_HOME)/include -arch=sm_20 -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

default: bin/face_detect_seq.exe bin/face_detect_1gpu.exe bin/face_detect_4gpu.exe bin/face_detect_1gpu_pin.exe bin/face_detect_4gpu_pin.exe

seq: bin/face_detect_seq.exe

bin/stbi.o: src/stbi.cpp
	$(NVCC) -std=c++11 -c src/stbi.cpp -o bin/stbi.o

bin/face_detect_seq.o: src/face_detect_seq.cpp src/common.h
	g++ -std=c++11 -c -o $@ src/face_detect_seq.cpp
bin/face_detect_1gpu.o: src/face_detect_1gpu.cu src/common.h src/kernels.h
	$(NVCC) -std=c++11 -c -o $@ src/face_detect_1gpu.cu $(NVCC_FLAGS)
bin/face_detect_4gpu.o: src/face_detect_4gpu.cu src/common.h src/kernels.h
	$(NVCC) -std=c++11 -c -o $@ src/face_detect_4gpu.cu $(NVCC_FLAGS)
bin/face_detect_1gpu_pin.o: src/face_detect_1gpu_pin.cu src/common.h src/kernels.h
	$(NVCC) -std=c++11 -c -o $@ src/face_detect_1gpu_pin.cu $(NVCC_FLAGS)
bin/face_detect_4gpu_pin.o: src/face_detect_4gpu_pin.cu src/common.h src/kernels.h
	$(NVCC) -std=c++11 -c -o $@ src/face_detect_4gpu_pin.cu $(NVCC_FLAGS)

bin/face_detect_seq.exe: bin/stbi.o bin/face_detect_seq.o
	g++ -std=c++11 bin/stbi.o bin/face_detect_seq.o -o bin/face_detect_seq.exe
bin/face_detect_1gpu.exe: bin/stbi.o bin/face_detect_1gpu.o
	$(NVCC) bin/stbi.o bin/face_detect_1gpu.o -o bin/face_detect_1gpu.exe $(LD_FLAGS)
bin/face_detect_4gpu.exe: bin/stbi.o bin/face_detect_4gpu.o
	$(NVCC) bin/stbi.o bin/face_detect_4gpu.o -o bin/face_detect_4gpu.exe $(LD_FLAGS)
bin/face_detect_1gpu_pin.exe: bin/stbi.o bin/face_detect_1gpu_pin.o
	$(NVCC) bin/stbi.o bin/face_detect_1gpu_pin.o -o bin/face_detect_1gpu_pin.exe $(LD_FLAGS)
bin/face_detect_4gpu_pin.exe: bin/stbi.o bin/face_detect_4gpu_pin.o
	$(NVCC) bin/stbi.o bin/face_detect_4gpu.o -o bin/face_detect_4gpu_pin.exe $(LD_FLAGS)

clean:
	rm -rf *.o bin/*
	
ultraclean:
	rm -rf *.o* *.e* bin/*

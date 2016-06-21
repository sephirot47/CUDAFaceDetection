CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=sm_20 -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

default: bin/face_detect_seq.exe bin/face_detect_1gpu.exe bin/face_detect_4gpu.exe bin/face_detect_4gpu_pin.exe

seq: bin/face_detect_seq.exe

bin/stbi.o: src/stbi.cpp
	$(NVCC) -std=c++11 -c $< -o $@

bin/face_detect_seq.o: src/face_detect_seq.cpp src/common.h
	g++ -std=c++11 -c -o $@ $<
bin/face_detect_1gpu.o: src/face_detect_1gpu.cu src/common.h src/kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $< $(NVCC_FLAGS)
bin/face_detect_4gpu.o: src/face_detect_4gpu.cu src/common.h src/kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $< $(NVCC_FLAGS)
bin/face_detect_4gpu_pin.o: src/face_detect_4gpu_pin.cu src/common.h src/kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $< $(NVCC_FLAGS)

bin/face_detect_seq.exe: bin/face_detect_seq.o bin/stbi.o
	g++ -std=c++11 $^ -o $@
bin/face_detect_1gpu.exe: bin/face_detect_1gpu.o bin/stbi.o
	$(NVCC) $^ -o $@ $(LD_FLAGS)
bin/face_detect_4gpu.exe: bin/face_detect_4gpu.o bin/stbi.o
	$(NVCC) $^ -o $@ $(LD_FLAGS)
bin/face_detect_4gpu_pin.exe: bin/face_detect_4gpu_pin.o bin/stbi.o
	$(NVCC) $^ -o $@ $(LD_FLAGS)

clean:
	rm -rf *.o bin/*
	
ultraclean:
	rm -rf *.o* *.e* bin/*

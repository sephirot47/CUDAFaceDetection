CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=sm_20 -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

SEQ         = bin/face_detect_seq
1GPU        = bin/face_detect_1gpu
4GPU_V1     = bin/face_detect_4gpu_v1
4GPU_PIN_V1 = bin/face_detect_4gpu_pin_v1
4GPU_V2     = bin/face_detect_4gpu_v2
4GPU_PIN_V2 = bin/face_detect_4gpu_pin_v2

default: $(SEQ).exe $(1GPU).exe $(4GPU_V1).exe $(4GPU_PIN_V1).exe $(4GPU_V2).exe $(4GPU_PIN_V2).exe

seq: $(SEQ).exe

bin/stbi.o: src/stbi.cpp
	$(NVCC) -std=c++11 -c $< -o $@

$(SEQ).o: src/face_detect_seq.cpp src/common.h
	g++ -std=c++11 -c -o $@ $<
$(1GPU).o: src/face_detect_1gpu.cu src/common.h src/kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $< $(NVCC_FLAGS)
$(4GPU_V1).o: src/face_detect_4gpu_v1.cu src/common.h src/kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $< $(NVCC_FLAGS)
$(4GPU_PIN_V1).o: src/face_detect_4gpu_pin_v1.cu src/common.h src/kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $< $(NVCC_FLAGS)
$(4GPU_V2).o: src/face_detect_4gpu_v2.cu src/common.h src/kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $< $(NVCC_FLAGS)
$(4GPU_PIN_V2).o: src/face_detect_4gpu_pin_v2.cu src/common.h src/kernels.cu
	$(NVCC) -std=c++11 -c -o $@ $< $(NVCC_FLAGS)

$(SEQ).exe: $(SEQ).o bin/stbi.o
	g++ -std=c++11 $^ -o $@
$(1GPU).exe: $(1GPU).o bin/stbi.o
	$(NVCC) $^ -o $@ $(LD_FLAGS)
$(4GPU_V1).exe: $(4GPU_V1).o bin/stbi.o
	$(NVCC) $^ -o $@ $(LD_FLAGS)
$(4GPU_PIN_V1).exe: $(4GPU_PIN_V1).o bin/stbi.o
	$(NVCC) $^ -o $@ $(LD_FLAGS)
$(4GPU_V2).exe: $(4GPU_V2).o bin/stbi.o
	$(NVCC) $^ -o $@ $(LD_FLAGS)
$(4GPU_PIN_V2).exe: $(4GPU_PIN_V2).o bin/stbi.o
	$(NVCC) $^ -o $@ $(LD_FLAGS)

clean:
	rm -rf *.o bin/*
	
ultraclean:
	rm -rf *.o* *.e* bin/*

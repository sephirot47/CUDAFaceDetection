CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=sm_20 -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	    = bin/face_detection.exe
OBJ	    = bin/stbi.o bin/face_detection.o

default: $(EXE)


bin/stbi.o: src/stbi.cpp
	$(NVCC) -std=c++11 -c src/stbi.cpp -o bin/stbi.o

bin/face_detection.o: src/face_detection.cu
	$(NVCC) -std=c++11 -c -o $@ src/face_detection.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
	
ultraclean:
	rm -rf *.o* *.e* $(EXE)

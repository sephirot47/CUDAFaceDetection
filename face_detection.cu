#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stbi.h"
#include "stbi_write.h"

typedef unsigned char uc;
using namespace std;

class Pixel
{
public:
  int x,y;

  Pixel() { x = y = 0; }
  Pixel(int _x, int _y) {x=_x; y=_y;}

  string toString() {
    ostringstream ss;
    ss << "(" << x << ",\t" << y << ")";
    return ss.str();

  }
};

class Color
{
public:
  uc r,g,b,a;

  Color(uc _r, uc _g, uc _b, uc _a) { r=_r; g=_g; b=_b; a=_a; }

  string toString() {
    ostringstream ss;
    ss << "(" << r << "," << g << "," << b << "," << a << ")";
    return ss.str();
  }
};

class Image
{
public:
  char* filename;

  Image(char* filename)
  {
    this->filename = filename;

    FILE *file = fopen(filename, "r");
    if (file != NULL)
    {
        data = stbi_load_from_file(file, &_width, &_height, &comp, 4); // rgba
        fclose (file);
        printf("%s read successfully\n", filename);
    }
    else printf("Image '%s' not found\n", filename);
  }

  int width() {
      return _width;
  }

  int height() {
      return _height;
  }

  Color getColor(Pixel p) {
      int offset = (p.y * _width + p.x) * comp;
      Color c((data[offset + 0]),
              (data[offset + 1]),
              (data[offset + 2]),
              (data[offset + 3]));
      return c;
  }

  uc getGrayScale(Pixel p) {
    return grayscale(getColor(p));
  }

private:
  uc *data;
  int _width, _height, comp;

  uc grayscale(const Color &c) {
      return 0.299*c.r + 0.587*c.g + 0.114*c.b;
  }

};


class FaceDetection
{
public:
  Image *container, *object;

  FaceDetection(char* containerFile, char* objectFile)
  {
    container = new Image(containerFile);
    object = new Image(objectFile);
  }

  // 0:   exact same image
  // 255: very different
  uc getWindowDiff(Pixel origin, int windowWidth, int windowHeight)
  {
    int totalDiff = 0;
    for(int y = 0; y < windowHeight; ++y)
    {
      for(int x = 0; x < windowWidth; ++x)
      {
        Pixel cPixel(origin.x + x, origin.y + y); // container pixel
        Pixel oPixel(x,y);                        // object pixel

        uc cAlpha = container->getColor(cPixel).a;
        uc oAlpha = container->getColor(oPixel).a;
        uc cGray = container->getGrayScale(cPixel);
        uc oGray = object->getGrayScale(oPixel);

        uc diffAlpha = getDiff(cAlpha, oAlpha);
        if(diffAlpha < 200) // if alpha channels differ, ignore the diff (consider pixels are equal)
        {
          uc diff = getDiff(cGray, oGray);
          totalDiff += diff;
        }
      }
    }

    return uc(totalDiff / (windowWidth * windowHeight));
  }


  // 0:   same grayscale
  // 255: very different
  uc getDiff(uc gray1, uc gray2)
  {
    return abs(gray1-gray2);
  }

  void saveResult(Pixel origin) {
      uc *result = new uc[container->width() * container->height() * 3 * sizeof(uc)];
      for(int y = 0; y < container->height(); ++y) {
          for(int x = 0; x < container->width();  ++x) {
              Pixel p(x,y);
              int offset = (y * container->width() + x) * 3;
              if (belongsTo(p,origin,object->width(),object->height())) {
                  result[offset + 0] = container->getColor(p).r;
                  result[offset + 1] = 255;
                  result[offset + 2] = container->getColor(p).b;
              }
              else {
                  result[offset + 0] = container->getColor(p).r;
                  result[offset + 1] = container->getColor(p).g;
                  result[offset + 2] = container->getColor(p).b;
              }
          }
      }
      stbi_write_bmp("result.bmp", container->width(), container->height(), 3, result);
  }

private:
  bool belongsTo(Pixel p, Pixel o, int w, int h) {
      return p.x >= o.x && p.x < o.x+w && p.y >= o.y && p.y < o.y+h;
  }

};

__global__ void getWindowDiff(uc *imgContainer, int containerWidth, int containerHeight,
                              uc *imgObject, int objectWidth, int objectHeight,
                              int  *resultMatrix, unsigned int step)
{

if(threadIdx.x != 0) return;
    //printf("threadIdx(%d,%d), blockIdx(%d,%d), blockDim(%d,%d).\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);//, tx=%d, ox=%d, oy=%d\n", blockIdx.x, blockDim.x, tx, ox, oy);
    
    int ox = blockIdx.x * step;
    int oy = blockIdx.y * step;

    //printf("GOOD [ox=%d, oy=%d]\n", ox, oy);

    int totalDiff = 0;
    int cOffset = oy * containerWidth + ox;
    for(int y = 0; y < objectHeight; ++y)
    {
        for(int x = 0; x < objectWidth; ++x)
        {
	    int objOffset = y * objectWidth + x;
            totalDiff += abs(imgContainer[cOffset + objOffset] - imgObject[objOffset]);
        }
    }

    resultMatrix[cOffset] = totalDiff;
}

void CheckCudaError(char sms[], int line);

int main(int argc, char** argv)
{
  if(argc < 4)
      cout << "usage: " << argv[0] << " <container file name> <object file name>" << endl;
  
  for (int i = 0; i < argc; ++i)
      cout << argv[i] << endl;

  FaceDetection fc(argv[1], argv[2]);


  unsigned int nThreads = 1024;
  unsigned int nBlocks = 254; // Assuming square matrices
  dim3 dimGrid(nBlocks, nBlocks, 1); //(nBlocks.x, nBlocks.y, 1)
  dim3 dimBlock(nThreads, 1, 1); //(nThreads.x, nThreads.y, 1)
  unsigned int dw = fc.container->width() - fc.object->width();
  unsigned int step = dw / nBlocks;
  printf("step: %d\n", step);

  int numBytesContainer = fc.container->width() * fc.container->height() * sizeof(uc);
  int numBytesObject = fc.object->width() * fc.object->height() * sizeof(uc);

  printf("Container File: %s, size(%d px, %d px), bytes(%d B)\n",
         fc.container->filename, fc.container->width(), fc.container->height(),
         numBytesContainer);
  printf("Object File: %s, size(%d px, %d px), bytes(%d B)\n",
         fc.object->filename, fc.object->width(), fc.object->height(),
         numBytesObject);
  
  // Obtener Memoria en el host
  printf("Getting memory in the host to allocate GS images and resultMatrix...\n");
  int *h_resultDiffMatrix = (int*) malloc(numBytesContainer * sizeof(int) / sizeof(uc));
  printf("resultMatrix memory in the host got.\n");
  uc *h_containerImageGS = (uc*) malloc(numBytesContainer);
  printf("ContainerGS memory in the host got.\n");
  uc *h_objectImageGS = (uc*) malloc(numBytesObject);
  printf("ObjectGS memory in the host got.\n");
  printf("Memory in the host got.\n");

  //Obtiene Memoria [pinned] en el host
  //cudaMallocHost((float**)&h_x, numBytes);
  //cudaMallocHost((float**)&h_y, numBytes);
  //cudaMallocHost((float**)&H_y, numBytes);   // Solo se usa para comprobar el resultado

  printf("Filling resultMatrix in the host...\n");
  for(int y = 0; y < fc.container->height(); ++y) {
      for(int x = 0; x < fc.container->width(); ++x) {
          h_resultDiffMatrix[y * fc.container->width() + x] = -1;
      }
  }

  printf("Filling ContainerGS in the host with GS values...\n");
  for(int y = 0; y < fc.container->height(); ++y) {
      for(int x = 0; x < fc.container->width(); ++x) {
          h_containerImageGS[y * fc.container->width() + x] = fc.container->getGrayScale(Pixel(x,y));
      }
  }

  //Fill object image with its grayscale
  printf("Filling ObjectGS in the host with GS values...\n");
  for(int y = 0; y < fc.object->height(); ++y) {
      for(int x = 0; x < fc.object->width(); ++x) {
          h_objectImageGS[y * fc.object->width() + x] = fc.object->getGrayScale(Pixel(x,y));
      }
  }


  //For every pixel(x,y), it contains the result of the avg diff of the window beginning in that pixel
  int *d_resultDiffMatrix;
  uc *d_containerImageGS; // bigger image
  uc *d_objectImageGS;    // smaller image

  // Obtener Memoria en el device
  printf("Getting memory in the device to allocate GS images and resultMatrix...\n");
  cudaMalloc((int**)&d_resultDiffMatrix, numBytesContainer * sizeof(int) / sizeof(uc)); //result diff matrix
  printf("resultMatrix memory in the device got.\n");
  cudaMalloc((uc**)&d_containerImageGS, numBytesContainer);
  printf("ContainerGS memory in the device got.\n");
  cudaMalloc((uc**)&d_objectImageGS, numBytesObject);
  printf("ObjectGS memory in the device got.\n");
  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);

  // Copiar datos desde el host en el device
  printf("Copying resultMatrix in the host to the device...\n");
  cudaMemcpy(d_resultDiffMatrix, h_resultDiffMatrix, numBytesContainer * sizeof(int) / sizeof(uc), cudaMemcpyHostToDevice);
  printf("Copying ContainerGS in the host to the device...\n");
  cudaMemcpy(d_containerImageGS, h_containerImageGS, numBytesContainer, cudaMemcpyHostToDevice);
  printf("Copying objectGS in the host to the device...\n");
  cudaMemcpy(d_objectImageGS, h_objectImageGS, numBytesObject, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  printf("Synchronizing device...\n");
  cudaDeviceSynchronize();
  printf("Device synchronized.\n");
  
// Ejecutar el kernel
  printf("Executing kernel getWindowDiff...\n");
  getWindowDiff<<<dimGrid, dimBlock>>>(
         d_containerImageGS, fc.container->width(), fc.container->height(),
         d_objectImageGS, fc.object->width(), fc.object->height(),
         d_resultDiffMatrix, step);

  printf("Synchronizing device...\n");
  cudaDeviceSynchronize();
  printf("Device synchronized.\n");

  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  // Obtener el resultado desde el host
  printf("Retrieving resultMatrix from device to host...\n");
  cudaMemcpy(h_resultDiffMatrix, d_resultDiffMatrix, numBytesContainer * sizeof(int) / sizeof(uc), cudaMemcpyDeviceToHost);
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Liberar Memoria del device
  printf("Freeing device memory...\n");
  cudaFree(d_resultDiffMatrix);
  cudaFree(d_containerImageGS);
  cudaFree(d_objectImageGS);

  printf("Synchronizing device...\n");
  cudaDeviceSynchronize();
  printf("Device synchronized.\n");

  //Treat Result
  int minDiff = 400000000;
  int resultX = 0, resultY = 0;
  for(int y = 0; y < fc.container->height(); ++y)
  {
      for(int x = 0; x < fc.container->width(); ++x)
      {
          int diff = h_resultDiffMatrix[y * fc.container->width() + x];
	  //printf("(%d,%d): %d\n", x, y, diff);
          if( (diff != -1 && diff < minDiff) || minDiff == 400000000)
          {
                minDiff = diff;
                resultX = x;
                resultY = y;
          }
      }
  }

  printf("Best guess: (%d, %d), with diff: %d\n\n", resultX, resultY, minDiff);

  printf("nThreads: %d\n", nThreads);
  printf("nBlocks: %d\n", nBlocks);
}


void CheckCudaError(char sms[], int line) {
  cudaError_t error;
  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }
}



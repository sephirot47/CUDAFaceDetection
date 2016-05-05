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
  Image(char* filename) {
    FILE *file = fopen(filename, "r");
    if (file != NULL)
    {
      data = stbi_load_from_file(file, &_width, &_height, &comp, 4); // rgba
      fclose (file);
    }
    else
    {
        printf("Image '%s' not found\n", filename);
    }
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

__global__ void getWindowDiff(uc *imgContainer, int containerWidth,
                              uc *imgObject, int objectWidth, int objectHeight,
                              uc *resultMatrix, unsigned int step)
{
    if(threadIdx.x != 0) return;

    int ox = blockIdx.x * (blockDim.x + step);// + threadIdx.x;
    int oy = blockIdx.y * (blockDim.y + step);// + threadIdx.y;

    int totalDiff = 0;
    for(int y = 0; y < objectHeight; ++y)
    {
        for(int x = 0; x < objectWidth; ++x)
        {
            totalDiff += abs(imgContainer[y * objectWidth + x] - imgObject[y * objectWidth + x]);
        }
    }

    resultMatrix[oy * containerWidth + ox] = totalDiff / (objectWidth * objectHeight);
}

void CheckCudaError(char sms[], int line);

int main(int argc, char** argv)
{
  if(argc < 4)
      cout << "usage: " << argv[0] << " <container file name> <object file name>" << endl;

  FaceDetection fc(argv[1], argv[2]);

  unsigned int nThreads = 1024;
  unsigned int nBlocks = 65535; //SUPONIENDO QUE LAS DOS IMAGENES SON CUADRADAS!
  unsigned int step = (fc.container->width() - fc.object->width()) / nBlocks;


  //Save two images in grayscale

  int numBytesContainer = fc.container->width() * fc.container->height() * sizeof(uc);
  int numBytesObject = fc.object->width() * fc.object->height() * sizeof(uc);
  uc *h_resultDiffMatrix = (uc*) malloc(numBytesContainer);
  uc *h_containerImageGS = (uc*) malloc(numBytesContainer);
  uc *h_objectImageGS = (uc*) malloc(numBytesObject);

  //Fill resultDiffMatrix with zeroes
  for(int y = 0; y < fc.container->height(); ++y) {
      for(int x = 0; x < fc.container->width(); ++x) {
          h_objectImageGS[y * fc.container->width() + x] = 0;
      }
  }

  //Fill container image with its grayscale
  for(int y = 0; y < fc.container->height(); ++y) {
      for(int x = 0; x < fc.container->width(); ++x) {
          h_containerImageGS[y * fc.container->width() + x] = fc.container->getGrayScale(Pixel(x,y));
      }
  }

  //Fill object image with its grayscale
  for(int y = 0; y < fc.object->height(); ++y) {
      for(int x = 0; x < fc.object->width(); ++x) {
          h_objectImageGS[y * fc.object->width() + x] = fc.object->getGrayScale(Pixel(x,y));
      }
  }

  // ////////////////////////////////////////


  //Obtiene Memoria [pinned] en el host
  //cudaMallocHost((float**)&h_x, numBytes);
  //cudaMallocHost((float**)&h_y, numBytes);
  //cudaMallocHost((float**)&H_y, numBytes);   // Solo se usa para comprobar el resultado


  // Obtener Memoria en el device

  //For every pixel(x,y), it contains the result of the avg diff of the window beginning in that pixel
  uc *d_resultDiffMatrix;
  uc *d_containerImageGS; //bigger image
  uc *d_objectImageGS;    //smaller image

  cudaMalloc((uc**)&d_resultDiffMatrix, numBytesContainer); //result diff matrix
  cudaMalloc((uc**)&d_containerImageGS, numBytesContainer);
  cudaMalloc((uc**)&d_objectImageGS, numBytesObject);
  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);

  // Copiar datos desde el host en el device
  cudaMemcpy(d_resultDiffMatrix, h_resultDiffMatrix,
             numBytesContainer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_containerImageGS, h_containerImageGS,
             numBytesContainer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_objectImageGS, h_objectImageGS,
             numBytesObject, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  // ////////////////////////////////////////

  // Ejecutar el kernel
  getWindowDiff<<<nBlocks, nThreads>>>(d_containerImageGS, fc.container->width(),
                                       d_objectImageGS, fc.object->width(), fc.object->height(),
                                       d_resultDiffMatrix, step);
  CheckCudaError((char *) "Invocar Kernel", __LINE__);


  // Obtener el resultado desde el host
  // Guardamos el resultado en H_y para poder comprobar el resultado
  cudaMemcpy(h_resultDiffMatrix, d_resultDiffMatrix, numBytesContainer, cudaMemcpyDeviceToHost);
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Liberar Memoria del device
  cudaFree(d_resultDiffMatrix);
  cudaFree(d_containerImageGS);
  cudaFree(d_objectImageGS);

  cudaDeviceSynchronize();

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



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

void saveImage(uc *img, int x, int y, int width, int height, int imgWidth, const char *filename)
{
	uc *aux = (uc*) malloc(width * height * 3 * sizeof(uc));
	for(int iy = 0; iy < height; ++iy)
	{
		for(int ix = 0; ix < width; ++ix)
		{
			int offset = (iy * width + ix) * 3;
			aux[offset + 0] = img[(iy+y) * imgWidth + ix + x];
			aux[offset + 1] = img[(iy+y) * imgWidth + ix + x];
			aux[offset + 2] = img[(iy+y) * imgWidth + ix + x];
			printf("offset:%d  |  x:%d, y:%d", offset, x, y); fflush(stdout);
		}
	}
	printf("wololo"); fflush(stdout);
	stbi_write_bmp(filename, width, height, 3, aux);
	free(aux);
}

//Increase contrast
void histogramEqualization(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
	int intensityToNPixels[256];
	for(int i = 0; i < 256; ++i) intensityToNPixels[i] = 0;

	for(int y = oy; y < oy + height; ++y)
	{
		for(int x = ox; x < ox + width; ++x)
		{
			int offset = y * imgWidth + x;
			uc v = img[offset];
		 	intensityToNPixels[v]++;	
		}
	}

	int npixels = width * height;
	float accumulatedProbs[256];
	accumulatedProbs[0] = intensityToNPixels[0];
	for(int i = 1; i < 256; ++i) accumulatedProbs[i] = accumulatedProbs[i-1] + intensityToNPixels[i];

	for(int y = oy; y < oy + height; ++y)
	{
		for(int x = ox; x < ox + width; ++x)
		{
			fflush(stdout);
			int offset = y * imgWidth + x;
			uc v = img[offset];
			printf("v: %d\n", v);
			fflush(stdout);
			printf("offset: %d\n", offset);
			fflush(stdout);
			printf("accumProbs[%d] = %f\n", v, accumulatedProbs[v]);
			fflush(stdout);
			img[offset] = floor( 255 * (accumulatedProbs[v] / npixels) );
			//img[offset] = uc( (float(img[offset]-minv) / (maxv-minv)) * 255 );
			printf("v: %d,   equalized(%d,%d): %d\n", v, x, y, img[offset]);
			fflush(stdout);
		}
	}
	printf("Finished!"); fflush(stdout);
}

//Always to a smaller size
void resize(uc *src, int srcx, int srcy, int srcw, int srch, //x,y,width,height
	    uc *dst, int dstx, int dsty, int dstw, int dsth) //x,y,width,height
{
    //Every square of size (bw,bh), will be substituted
    //by one pixel in the dst image
    int bw = srcw / dstw;
    int bh = srch / dsth;

    //For each pixel in the dst
    for(int dy = 0; dy < dsth; ++dy)
    {
        for(int dx = 0; dx < dstw; ++dx)
        {
	    //Offset per dst pixel. Every pixel we move in x,y in dst,
	    //we move bw,bh in the src image.
            int resizeOffset = (dy * bh) * srcw + (dx * bw);
            
	    //Save in its position the mean of the corresponding window pixels
            int mean = 0;
            for(int sy = 0; sy < bh; ++sy)
            {
                for(int sx = 0; sx < bw; ++sx)
               {
                    int srcOffset = sy * srcw + sx;
                    uc v = src[srcOffset + resizeOffset];
                    mean += v;
                }
            }
            mean /= bw * bh;
    	    dst[dy * dstw + dx] = mean;
        }
    }
}

__global__ void getWindowDiff(uc *imgContainer, int containerWidth, int containerHeight,
                              uc *imgObject, int objectWidth, int objectHeight,
                              int  *resultMatrix, unsigned int step,
                              uc *resizeMatrix9x9, uc *resizeMatrix30x30)
{
return;
    if(threadIdx.x != 0) return;
    //printf("threadIdx(%d,%d), blockIdx(%d,%d), blockDim(%d,%d).\n",
              //threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);

    int ox = blockIdx.x * step;
    int oy = blockIdx.y * step;

    int cOffset = oy * containerWidth + ox;

    //FIRST STEP: Resize the current window to 9x9
    // For every pixel of the 9x9 resized image,
    // how many pixels of the current window have to be sampled?
    // (do the mean with them)
    int xMinification9x9 = objectWidth / 9;
    int yMinification9x9 = objectHeight / 9;
    //For each pixel in the 9x9 image
    //


    int totalDiff = 0;
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
  cout << "Usage: " << argv[0] << " <container file name> <object file name>" << endl;
  for (int i = 0; i < argc; ++i) { cout << argv[i] << endl; }


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
  int numBytesResultMatrix = numBytesContainer * sizeof(int) / sizeof(uc);

  printf("Container File: %s, size(%d px, %d px), bytes(%d B)\n",
         fc.container->filename, fc.container->width(), fc.container->height(),
         numBytesContainer);
  printf("Object File: %s, size(%d px, %d px), bytes(%d B)\n",
         fc.object->filename, fc.object->width(), fc.object->height(),
         numBytesObject);
  
  // Obtener Memoria en el host
  printf("Getting memory in the host to allocate GS images and resultMatrix...\n");
  int *h_resultDiffMatrix = (int*) malloc(numBytesResultMatrix);
  uc *h_containerImageGS = (uc*) malloc(numBytesContainer);
  uc *h_objectImageGS = (uc*) malloc(numBytesObject);
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
  uc *d_containerImageGS, *d_objectImageGS, *d_resizeMatrix9x9, *d_resizeMatrix30x30;

  // Obtener Memoria en el device
  printf("Getting memory in the device to allocate GS images, resultMatrix, and resizeMatrices...\n");
  cudaMalloc((int**)&d_resultDiffMatrix, numBytesResultMatrix); //result diff matrix
  cudaMalloc((uc**)&d_resizeMatrix9x9, numBytesContainer * 9 * 9);
  cudaMalloc((uc**)&d_resizeMatrix30x30, numBytesContainer * 30 * 30);
  cudaMalloc((uc**)&d_containerImageGS, numBytesContainer);
  cudaMalloc((uc**)&d_objectImageGS, numBytesObject);
  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);

  // Copiar datos desde el host en el device
  printf("Copying matrices in the host to the device...\n");
  cudaMemcpy(d_resultDiffMatrix, h_resultDiffMatrix, numBytesResultMatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_containerImageGS, h_containerImageGS, numBytesContainer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_objectImageGS, h_objectImageGS, numBytesObject, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  cudaDeviceSynchronize();
  
  // Ejecutar el kernel
  printf("Executing kernel getWindowDiff...\n");
  getWindowDiff<<<dimGrid, dimBlock>>>(
         d_containerImageGS, fc.container->width(), fc.container->height(),
         d_objectImageGS, fc.object->width(), fc.object->height(),
         d_resultDiffMatrix, step,
         d_resizeMatrix9x9, d_resizeMatrix30x30);

  printf("MaxContrasting...");
  //saveImage(h_containerImageGS, 0, 0, 9, 9, "test.bmp");
  histogramEqualization(h_containerImageGS, 340, 440, 270, 250, fc.container->width());
  printf("MaxContrasted!"); fflush(stdout);
  
  printf("Resizing...");
  resize(h_containerImageGS, 340, 440, 270, 250, 
   	 h_containerImageGS, 0, 0, 9, 9);
  printf("Resized!");
  
  printf("Saving img..."); fflush(stdout);
  saveImage(h_containerImageGS, 0, 0, 9, 9, fc.container->width(), "test.bmp");
  printf("Image saved!");

  cudaDeviceSynchronize();
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  // Obtener el resultado desde el host
  printf("Retrieving resultMatrix from device to host...\n");
  cudaMemcpy(h_resultDiffMatrix, d_resultDiffMatrix, numBytesResultMatrix, cudaMemcpyDeviceToHost);
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Liberar Memoria del device
  printf("Freeing device memory...\n");
  cudaFree(d_resultDiffMatrix);
  cudaFree(d_containerImageGS);
  cudaFree(d_objectImageGS);

  cudaDeviceSynchronize();

  //Treat Result
  //int minDiff = 400000000;
/*  int resultX = 0, resultY = 0;
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
*/
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



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
  Image *image;
  
  FaceDetection(char* imageFile)
  {
    image = new Image(imageFile);
  }

  void saveResult(Pixel origin) {
      uc *result = new uc[image->width() * image->height() * 3 * sizeof(uc)];
      for(int y = 0; y < image->height(); ++y) {
          for(int x = 0; x < image->width();  ++x) {
              Pixel p(x,y);
              int offset = (y * image->width() + x) * 3;
              if (belongsTo(p,origin,object->width(),object->height())) {
                  result[offset + 0] = image->getColor(p).r;
                  result[offset + 1] = 255;
                  result[offset + 2] = image->getColor(p).b;
              }
              else {
                  result[offset + 0] = image->getColor(p).r;
                  result[offset + 1] = image->getColor(p).g;
                  result[offset + 2] = image->getColor(p).b;
              }
          }
      }
      stbi_write_bmp("output/result.bmp", image->width(), image->height(), 3, result);
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

__global__ void getWindowDiff(uc *imgContainer, int imageWidth, int imageHeight,
                              int  *resultMatrix, unsigned int step)
{
return;
    if(threadIdx.x != 0) return;
    //printf("threadIdx(%d,%d), blockIdx(%d,%d), blockDim(%d,%d).\n",
              //threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);

    int ox = blockIdx.x * step;
    int oy = blockIdx.y * step;

    int cOffset = oy * imageWidth + ox;

    //FIRST STEP: Resize the current window to 9x9
    // For every pixel of the 9x9 resized image,
    // how many pixels of the current window have to be sampled?
    // (do the mean with them)
    //For each pixel in the 9x9 image
    // TODO

    resultMatrix[cOffset] = 0;
}

void CheckCudaError(char sms[], int line);

int main(int argc, char** argv)
{
  cout << "Usage: " << argv[0] << " <image file name>" << endl;
  for (int i = 0; i < argc; ++i) { cout << argv[i] << endl; }


  FaceDetection fc(argv[1]);


  unsigned int nThreads = 1024;
  unsigned int nBlocks = 254; // Assuming square matrices
  dim3 dimGrid(nBlocks, nBlocks, 1); //(nBlocks.x, nBlocks.y, 1)
  dim3 dimBlock(nThreads, 1, 1); //(nThreads.x, nThreads.y, 1)
  int windowWidth = 250;
  unsigned int dw = fc.image->width() - windowWidth;
  unsigned int step = dw / nBlocks;
  printf("step: %d\n", step);

  int numBytesContainer = fc.image->width() * fc.image->height() * sizeof(uc);
  int numBytesResultMatrix = numBytesContainer * sizeof(int) / sizeof(uc);

  printf("Container File: %s, size(%d px, %d px), bytes(%d B)\n",
         fc.image->filename, fc.image->width(), fc.image->height(),
         numBytesContainer);
  
  // Obtener Memoria en el host
  printf("Getting memory in the host to allocate GS images and resultMatrix...\n");
  int *h_resultDiffMatrix = (int*) malloc(numBytesResultMatrix);
  uc *h_imageGS = (uc*) malloc(numBytesContainer);
  printf("Memory in the host got.\n");

  //Obtiene Memoria [pinned] en el host
  //cudaMallocHost((float**)&h_x, numBytes);
  //cudaMallocHost((float**)&h_y, numBytes);
  //cudaMallocHost((float**)&H_y, numBytes);   // Solo se usa para comprobar el resultado

  printf("Filling resultMatrix in the host...\n");
  for(int y = 0; y < fc.image->height(); ++y) {
      for(int x = 0; x < fc.image->width(); ++x) {
          h_resultDiffMatrix[y * fc.image->width() + x] = -1;
      }
  }

  printf("Filling ContainerGS in the host with GS values...\n");
  for(int y = 0; y < fc.image->height(); ++y) {
      for(int x = 0; x < fc.image->width(); ++x) {
          h_imageGS[y * fc.image->width() + x] = fc.image->getGrayScale(Pixel(x,y));
      }
  }


  //For every pixel(x,y), it contains the heuristic value of the window beginning in that pixel
  int *d_resultDiffMatrix;
  uc *d_imageGS;

  // Obtener Memoria en el device
  printf("Getting memory in the device to allocate GS images, resultMatrix, and resizeMatrices...\n");
  cudaMalloc((int**)&d_resultDiffMatrix, numBytesResultMatrix); //result diff matrix
  cudaMalloc((uc**)&d_imageGS, numBytesContainer);
  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);

  // Copiar datos desde el host en el device
  printf("Copying matrices in the host to the device...\n");
  cudaMemcpy(d_resultDiffMatrix, h_resultDiffMatrix, numBytesResultMatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageGS, h_imageGS, numBytesContainer, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  cudaDeviceSynchronize();
  
  // Ejecutar el kernel
  printf("Executing kernel getWindowDiff...\n");
  getWindowDiff<<<dimGrid, dimBlock>>>(
         d_imageGS, fc.image->width(), fc.image->height(),
         d_resultDiffMatrix, step);

  printf("Histogram equalization...");
  histogramEqualization(h_imageGS, 340, 440, 270, 250, fc.image->width());
  printf("Equalized!"); fflush(stdout);
  
  printf("Resizing...");
  resize(h_imageGS, 340, 440, 270, 250, 
   	 h_imageGS, 0, 0, 9, 9);
  printf("Resized!");
  
  printf("Saving image..."); fflush(stdout);
  saveImage(h_imageGS, 0, 0, 9, 9, fc.image->width(), "test.bmp");
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
  cudaFree(d_imageGS);

  cudaDeviceSynchronize();

  //TODO: treat result  

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



#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
#define NUM_THREADS 1024
#define NUM_BLOCKS 64
#define IMG_CHANNELS 3

#define THRESH_9x9 40
#define THRESH_30x30 550

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
        data = stbi_load_from_file(file, &_width, &_height, &comp, IMG_CHANNELS); // rgba
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

struct Box { int x,y,w,h; Box(int _x, int _y, int _w, int _h) { x=_x; y=_y; w=_w; h=_h; } };

class FaceDetection
{
public:
  Image *image;
  vector<Box> resultWindows;
  
  FaceDetection(char* imageFile)
  {
    image = new Image(imageFile);
  }

  void saveResult()
  {
      int boxStroke = 1;
      printf("Saving result...\n");

      uc *result = new uc[image->width() * image->height() * 3 * sizeof(uc)];
      for(int y = 0; y < image->height(); ++y)
      {
          for(int x = 0; x < image->width();  ++x)
          {
              Pixel p(x,y);
              bool isBoundary = false;
              for(Box b : resultWindows)
              {
                  if(belongsTo(p, b))
                  {
                      if(abs(x - b.x) <= boxStroke ||
                         abs(y - b.y) <= boxStroke ||
                         abs(x - (b.x + b.w - 1)) <= boxStroke ||
                         abs(y - (b.y + b.h - 1)) <= boxStroke)
                      {
                          isBoundary = true;
                      }
                  }
              }

              int offset = (y * image->width() + x) * 3;
              if(isBoundary)
              {
                  result[offset + 0] = 255 - image->getColor(p).r;
                  result[offset + 1] = 255 - image->getColor(p).g;
                  result[offset + 2] = 255 - image->getColor(p).b;
              }

              else
              {
                  result[offset + 0] = image->getColor(p).r;
                  result[offset + 1] = image->getColor(p).g;
                  result[offset + 2] = image->getColor(p).b;
              }
          }
      }

      stbi_write_bmp("output/result.bmp", image->width(), image->height(), 3, result);
      //saveImage(result, 0, 0, image->width(), image->height(), image->width()*3, );
  }

private:
  bool belongsTo(Pixel p, Box b) {
      return p.x >= b.x && p.x <= b.x+b.w && p.y >= b.y && p.y <= b.y+b.h;
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
        }
    }
    stbi_write_bmp(filename, width, height, 3, aux);
    free(aux);
}

// Resize always to a smaller size -> downsample
__host__ __device__ void resize(uc *src, int srcx, int srcy, int srcw, int srch, int srcTotalWidth, //x,y,width,height
            uc *dst, int dstx, int dsty, int dstw, int dsth, int dstTotalWidth) //x,y,width,height
{
    //Every square of size (bw,bh), will be substituted
    //by one pixel in the dst image
    float bw = float(srcw) / dstw;
    float bh = float(srch) / dsth;

    //For each pixel in the dst
    for(int dy = dsty; dy < dsty + dsth; ++dy)
    {
        for(int dx = dstx; dx < dstx + dstw; ++dx)
        {
            //Offset per dst pixel. Every pixel we move in x,y in dst,
            //we move bw,bh in the src image.
            int resizeOffset = ceil((dy * bh)) * srcTotalWidth + ceil((dx * bw));

            //Save in its position the mean of the corresponding window pixels
            int mean = 0;
            for(int sy = 0; sy < floor(bh); ++sy)
            {
                for(int sx = 0; sx < floor(bw); ++sx)
               {
                    int srcOffset = (srcy + sy) * srcTotalWidth + (srcx + sx);
                    uc v = src[srcOffset + resizeOffset];
                    mean += v;
                }
            }

            mean /= floor(bw * bh);
            dst[dy * dstTotalWidth + dx] = mean;
        }
    }
}


__device__ float getHistogram(uc *img, int ox, int oy, int width, int height, int imgWidth, float histogram[256]) {

  float npixels = width * height;
  float unitProb = 1.0f/npixels;
  float maxfreq = 0.0f;
  for(int i = 0; i < 256; ++i) histogram[i] = 0;
  for(int y = oy; y < oy + height; ++y)
  {
      for(int x = ox; x < ox + width; ++x)
      {
      int offset = y * imgWidth + x;
      uc v = img[offset];
      histogram[v] += unitProb;
      if(histogram[v] > maxfreq) maxfreq = histogram[v];
      }
  }
  return maxfreq;
}

__device__ void toBlackAndWhite(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    for(int y = oy; y < oy + height; ++y)
    {
        for(int x = ox; x < ox + width; ++x)
        {
            int offset = y * imgWidth + x;
            uc v = img[offset];
            img[offset] = v > 200 ? 255 : 0;
        }
    }
}

// Increase contrast
__device__ void histogramEqualization(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    __shared__ float histogram[256];
    getHistogram(img, ox, oy, width, height, imgWidth, histogram);

    __shared__ float accumulatedProbs[256];
    accumulatedProbs[0] = histogram[0];
    for(int i = 1; i < 256; ++i) accumulatedProbs[i] = accumulatedProbs[i-1] + histogram[i];

    for(int y = oy; y < oy + height; ++y)
    {
        for(int x = ox; x < ox + width; ++x)
        {
            int offset = y * imgWidth + x;
            uc v = img[offset];
            img[offset] = floor(255 * accumulatedProbs[v]);
        }
    }
}

__device__ uc getFirstStageHeuristic(uc *img) {
    int v = img[22] - (img[19]+img[20]+img[24]+img[25]+img[58])/5;
    return v < 0 ? 0 : v;
}

// Find edges in horizontal direction
__device__ void sobelEdgeDetection(uc *img,
                                   int ox, int oy,
                                   int winWidth, int winHeight,
                                   int imgWidth,
                                   uc *sobelImg)
{
    uc threshold = 24;

    //unsigned int i = ox + threadIdx.x + oy * imgWidth;
    int size = winWidth * winHeight;
    for (int i = threadIdx.x; i < size; i+=NUM_THREADS)
    {
        int wx = i % winWidth;
        int wy = i / winWidth;
        int winOffset = wy * winWidth + wx;

        int x = ox + wx;
        int y = oy + wy;
        int imgOffset = y * imgWidth + x;

        if (y == oy or y == oy+winHeight-1 or x == ox or x == ox+winWidth-1)
            sobelImg[winOffset] = 255;
        else {
            uc upperLeft  = img[imgOffset - imgWidth - 1];
            uc upperRight = img[imgOffset - imgWidth + 1];
            uc up         = img[imgOffset - imgWidth];
            uc down       = img[imgOffset + imgWidth];
            uc lowerLeft  = img[imgOffset + imgWidth - 1];
            uc lowerRight = img[imgOffset + imgWidth + 1];

            int sum = -upperLeft - upperRight - 2*up + 2*down + lowerLeft + lowerRight;
            if(sum >= threshold)
                sobelImg[winOffset] = 0;
            else
                sobelImg[winOffset] = 255;
        }
    }
}

__device__ uc getWindowMeanGS(uc *img, int ox, int oy, int winWidth, int winHeight, int imgWidth) {
    int sum = 0;
    for(int y = oy; y < oy + winHeight; ++y)
    {
        for(int x = ox; x < ox + winWidth; ++x)
        {
            int offset = y * imgWidth + x;
            sum += img[offset];
        }
    }

    return uc(sum / (winWidth * winHeight));
}

__device__ int getSecondStageHeuristic(uc *img) {
    int sumDiff = 0;
    int leftEye = getWindowMeanGS(img,2,4,9,5,30);
    sumDiff += leftEye;
    int rightEye = getWindowMeanGS(img,18,4,9,5,30);
    sumDiff += rightEye;
    sumDiff += abs(rightEye - leftEye); // simmetry
    sumDiff += 255-getWindowMeanGS(img,11,1,6,13,30); // upper nose
    sumDiff += abs(125 - getWindowMeanGS(img,10,15,9,5,30)); // lower nose
    int leftCheek = 255-getWindowMeanGS(img,1,10,8,10,30); // left cheek
    sumDiff += leftCheek;
    int rightCheek = 255-getWindowMeanGS(img,19,10,8,10,30); // right cheek
    sumDiff += rightCheek;
    sumDiff += abs(leftCheek - rightCheek);
    sumDiff += getWindowMeanGS(img,8,21,13,5,30); // mouth
    return sumDiff;
}

__global__ void detectFaces(uc *img,
                            int winWidth, int winHeight,
                            uc  *resultMatrix)
{
    int step = (IMG_WIDTH - winWidth) / NUM_BLOCKS;
    if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        printf("step: %d\n", step);

    int x = blockIdx.x * step;
    int y = blockIdx.y * step;

    // FIRST HEURISTIC
    __shared__ uc window30x30[30*30];
    __shared__ uc hv1;
    if(threadIdx.x == 0) {
        resize(img,
               x, y, winWidth, winHeight, IMG_WIDTH,
               window30x30,
               0, 0, 9, 9, 9);
        histogramEqualization(window30x30, 0, 0, 9, 9, 9);
        hv1 = getFirstStageHeuristic(window30x30);
    }

    __syncthreads();

    int cOffset = y * NUM_BLOCKS + x;
    if (hv1 >= THRESH_9x9)
    {
        // SECOND HEURISTIC
        //uc *sobelImg = (uc*) malloc(winWidth * winHeight * sizeof(uc));
        __shared__ uc sobelImg[200*200];
        sobelEdgeDetection(img, x, y, winWidth, winHeight, IMG_WIDTH, sobelImg);

        if(threadIdx.x == 0) {
            resize(sobelImg,
                   0, 0, winWidth, winHeight, winWidth,
                   window30x30,
                   0, 0, 30, 30, 30);
            toBlackAndWhite(window30x30, 0, 0, 30, 30, 30);

            //free(sobelImg);

            int hv2 = getSecondStageHeuristic(window30x30);
            if (hv2 <= THRESH_30x30)
            {
                // Save result! We detected a face yayy
                resultMatrix[cOffset] = 1;
            }
            else resultMatrix[cOffset] = 0;
        }
    }
    else resultMatrix[cOffset] = 0;


    if(threadIdx.x == 0) {
        resultMatrix[cOffset] = 5;
    }

    /*if(threadIdx.x == 0) {
        resultMatrix[cOffset] = 0;
    }*/
}

void CheckCudaError(char sms[], int line);

int main(int argc, char** argv)
{
  cout << "Usage: " << argv[0] << " <image file name>" << endl;
  for (int i = 0; i < argc; ++i) { cout << argv[i] << endl; }


  //Read input
  FaceDetection fc(argv[1]);
  printf("image File: %s, size(%d px, %d px)\n",
         fc.image->filename, fc.image->width(), fc.image->height());
  //


  //Adapt input
  int numBytesImageOriginal = fc.image->width() * fc.image->height() * sizeof(uc);
  uc *h_imageGSOriginal = (uc*) malloc(numBytesImageOriginal);
  printf("Adapting input. Creating grayscale image....\n");
  for(int y = 0; y < fc.image->height(); ++y) {
      for(int x = 0; x < fc.image->width(); ++x) {
          h_imageGSOriginal[y * fc.image->width() + x] = fc.image->getGrayScale(Pixel(x,y));
      }
  }

  printf("Resizing original image....\n");
  int numBytesImage = IMG_WIDTH * IMG_HEIGHT * sizeof(uc);
  uc *h_imageGS = (uc*) malloc(numBytesImage);
  resize(h_imageGSOriginal,
         0, 0, fc.image->width(), fc.image->height(), fc.image->width(),
         h_imageGS,
         0, 0, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH
         );
  saveImage(h_imageGS,0,0,IMG_WIDTH,IMG_HEIGHT,IMG_WIDTH,"initialResizedImage.bmp");
  //


  //Obtiene Memoria [pinned] en el host
  //cudaMallocHost((float**)&h_x, numBytes);
  //cudaMallocHost((float**)&h_y, numBytes);
  //cudaMallocHost((float**)&H_y, numBytes);   // Solo se usa para comprobar el resultado

  printf("Getting memory in the host to allocate resultMatrix...\n");
  int numBytesResultMatrix = NUM_BLOCKS * NUM_BLOCKS * sizeof(uc);
  uc *h_resultMatrix = (uc*) malloc(numBytesResultMatrix);

  printf("Getting memory in the device...\n");
  uc *d_imageGS, *d_resultMatrix;
  cudaMalloc((uc**)&d_imageGS, numBytesImage);
  cudaMalloc((uc**)&d_resultMatrix, numBytesResultMatrix); //result diff matrix
  CheckCudaError((char *) "Get Device memory", __LINE__);

  printf("Copying matrices from host to device...\n");
  cudaMemcpy(d_resultMatrix, h_resultMatrix, numBytesResultMatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageGS, h_imageGS, numBytesImage, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copy data from host to device", __LINE__);

  cudaDeviceSynchronize();

  dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS, 1);
  dim3 dimBlock(NUM_THREADS, 1, 1);
  printf("Executing kernel detectFaces...\n");
  detectFaces<<<dimGrid, dimBlock>>>(
         d_imageGS,
         70, 100,
         d_resultMatrix);

  cudaDeviceSynchronize();
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  printf("Retrieving resultMatrix from device to host...\n");
  cudaMemcpy(h_resultMatrix, d_resultMatrix, numBytesResultMatrix, cudaMemcpyDeviceToHost);
  CheckCudaError((char *) "Retrieving resultMatrix from device to host", __LINE__);

  cudaDeviceSynchronize();

  //TODO: treat result
  for(int i = 0; i < NUM_BLOCKS; ++i)
  {
      for(int j = 0; j < NUM_BLOCKS; ++j)
      {
          printf("%d ", h_resultMatrix[i * NUM_BLOCKS + j]);
      }
      printf("\n");
  }

  // Liberar Memoria del device
  printf("Freeing device memory...\n");
  cudaFree(d_resultMatrix);
  cudaFree(d_imageGS);

  printf("FINISHED!");

}


void CheckCudaError(char sms[], int line) {
  cudaError_t error;
  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }
}



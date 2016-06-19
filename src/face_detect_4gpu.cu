#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stbi.h"
#include "../include/stbi_write.h"

#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
#define NUM_THREADS 1024
#define NUM_BLOCKS 255
#define IMG_CHANNELS 3
#ifndef NUM_DEVICES
#define NUM_DEVICES 4
#endif

// Optimal values 40, 550
#define THRESH_9x9 20     // Greater = more restrictive
#define THRESH_30x30 475  // Greater = less restrictive

typedef unsigned char uc;
using namespace std;

struct Pixel { int x,y; Pixel(int _x, int _y) {x=_x; y=_y;} };

struct Color { int r,g,b,a; Color(int _r, int _g, int _b, int _a) { r=_r; g=_g; b=_b; a=_a; } };

class Image
{
public:
    uc *data;
    char* filename;

    Image(char* _filename) {
	filename = _filename;
	FILE *file = fopen(_filename, "r");
	if (file != NULL) {
	    data = stbi_load_from_file(file, &_width, &_height, &comp, IMG_CHANNELS); // rgb
	    fclose (file);
	    printf("%s read successfully\n", _filename);
	}
	else printf("Image '%s' not found\n", _filename);
    }

    int width() { return _width;  }
    int height() { return _height; }

    Color getColor(Pixel p) {
	int offset = (p.y * _width + p.x) * comp;
	return Color((data[offset + 0]), (data[offset + 1]),
		     (data[offset + 2]), (data[offset + 3]));
    }

    void setColor(Pixel p, Color c) {
	int offset = (p.y * _width + p.x) * 3;
	data[offset + 0] = c.r;
	data[offset + 1] = c.g;
	data[offset + 2] = c.b;
    }

    uc getGrayScale(Pixel p) { return grayscale(getColor(p)); }

private:
    int _width, _height, comp;
    uc grayscale(const Color &c) { return 0.299*c.r + 0.587*c.g + 0.114*c.b; }
};

struct Box { int x,y,w,h; Box(int _x, int _y, int _w, int _h) { x=_x; y=_y; w=_w; h=_h; } };

class FaceDetection
{
public:
    vector<Box> resultWindows;
    Image *image;

    FaceDetection(char* imageFile)
    {
	image = new Image(imageFile);
    }

    void saveResult() {
	printf("Saving result...\n");

	Color c = Color(0,255,0,255);
	for(Box b : resultWindows) {
	    for (int i = b.x; i < b.x+b.w; ++i) {
		image->setColor(Pixel(i,b.y),c);
		image->setColor(Pixel(i,b.y+1),c);
	    }
	    for (int i = b.x; i < b.x+b.w; ++i) {
		image->setColor(Pixel(i,b.y+b.h-1),c);
		image->setColor(Pixel(i,b.y+b.h-2),c);
	    }
	    for (int i = b.y; i < b.y+b.h; ++i) {
		image->setColor(Pixel(b.x,i),c);
		image->setColor(Pixel(b.x+1,i),c);
	    }
	    for (int i = b.y; i < b.y+b.h; ++i) {
		image->setColor(Pixel(b.x+b.w-1,i),c);
		image->setColor(Pixel(b.x+b.w-2,i),c);
	    }
	}

	stbi_write_bmp("output/result.bmp", image->width(), image->height(), 3, image->data);
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

__device__ __host__ uc getWindowMeanGS(uc *img, int ox, int oy, int winWidth, int winHeight, int imgWidth) {
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

void resize_seq(uc *src, int srcx, int srcy, int srcw, int srch, int srcTotalWidth, //x,y,width,height
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
            //Save in its position the mean of the corresponding window pixels
            uc mean = getWindowMeanGS(src,
                                      srcx + ceil(dx*bw), srcy + ceil(dy*bh), //x, y
                                      floor(bw), floor(bh),                   //width height
                                      srcTotalWidth                           //totalWidth
                                      );

            dst[dy * dstTotalWidth + dx] = mean;
        }
    }
}

// Resize always to a smaller size -> downsample
__device__ void resize(uc *src, int srcx, int srcy, int srcw, int srch, int srcTotalWidth, //x,y,width,height
                       uc *dst, int dstx, int dsty, int dstw, int dsth, int dstTotalWidth) //x,y,width,height
{
    float bw = float(srcw) / dstw;
    float bh = float(srch) / dsth;

    int size = dsth * dstw;
    for(int i = threadIdx.x; i < size; i += NUM_THREADS)
    {
        int dx = dstx + (i % dstw);
        int dy = dsty + (i / dstw);

        uc mean = getWindowMeanGS(src,
                                  srcx + ceil(dx*bw), srcy + ceil(dy*bh), //x, y
                                  floor(bw), floor(bh),                   //width height
                                  srcTotalWidth                           //totalWidth
                                  );

        dst[dy * dstTotalWidth + dx] = mean;
    }
}


__device__ void getHistogram(uc *img, int ox, int oy, int width, int height, int imgWidth, float histogram[256]) {

    float npixels = width * height;
    float unitProb = 1.0f/npixels;

    if(threadIdx.x < 256) histogram[threadIdx.x] = 0;

    for(int i = threadIdx.x; i < npixels; i += NUM_THREADS)
    {
        int wx = i % width;
        int wy = i / width;
        int offset = (oy + wy) * imgWidth + (ox + wx);
        uc v = img[offset];
        atomicAdd(&histogram[v], unitProb);
    }
}

// Increase contrast
__device__ void histogramEqualization(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    __shared__ float histogram[256];
    __shared__ float accumulatedProbs[256];
    
    getHistogram(img, ox, oy, width, height, imgWidth, histogram);
    
    if(threadIdx.x == 0)
    { 
        accumulatedProbs[0] = histogram[0];
    	for(int i = 1; i < 256; ++i)
            accumulatedProbs[i] = accumulatedProbs[i-1] + histogram[i];
    }
    __syncthreads();
    
    
    int size = width * height;
    for(int i = threadIdx.x; i < size; i += NUM_THREADS)
    {
	 int wx = i % width;
	 int wy = i / width;
	 int offset = (oy + wy) * imgWidth + (ox + wx);
    	 uc v = img[offset];
         img[offset] = floor(255 * accumulatedProbs[v]);
    }
}

__device__ void toBlackAndWhite(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    int size = width * height;
    for(int i = threadIdx.x; i < size; i += NUM_THREADS)
    {
        int wx = i % width;
        int wy = i / width;
        int offset = (oy + wy) * width + (ox + wx);
        uc v = img[offset];
        img[offset] = v > 200 ? 255 : 0;
    }
}

__device__ uc getFirstStageHeuristic(uc *img) {
    int v = img[22] - (img[19]+img[20]+img[24]+img[25]+img[58])/5;
    return v < 0 ? 0 : v;
}

// Find edges in horizontal direction
__device__ void sobelEdgeDetection(uc *img, int ox, int oy,  int winWidth, int winHeight, int imgWidth, uc *sobelImg)
{
    uc threshold = 24;

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

__device__ int getSecondStageHeuristic(uc *img) {
    int sumDiff    = 0;
    int leftEye    = getWindowMeanGS(img, 2, 4, 9, 5,30);
    int rightEye   = getWindowMeanGS(img,18, 4, 9, 5,30);
    int upperNose  = getWindowMeanGS(img,11, 1, 6,13,30);
    int lowerNose  = getWindowMeanGS(img,10,15, 9, 5,30);
    int leftCheek  = getWindowMeanGS(img, 1,10, 8,10,30);
    int rightCheek = getWindowMeanGS(img,19,10, 8,10,30);
    int mouth      = getWindowMeanGS(img, 8,21,13, 5,30);

    sumDiff += leftEye;
    sumDiff += rightEye;
    sumDiff += abs(leftEye - rightEye); // simmetry

    sumDiff += 255-upperNose;
    sumDiff += abs(125-lowerNose);

    sumDiff += 255-leftCheek;
    sumDiff += 255-rightCheek;
    sumDiff += abs(leftCheek - rightCheek); // simmetry

    sumDiff += mouth; // mouth

    return sumDiff;
}

__global__ void detectFaces(uc *img, int winWidth, int winHeight, uc  *resultMatrix)
{
    int xstep = (IMG_WIDTH - winWidth) / NUM_BLOCKS + 1;
    int ystep = (IMG_HEIGHT - winHeight) / NUM_BLOCKS + 1;
    if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
    	printf("Kernel width:%i, height:%i\n", winWidth, winHeight);
        printf("steps: %d, %d\n", xstep, ystep);
    }

    // Window origin
    int x = blockIdx.x * xstep;
    int y = blockIdx.y * ystep;
    int blockId = blockIdx.y * NUM_BLOCKS + blockIdx.x;

    if(x + winWidth > IMG_WIDTH || y + winHeight > IMG_HEIGHT)
    {
        resultMatrix[blockId] = 0;
        return;
    }

    // FIRST HEURISTIC
    __shared__ uc window30x30[30*30];
    resize(img,
           x, y, winWidth, winHeight, IMG_WIDTH,
           window30x30,
           0, 0, 9, 9, 9);
    __syncthreads();

    histogramEqualization(window30x30, 0, 0, 9, 9, 9);
    __syncthreads();
    
    __shared__ uc hv1;
    if(threadIdx.x == 0) {
        hv1 = getFirstStageHeuristic(window30x30);
    }
    __syncthreads();

    if (hv1 >= THRESH_9x9)
    {
        // SECOND HEURISTIC
        __shared__ uc sobelImg[200*200];
        sobelEdgeDetection(img, x, y, winWidth, winHeight, IMG_WIDTH, sobelImg);
        __syncthreads();

	resize(sobelImg,
               0, 0, winWidth, winHeight, winWidth,
               window30x30,
               0, 0, 30, 30, 30);
        __syncthreads();

        toBlackAndWhite(window30x30, 0, 0, 30, 30, 30);
        __syncthreads();

        if(threadIdx.x == 0) {
            int hv2 = getSecondStageHeuristic(window30x30);

            if (hv2 <= THRESH_30x30)
            {
		printf("Face detected with heuristic: %i\n", hv2);
                // Save result! We detected a face yayy
                resultMatrix[blockId] = 1;
            }
            else resultMatrix[blockId] = 0;
        }
    }
    else resultMatrix[blockId] = 0;
}

void CheckCudaError(int line) {
    cudaError_t error;
    error = cudaGetLastError();
    if (error) {
	printf("(ERROR) - %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, line);
	exit(EXIT_FAILURE);
    }
}

#define CE() { CheckCudaError(__LINE__); }

int main(int argc, char** argv)
{
    cout << "Usage: " << argv[0] << " <image file name>" << endl;
    for (int i = 0; i < argc; ++i) { cout << argv[i] << endl; }

    //Read input
    FaceDetection fc(argv[1]);
    printf("image File: %s, size(%d px, %d px)\n",
	    fc.image->filename, fc.image->width(), fc.image->height());

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
    resize_seq(h_imageGSOriginal,
	       0, 0, fc.image->width(), fc.image->height(), fc.image->width(),
	       h_imageGS,
	       0, 0, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < NUM_DEVICES) { printf("Not enough GPUs\n"); exit(-1); }

    // Get device memory
    dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);
    int winWidths[] = {35, 40, 45, 50, 55, 60, 65}; //, 75, 85, 95, 105, 115, 125, 140, 150, 160, 170, 180, 190};
    int winHeights[] = {35, 40, 45, 50, 55, 60, 65}; //, 75, 85, 95, 105, 115, 125, 140, 150, 160, 170, 180, 190};

    printf("Getting memory in the host to allocate resultMatrix...\n");
    int numBytesResultMatrix = NUM_BLOCKS * NUM_BLOCKS * sizeof(uc);

    const int numWindowsWidth  = (sizeof(winWidths) / sizeof(int));
    const int numWindowsHeight = (sizeof(winHeights) / sizeof(int));
    const int numWindows = numWindowsWidth * numWindowsHeight;
    uc *h_resultMatrix[numWindows];
    for(int i = 0; i < numWindows; ++i)
	h_resultMatrix[i]= (uc*) malloc(numBytesResultMatrix);

    uc *d_imageGS[NUM_DEVICES], *d_resultMatrix[NUM_DEVICES];
    for(int i = 0; i < NUM_DEVICES; ++i)
    {
	printf("Getting memory in device %d...\n", i);
	cudaSetDevice(i);
	#ifndef USE_PINNED
	cudaMalloc((uc**)&d_imageGS[i], numBytesImage); CE();
	cudaMalloc((uc**)&d_resultMatrix[i], numBytesResultMatrix); CE();
	#else
	cudaMallocHost((uc**)&d_imageGS[i], numBytesImage); CE();
	cudaMallocHost((uc**)&d_resultMatrix[i], numBytesResultMatrix); CE();
	#endif
    }

    // Copy data from host to device, execute kernel, copy data from device to host
    for(int i = 0; i < NUM_DEVICES; ++i)
    {
	printf("Getting memory in device %d...\n", i);
	cudaSetDevice(i);
	#ifndef USE_PINNED
	cudaMalloc((uc**)&d_imageGS[i], numBytesImage); CE();
	cudaMalloc((uc**)&d_resultMatrix[i], numBytesResultMatrix); CE();
	#else
	cudaMallocHost((uc**)&d_imageGS[i], numBytesImage); CE();
	cudaMallocHost((uc**)&d_resultMatrix[i], numBytesResultMatrix); CE();
	#endif
    }

    cudaEvent_t E0, E1;
    cudaEventCreate(&E0);
    cudaEventCreate(&E1);

    float widthRatio =  float(fc.image->width())/IMG_WIDTH;
    float heightRatio =  float(fc.image->height())/IMG_HEIGHT;
    const int windowsPerDevice = numWindows / NUM_DEVICES;

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    printf("wr: %f\n", widthRatio);
    printf("hr: %f\n", heightRatio);
    printf("Num windows: %i\n", numWindows);
    printf("Windows per device: %i\n", windowsPerDevice);
    for(int i = 0; i < NUM_DEVICES; ++i)
    {
	cudaSetDevice(i);
	for(int j = 0; j < windowsPerDevice; ++j)
	{
	    int index = (i*windowsPerDevice + j);
	    int wi = index / numWindowsWidth;
	    int hi = index % numWindowsHeight;
	    printf("\n");
	    printf("index: %i\n", index);
	    printf("wi: %i, hi: %i\n", wi, hi);
	    printf("width: %i, height:%i\n", winWidths[wi], winHeights[hi]);
	    printf("Copying matrices from host to device %d...\n", i);
	    cudaMemcpyAsync(d_imageGS[i], h_imageGS, numBytesImage, cudaMemcpyHostToDevice); //CE();
	    printf("Executing kernel detectFaces on device %d...\n", i);
	    detectFaces<<<dimGrid, dimBlock>>>(d_imageGS[i], winWidths[wi] / widthRatio, winHeights[hi] / heightRatio, d_resultMatrix[i]); //CE();
	    printf("Retrieving resultMatrix from device %d to host...\n", i);
	    cudaMemcpyAsync(h_resultMatrix[index], d_resultMatrix[i], numBytesResultMatrix, cudaMemcpyDeviceToHost); //CE();
	}
    }

    for(int i = 0; i < NUM_DEVICES; ++i) { cudaSetDevice(i); cudaDeviceSynchronize(); }

    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,  E0, E1);
    printf("Kernel elapsed time: %4.6f\n", elapsedTime);

    // Process results
    for(int k = 0; k < numWindows; ++k)
    {
    for(int i = 0; i < NUM_BLOCKS; ++i)
    {
	for(int j = 0; j < NUM_BLOCKS; ++j)
	{
	    if (h_resultMatrix[k][i * NUM_BLOCKS + j] == 1) 
	    {
		int wi = k / numWindowsWidth;
		int hi = k % numWindowsHeight;
		int kernelStepWidth = (IMG_WIDTH - winWidths[wi]/widthRatio) / NUM_BLOCKS + 1;
		int kernelStepHeight = (IMG_HEIGHT - winHeights[hi]/heightRatio) / NUM_BLOCKS + 1;
		printf("Result found for size(%d,%d) in x,y: (%d,%d)\n", winWidths[wi], winHeights[hi], j, i);
		fc.resultWindows.push_back(Box(int(j * kernelStepWidth * widthRatio),
					       int(i * kernelStepHeight * heightRatio),
					       int(winWidths[wi]),
					       int(winHeights[hi])));
	    }
	}
    }
    }
    fc.saveResult();

    // Free device memory
    printf("Freeing device memory...\n");
    for(int i = 0; i < NUM_DEVICES; ++i)
    {
	cudaSetDevice(i); 
	cudaFree(d_imageGS[i]); 
	cudaFree(d_resultMatrix[i]);
    }

    printf("Done.");
}


